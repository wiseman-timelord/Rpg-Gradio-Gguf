# scripts/inference.py
# Model loading, text inference, prompt formatting, and image generation.
#
# Text model  : Qwen3-4b-Z-Image-Turbo-AbliteratedV1 (GGUF) via llama-cpp-python (Vulkan)
#               Also serves as the text encoder (llm_path) for image generation.
# Image model : Z-Image-Turbo (GGUF)  via stable-diffusion-cpp-python (Vulkan)
#               Requires ae.safetensors VAE in the same folder.
#
# GPU SELECTION
# -------------
# Both llama.cpp and stable-diffusion.cpp use the ggml Vulkan backend.
# The environment variable GGML_VK_VISIBLE_DEVICES controls which physical
# GPU the Vulkan backend binds to (same mechanism as CUDA_VISIBLE_DEVICES).
# We set it from cfg.selected_gpu before each model is initialised so the
# user's dropdown choice is respected.
#
# VRAM STRATEGY
# -------------
# The Qwen3 text model (~2.5 GB) and Z-Image-Turbo diffusion model (~4.6 GB)
# together exceed typical 8 GB VRAM budgets.  compute_vram_allocation() in
# utilities.py determines how many text-model layers to offload to the GPU
# and whether the image model should offload its parameters to system RAM.
#
# UPDATED: Images are saved into per-session folders under ./output/.
#          Session folder is created lazily on the first image of each session.

import os
import re
import uuid

from scripts import configure as cfg
from scripts.utilities import (
    scan_for_gguf,
    calculate_optimal_threads,
    compute_vram_allocation,
    ensure_vae_in_image_folder,
    generate_session_folder_name,
)


# -----------------------------------------------------------------------
# Qwen3 output-cleaning helpers
# -----------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_METADATA_RE = re.compile(
    r"\[Technical Metadata\].*",
    re.DOTALL | re.IGNORECASE,
)


def _strip_think_tags(text: str) -> str:
    """Remove Qwen3 ``<think>…</think>`` reasoning blocks from output.

    Qwen3 may emit these even in /no_think mode.  The blocks waste tokens
    and confuse downstream parsing, so we strip them everywhere.
    """
    return _THINK_RE.sub("", text).strip()


def _clean_image_prompt(raw: str) -> str:
    """Post-process the Z-Engineer image-prompt output.

    1. Strip ``<think>`` blocks.
    2. Remove ``[Technical Metadata]`` blocks appended by the model.
    3. Remove any leading label like ``Enhanced prompt:`` or ``Output:``.
    4. Collapse whitespace to a single clean paragraph.
    """
    text = _strip_think_tags(raw)
    text = _METADATA_RE.sub("", text).strip()
    # Remove common preamble labels the model sometimes adds
    text = re.sub(
        r"^(Enhanced\s+prompt|Output|Prompt|Result)\s*:\s*",
        "", text, flags=re.IGNORECASE,
    )
    # Collapse to single paragraph
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------------------------------------------------
# Internal helper
# -----------------------------------------------------------------------
def _apply_gpu_selection() -> None:
    """
    Set GGML_VK_VISIBLE_DEVICES so that both llama-cpp-python and
    stable-diffusion-cpp-python use the GPU chosen in the UI.

    Vulkan device indices reported by the ggml backend match the order
    returned by the Vulkan instance enumeration, which on Windows is the
    same order as WMIC / Windows Device Manager — i.e. the same index
    that detect_gpus() assigns.
    """
    gpu_idx = str(max(0, int(cfg.selected_gpu)))
    os.environ["GGML_VK_VISIBLE_DEVICES"] = gpu_idx
    print(f"  GPU selection applied: GGML_VK_VISIBLE_DEVICES={gpu_idx}")


def _find_text_model_path() -> str | None:
    """Return the path to the first .gguf file in the text model folder."""
    gguf_files = scan_for_gguf(cfg.text_model_folder)
    return gguf_files[0] if gguf_files else None


def _find_image_model_path() -> str | None:
    """Return the path to the first .gguf file in the image model folder."""
    gguf_files = scan_for_gguf(cfg.image_model_folder)
    return gguf_files[0] if gguf_files else None


# -----------------------------------------------------------------------
# Module-level VRAM allocation cache
# -----------------------------------------------------------------------
# Computed once when the first model loads, then reused so both models
# share a consistent plan.  Reset to None when models are unloaded or
# when settings change (new model folder, new VRAM budget, etc.).
_vram_plan: dict | None = None


def _get_vram_plan() -> dict:
    """
    Compute (or return cached) VRAM allocation for both models.

    The plan is computed based on:
      - cfg.text_model_folder (first .gguf found)
      - cfg.image_model_folder (first .gguf found)
      - cfg.vram_assigned
    """
    global _vram_plan
    if _vram_plan is not None:
        return _vram_plan

    text_path = _find_text_model_path()
    image_path = _find_image_model_path()

    _vram_plan = compute_vram_allocation(
        text_model_path=text_path,
        image_model_path=image_path,
        vram_budget_mb=cfg.vram_assigned,
    )
    print(f"VRAM allocation plan: {_vram_plan['info']}")
    return _vram_plan


def _invalidate_vram_plan() -> None:
    """Reset the cached VRAM plan so it is recomputed on next load."""
    global _vram_plan
    _vram_plan = None


# -----------------------------------------------------------------------
# Model unloading
# -----------------------------------------------------------------------
def unload_text_model() -> None:
    """Release the text model from memory and reset the loaded flag."""
    if cfg.text_model is not None:
        print("Unloading text model...")
        try:
            if hasattr(cfg.text_model, "close"):
                cfg.text_model.close()
        except Exception as e:
            print(f"  Warning during text model close: {e}")
        cfg.text_model = None
    cfg.text_model_loaded = False
    _invalidate_vram_plan()
    import gc
    gc.collect()
    print("Text model unloaded.")


def unload_image_model() -> None:
    """Release the image model from memory and reset the loaded flag."""
    if cfg.image_model is not None:
        print("Unloading image model...")
        try:
            if hasattr(cfg.image_model, "close"):
                cfg.image_model.close()
        except Exception as e:
            print(f"  Warning during image model close: {e}")
        cfg.image_model = None
    cfg.image_model_loaded = False
    _invalidate_vram_plan()
    import gc
    gc.collect()
    print("Image model unloaded.")


def ensure_models_loaded(need_image: bool = False) -> tuple[bool, str]:
    """
    Lazy-load text and image models if they are not already in memory.

    Called by chat_with_model() in displays.py before each inference run
    so the user does not have to restart the application after changing
    model folders or after an idle-timeout unload.

    Parameters
    ----------
    need_image : bool
        When True (i.e. Visualizer is set to "Z-Image-Turbo"), the image
        model is a hard requirement and a load failure will return ok=False.
        When False ("No Generation" mode), the image model is optional and
        a load failure is silently skipped.

    Returns
    -------
    (ok: bool, message: str)
        ok      — True if all required models are available.
        message — Human-readable status suitable for the UI status bar.
    """
    messages: list[str] = []

    # Text model is always required.
    if not cfg.text_model_loaded:
        print("Lazy-loading text model...")
        messages.append("Loading text model…")
        if load_text_model():
            messages.append("Text model loaded.")
        else:
            messages.append(
                "Text model failed to load — check the Model Folder path in Configuration."
            )
            return False, " ".join(messages)

    # Image model: required when Visualizer is "Z-Image-Turbo", optional otherwise.
    if not cfg.image_model_loaded:
        print("Lazy-loading image model...")
        messages.append("Loading image model…")
        if load_image_model():
            messages.append("Image model loaded.")
        elif need_image:
            # Hard failure — image generation was requested but model is unavailable.
            messages.append(
                "Image model failed to load — check the Image Model Folder in Configuration."
            )
            return False, " ".join(messages)
        else:
            # Soft failure — "No Generation" mode, image model not needed right now.
            messages.append("Image model unavailable (images will be skipped).")

    return True, " ".join(messages) if messages else "Models ready."


# -----------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------
def load_text_model() -> bool:
    """
    Scan cfg.text_model_folder for the first .gguf file and load it via
    llama-cpp-python (with Vulkan GPU offloading via smart VRAM allocation).

    Expected model: Qwen3-4b-Z-Image-Turbo-AbliteratedV1 Q4_K_M (~2.5 GB)

    This model also serves as the text encoder (llm_path) for image
    generation.  Its path is read again by load_image_model().
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python is not installed. Text model unavailable.")
        return False

    model_path = _find_text_model_path()
    if not model_path:
        print(f"No .gguf text models found in {cfg.text_model_folder}.")
        return False

    print(f"Loading text model: {model_path}")

    # Apply GPU device selection BEFORE the Vulkan instance is created
    _apply_gpu_selection()

    # Determine context length from GGUF metadata when possible
    n_ctx = 2048
    try:
        from gguf_parser import GGUFParser
        parser = GGUFParser(model_path)
        parser.parse()
        n_ctx = parser.metadata.get("llama.context_length", n_ctx)
    except Exception:
        print("GGUF metadata parser unavailable; using default n_ctx=2048.")

    threads = calculate_optimal_threads(cfg.threads_percent)

    # Smart VRAM allocation — accounts for the image model sharing GPU
    plan = _get_vram_plan()
    n_gpu = plan["text_gpu_layers"]

    print(f"  Text model GPU layers: {n_gpu} "
          f"({'all' if n_gpu == -1 else n_gpu})")

    try:
        cfg.text_model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=threads,
            n_batch=512,
            n_gpu_layers=n_gpu,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        cfg.text_model_loaded = True
        print(f"Text model ready (ctx={n_ctx}, gpu_layers={n_gpu}, "
              f"gpu_index={cfg.selected_gpu}).")
        return True
    except Exception as e:
        print(f"Failed to load text model: {e}")
        return False


def load_image_model() -> bool:
    """
    Scan cfg.image_model_folder for the first .gguf file and load it via
    stable-diffusion-cpp-python as a Z-Image-Turbo diffusion model.

    Required files:
      - Diffusion GGUF:  z_image_turbo-*.gguf  in cfg.image_model_folder
      - Text encoder:    Qwen3 .gguf           in cfg.text_model_folder  (llm_path)
      - VAE:             ae.safetensors         in cfg.image_model_folder

    The text encoder is the same Qwen3-4b model used for chat — it doubles
    as the image model's text encoder because it was specifically
    fine-tuned for Z-Image-Turbo.
    """
    try:
        from stable_diffusion_cpp import StableDiffusion
    except ImportError:
        print("stable-diffusion-cpp-python is not installed. Image model unavailable.")
        return False

    # --- Locate the diffusion GGUF -------------------------------------------
    image_model_path = _find_image_model_path()
    if not image_model_path:
        print(f"No .gguf image models found in {cfg.image_model_folder}.")
        return False

    # --- Locate the text encoder (llm_path) ----------------------------------
    llm_path = _find_text_model_path()
    if not llm_path:
        print("No text encoder GGUF found for image model. "
              "The Qwen3 text model is required as the image encoder (llm_path).")
        print(f"  Looked in: {cfg.text_model_folder}")
        return False

    # --- Locate the VAE (ae.safetensors) -------------------------------------
    vae_path = ensure_vae_in_image_folder(cfg.image_model_folder)
    if not vae_path:
        print("ae.safetensors VAE not found. Z-Image-Turbo requires this file.")
        print("  Re-run the installer (option 3) to download it, or place it "
              "manually in the image model folder.")
        return False

    print(f"Loading image model: {image_model_path}")
    print(f"  Text encoder (llm_path): {llm_path}")
    print(f"  VAE: {vae_path}")

    # Apply GPU device selection BEFORE the Vulkan instance is created
    _apply_gpu_selection()

    # Smart VRAM allocation
    plan = _get_vram_plan()
    offload_cpu = plan["image_offload_cpu"]

    print(f"  Image model offload_params_to_cpu={offload_cpu}")

    threads = calculate_optimal_threads(cfg.threads_percent)

    try:
        # Z-Image-Turbo uses llm_path (Qwen3 LLM encoder), NOT clip_l_path.
        # The Qwen3 model serves as a full LLM text encoder for the S3-DiT
        # architecture — unlike FLUX which uses a traditional CLIP encoder.
        init_kwargs = {
            "diffusion_model_path": image_model_path,
            "llm_path": llm_path,
            "vae_path": vae_path,
            "n_threads": threads,
        }

        # CPU offload if VRAM is constrained
        if offload_cpu:
            init_kwargs["offload_params_to_cpu"] = True

        cfg.image_model = StableDiffusion(**init_kwargs)
        cfg.image_model_loaded = True
        print(f"Image model ready (Z-Image-Turbo via stable-diffusion.cpp, "
              f"gpu_index={cfg.selected_gpu}).")
        return True
    except Exception as e:
        print(f"Failed to load image model: {e}")
        print("  Troubleshooting tips:")
        print("  - Ensure the GGUF is from a compatible source (e.g. leejet repo)")
        print("  - Ensure ae.safetensors is present in the image model folder")
        print("  - Ensure the Vulkan SDK is installed")
        return False


# -----------------------------------------------------------------------
# Text inference
# -----------------------------------------------------------------------
def run_text_inference(
    prompt: str,
    max_tokens: int = 2000,
    temperature: float = 0.7,
    repeat_penalty: float = 1.1,
    system_prompt: str | None = None,
) -> str:
    """Send a prompt to the loaded Qwen3 text model and return the generated text.

    Qwen3-specific handling
    -----------------------
    * A ``/no_think`` directive is prepended to the user message so the
      model skips its chain-of-thought ``<think>`` block.  This avoids
      wasting tokens on internal reasoning and prevents ``<think>`` tags
      from leaking into outputs.
    * An optional *system_prompt* is sent as a ``system`` role message,
      which Qwen3's chat template places before the user turn.
    * Any residual ``<think>`` tags are stripped from the response.
    """
    if not cfg.text_model_loaded or cfg.text_model is None:
        raise RuntimeError("Text model is not loaded.")

    try:
        messages: list[dict] = []

        # System message (optional) — Qwen3 respects this via its Jinja
        # chat template.  We use /no_think here too for belt-and-suspenders.
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt + " /no_think",
            })

        # User turn — the /no_think soft-switch at the end of the content
        # tells Qwen3 to skip its reasoning phase for this turn.
        messages.append({
            "role": "user",
            "content": prompt + " /no_think",
        })

        response = cfg.text_model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
        )
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                raw = msg.get("content", "").strip()
                return _strip_think_tags(raw)
            return "No output from model."
        return "Unexpected response format."
    except Exception as e:
        print(f"Inference error: {e}")
        return f"Error: {e}"


# -----------------------------------------------------------------------
# System prompts (Qwen3 chat-template ``system`` role)
# -----------------------------------------------------------------------
# Qwen3 places the system message before the first user turn.  These give
# the model a strong identity and task framing which dramatically improves
# output quality for each pipeline stage.
#
# The Z-Engineer system prompt is adapted from the official prompt used to
# fine-tune the Qwen3-4b-Z-Image-Turbo model.  The conversation system
# prompt leverages Qwen3's excellent roleplay / multi-turn dialogue
# capabilities.

SYSTEM_PROMPTS: dict[str, str] = {
    "converse": (
        "You are a creative roleplay writer. Your task is to write the next "
        "response for a specific named character in an ongoing scene. "
        "Stay completely in character. "
        "Write exactly one sentence of spoken dialogue followed by exactly one "
        "sentence describing a physical action performed by that character. "
        "Do not write dialogue or actions for any other character. "
        "Do not add commentary, scene headers, or break the fourth wall."
    ),
    "consolidate": (
        "You are a narrative chronicler maintaining a running third-person "
        "prose record of an ongoing roleplay scene. "
        "Write only in descriptive, observational prose — "
        "absolutely no spoken dialogue, quoted speech, or character lines. "
        "When characters speak, summarise the topic and emotional tone of "
        "the exchange (e.g. 'they discussed the nature of the mist') rather "
        "than reproducing any words. "
        "Capture character actions, scene atmosphere, mood, and any "
        "meaningful shifts in topic or situation. "
        "Never add events that did not occur."
    ),
    "instance": (
        "You are a visual scene recorder. Given a single roleplay exchange, "
        "write a compact third-person descriptive paragraph (50-80 words) "
        "capturing the physical scene of that specific moment: character "
        "positions, expressions, body language, actions, and atmosphere. "
        "Write only what can be seen — no dialogue, no quoted speech, "
        "no inner thoughts. "
        "Every character listed under 'Characters present', including the "
        "human participant, is physically in the scene and MUST be described. "
        "Describe each character once using their appearance or role label. "
        "This paragraph will feed directly into an image generation model, "
        "so make it visually precise and concrete."
    ),
    "image_prompt": (
        'You are Z-Engineer, an expert prompt engineering AI specialising '
        'in the Z-Image Turbo architecture (S3-DiT with Qwen-3 text '
        'encoder). Your goal is to rewrite scene descriptions into '
        'high-fidelity "Positive Constraint" prompts optimised for the '
        '8-step distilled inference process.\n'
        '\n'
        'CORE RULES:\n'
        '1. NO negative language — use Positive Constraints only (e.g. '
        'instead of "no blur" write "razor-sharp focus, pristine imaging").\n'
        '2. Natural language sentences — the Qwen-3 encoder requires '
        'coherent grammar. NEVER use "tag salad" (comma-separated keyword '
        'lists).\n'
        '3. Texture density — aggressively describe textures (weathered '
        'skin, visible pores, fabric weave, film grain) to avoid the '
        '"plastic" look.\n'
        '4. Spatial precision — use specific spatial prepositions '
        '(foreground, midground, background, left, right, camera height, '
        'gaze direction) to leverage 3D RoPE embeddings.\n'
        '5. Proper anatomy — for living subjects state "properly formed" '
        'or "perfectly formed" when describing body parts.\n'
        '6. Camera & lens — choose a camera and lens that suit the style, '
        'always using "shot on" or "shot with" phrasing.\n'
        '\n'
        'PROMPT STRUCTURE (follow this order):\n'
        '1. Subject anchoring (WHO / WHAT)\n'
        '2. Action & context (DOING / WHERE)\n'
        '3. Aesthetic & lighting (HOW — lighting, atmosphere, colour palette)\n'
        '4. Technical modifiers (CAMERA — lens, film stock, resolution)\n'
        '5. Positive constraints (QUALITY — clean background, architectural '
        'perfection, proper anatomy)\n'
        '\n'
        'OUTPUT: Return ONLY the enhanced prompt paragraph (120-180 words). '
        'Do NOT include any metadata, labels, or explanations.'
    ),
}


# -----------------------------------------------------------------------
# Prompt templates (inline — no external files needed)
# -----------------------------------------------------------------------
PROMPT_TEMPLATES: dict[str, str] = {
    # ── Converse ──────────────────────────────────────────────────────────
    # {characters_context}       — all active agents with roles
    # {scene_context}            — location + optional time
    # {responding_agent_name}    — Agent 1 (the speaker for this turn)
    # {responding_agent_role}    — Agent 1's role
    # {human_description}        — human name + optional decade/gender
    # {session_history}          — consolidated narrative so far
    # {human_input}              — what the human just said
    "converse": (
        "Scene characters (only these characters exist — do not invent others): "
        "{characters_context}.\n"
        "Setting: {scene_context}.\n"
        "\n"
        "You are writing for {responding_agent_name}, whose role is "
        "'{responding_agent_role}'.\n"
        "\n"
        "Story so far: {session_history}\n"
        "\n"
        "{human_description} just said: \"{human_input}\"\n"
        "\n"
        "Write {responding_agent_name}'s response now. "
        "One sentence of dialogue, then one sentence of physical action. "
        "Example: '\"How wonderful that you found us here,\" "
        "{responding_agent_name} says, tilting their head toward "
        "{human_description}.'\n"
        "\n"
        "Output only {responding_agent_name}'s single response — "
        "no other characters, no scene headers, no meta-commentary."
    ),

    # ── Consolidate ───────────────────────────────────────────────────────
    # {agent_exchange} holds the full agent response block assembled by
    # displays.py — one or more lines of "AgentName: response" depending
    # on how many agents are active.  This replaces the old single-agent
    # {responding_agent_name}: {agent_output} pattern.
    "consolidate": (
        "Update the narrative chronicle below by incorporating the new exchange.\n"
        "\n"
        "Characters: {characters_context}\n"
        "Setting: {scene_context}\n"
        "\n"
        "CURRENT CHRONICLE:\n"
        "{session_history}\n"
        "\n"
        "NEW EXCHANGE:\n"
        "{human_description}: {human_input}\n"
        "{agent_exchange}\n"
        "\n"
        "Write an updated third-person narrative chronicle (under 200 words) "
        "that covers all prior events and the new exchange above.\n"
        "\n"
        "STRICT RULES — follow every one:\n"
        "• NO dialogue, quotes, or speech reproduction of any kind.\n"
        "• Summarise what characters discussed by topic and tone only "
        "(e.g. 'they spoke of ancient paths' — never quote what was said).\n"
        "• Describe physical actions, expressions, and scene atmosphere.\n"
        "• Note any significant shift in topic, mood, or location.\n"
        "• Write in flowing prose sentences — no bullet points or lists.\n"
        "\n"
        "Updated narrative chronicle:"
    ),

    # ── Instance ──────────────────────────────────────────────────────────
    # Summarises only the most recent rotation (human_input + agent_exchange)
    # into a compact visual scene description.  Stored as
    # cfg.consolidated_instance and used as the source for image_prompt.
    #
    # Uses {characters_visual} (agents by role + human with visual anchor)
    # rather than {characters_context} (agents only) so the human participant
    # is explicitly listed as a character the model must describe.
    "instance": (
        "Describe the physical scene of this single roleplay moment "
        "as a compact third-person paragraph (50-80 words).\n"
        "\n"
        "Characters present (ALL must appear in your description):\n"
        "{characters_visual}\n"
        "Setting: {scene_context}\n"
        "\n"
        "EXCHANGE:\n"
        "{human_description}: {human_input}\n"
        "{agent_exchange}\n"
        "\n"
        "STRICT RULES:\n"
        "• Every character listed above is physically in the scene — "
        "describe each one visually, including {human_description}.\n"
        "• No dialogue or quoted speech — describe only what can be seen.\n"
        "• Mention each character ONCE using their appearance or role.\n"
        "• Describe body language, expressions, positions, and atmosphere.\n"
        "• Flowing prose only — no bullet points.\n"
        "\n"
        "Visual scene description:"
    ),

    # ── Image prompt ──────────────────────────────────────────────────────
    # Uses consolidated_instance (single-rotation visual summary) rather than
    # the full session_history.  characters_visual lists each character once
    # by appearance/role (no repeated names) to prevent phantom duplicates.
    "image_prompt": (
        "Rewrite the following scene description as a single vivid visual "
        "prompt paragraph (120-180 words) optimised for Z-Image Turbo. "
        "Describe the setting, lighting, mood, character appearances, "
        "poses, composition, textures, and camera angle. Use flowing "
        "natural sentences — no tag lists. Apply all Positive Constraint "
        "rules.\n"
        "\n"
        "Characters present (each described once by appearance): "
        "{characters_visual}\n"
        "Setting: {scene_context}\n"
        "\n"
        "Scene moment:\n"
        "{consolidated_instance}\n"
        "\n"
        "Enhanced prompt:"
    ),
}


def format_prompt(task_name: str, data: dict) -> str | None:
    """Look up the inline template for *task_name* and fill placeholders."""
    template = PROMPT_TEMPLATES.get(task_name)
    if template is None:
        print(f"No prompt template defined for task '{task_name}'.")
        return None
    try:
        return template.format(**data)
    except KeyError as e:
        print(f"Missing placeholder {e} in prompt template.")
        return None


def parse_agent_response(raw: str) -> str:
    """Clean up raw model output.

    Handles Qwen3-specific artefacts (``<think>`` blocks) and general
    formatting noise (scene headers, runaway multi-paragraph generation).
    """
    cleaned = _strip_think_tags(raw).strip()
    cleaned = re.sub(r"^###\s.*:\n", "", cleaned, flags=re.MULTILINE)
    # Take only the first meaningful paragraph to avoid runaway generation
    lines = [l for l in cleaned.split("\n") if l.strip()]
    if lines:
        cleaned = lines[0].strip()
    return cleaned


def _is_blank(val: str) -> bool:
    """Return True if a value should be treated as unset/inactive.

    Treats the following as blank (no detail to include in prompts):
      - None / empty string
      - "none"  (the literal string)
      - "0" or "-1"  (numeric sentinel values users might type)
    """
    if not val:
        return True
    return val.strip().lower() in ("", "none", "0", "-1")


def _build_characters_context() -> str:
    """
    Build a natural-language string listing all active agents.

    An agent is considered active when it has a non-blank, non-'None' name.
    Each active entry includes the agent's role if set.

    Example output:
        "Wise-Llama — A wise oracle llama and Blue-Bird — A bird speaking in songs"
    """
    entries: list[str] = []
    agent_slots = [
        (cfg.agent1_name, cfg.agent1_role),
        (cfg.agent2_name, cfg.agent2_role),
        (cfg.agent3_name, cfg.agent3_role),
    ]
    for name, role in agent_slots:
        name = (name or "").strip()
        role = (role or "").strip()
        if _is_blank(name):
            continue
        if not _is_blank(role):
            entries.append(f"{name} — {role}")
        else:
            entries.append(name)

    if not entries:
        return "No characters defined."
    if len(entries) == 1:
        return entries[0]
    return ", ".join(entries[:-1]) + f" and {entries[-1]}"


def _build_characters_visual() -> str:
    """
    Build a visual/role-based description of all active characters for use
    in the image_prompt template.

    Each character is described by their role alone (e.g. "a wise oracle
    llama", "a jovial song bird") so that the image model receives a concise
    appearance cue rather than a repeated name.  The human player is included
    last using their _build_human_description() phrase.

    This avoids repeating character names multiple times in image prompts,
    which was causing phantom duplicate characters in generated images.

    Example output:
        "a wise oracle llama, a jovial song bird, and a benevolent adventurer
        in their 30s"
    """
    entries: list[str] = []
    agent_slots = [
        (cfg.agent1_name, cfg.agent1_role),
        (cfg.agent2_name, cfg.agent2_role),
        (cfg.agent3_name, cfg.agent3_role),
    ]
    for name, role in agent_slots:
        name = (name or "").strip()
        role = (role or "").strip()
        if _is_blank(name):
            continue
        # Use the role description if available, otherwise fall back to the name
        if not _is_blank(role):
            # Lowercase and prefix with "a" / "an" only if it doesn't already
            # start with an article — keeps language natural.
            role_lower = role.lower()
            if role_lower[:2] in ("a ", "an"):
                entries.append(role_lower)
            else:
                # Pick "an" for vowel-initial words, "a" otherwise
                article = "an" if role_lower[0] in "aeiou" else "a"
                entries.append(f"{article} {role_lower}")
        else:
            entries.append(name)

    # Add the human player with a visual anchor phrase.
    # _build_human_description() returns one of:
    #   "Name in his/her/their Ns"  (age set)
    #   "Name, male/female"         (gender set, no age)
    #   "Name"                      (only name — the default case)
    #
    # For the bare-name case we wrap it as "a traveller named Name" so the
    # image model has a recognisable subject type, not just an abstract label.
    human_name = (cfg.human_name or "Human").strip()
    human_desc = _build_human_description()
    human_desc_lower = human_desc.lower()

    # If _build_human_description returned only the name (no age/gender detail),
    # augment it into "a traveller named <Name>" for a concrete visual anchor.
    if human_desc_lower == human_name.lower():
        entries.append(f"a traveller named {human_name}")
    else:
        # Age/gender were present — use the richer description as-is
        entries.append(human_desc_lower)

    if not entries:
        return "figures in the scene"
    if len(entries) == 1:
        return entries[0]
    return ", ".join(entries[:-1]) + f", and {entries[-1]}"


def _age_to_decade_phrase(age_str: str, gender: str) -> str:
    """Convert a numeric age string to a natural decade phrase.

    Gender-aware pronoun selection:
      Male   → "in his 40s"
      Female → "in her 40s"
      Other  → "in their 40s"

    Returns an empty string for invalid / non-positive ages.
    """
    try:
        age_int = int(age_str.strip())
        if age_int <= 0:
            return ""
        decade = (age_int // 10) * 10
        gender_lower = (gender or "").strip().lower()
        if gender_lower == "male":
            pronoun = "his"
        elif gender_lower == "female":
            pronoun = "her"
        else:
            pronoun = "their"
        return f"in {pronoun} {decade}s"
    except (ValueError, AttributeError):
        return ""


def _build_human_description() -> str:
    """
    Build a concise, natural-language human/player description.

    Logic:
    - If age is set (positive integer): use decade phrase which encodes
      gender implicitly via pronoun — e.g. "Benevolent-Adventurer in his 40s"
    - If age is blank but gender is set: append gender label only —
      e.g. "Benevolent-Adventurer, male"
    - If both are blank/None/0: return name only —
      e.g. "Benevolent-Adventurer"

    Blank checks include: empty string, "none", "0", "-1".
    """
    name   = (cfg.human_name   or "Human").strip()
    age    = (cfg.human_age    or "").strip()
    gender = (cfg.human_gender or "").strip()

    age_phrase = _age_to_decade_phrase(age, gender)

    if age_phrase:
        # e.g. "Benevolent-Adventurer in his 40s"
        # The pronoun already conveys gender, no need to repeat it.
        return f"{name} {age_phrase}"
    if not _is_blank(gender):
        # e.g. "Benevolent-Adventurer, male"
        return f"{name}, {gender.lower()}"
    return name


def _build_scene_context() -> str:
    """
    Build the scene/setting string, appending event time only if set.

    Example: "A misty forest clearing at 16:20" or "A misty forest clearing"
    """
    location   = (cfg.scene_location or "Unknown Location").strip()
    event_time = (cfg.event_time     or "").strip()
    if not _is_blank(event_time):
        return f"{location} at {event_time}"
    return location


def _get_responding_agent() -> tuple[str, str]:
    """
    Return ``(name, role)`` for the primary responding agent (Agent 1).
    """
    name = (cfg.agent1_name or "Agent").strip()
    role = (cfg.agent1_role or "").strip()
    if _is_blank(name):
        name = "Agent"
    return name, role


def get_active_agents() -> list[tuple[str, str]]:
    """
    Return a list of ``(name, role)`` tuples for every configured agent
    whose name is non-blank and not a sentinel value ("none", "0", "-1").

    Agents are returned in slot order (1 → 2 → 3).
    If no agents are configured, a single fallback entry is returned so
    that callers always get at least one agent to prompt.

    This is the public API used by displays.py to drive the per-agent
    converse loop.
    """
    slots = [
        (cfg.agent1_name, cfg.agent1_role),
        (cfg.agent2_name, cfg.agent2_role),
        (cfg.agent3_name, cfg.agent3_role),
    ]
    active: list[tuple[str, str]] = []
    for name, role in slots:
        name = (name or "").strip()
        role = (role or "").strip()
        if not _is_blank(name):
            active.append((name, role))

    return active if active else [("Agent", "")]


def _build_runtime_data(responding_agent: tuple[str, str] | None = None) -> dict:
    """
    Assemble the placeholder dict used by all prompt templates.

    Dynamic context strings are computed here so templates stay readable and
    any blank / 'None' fields are silently omitted from the resulting prompt.

    Parameters
    ----------
    responding_agent : (name, role) tuple, optional
        When supplied, overrides the default Agent 1 as the responding
        character.  Used by the per-agent converse loop so each active
        agent gets its own focused inference call.
    """
    characters_ctx    = _build_characters_context()
    characters_visual = _build_characters_visual()
    human_desc        = _build_human_description()
    scene_ctx         = _build_scene_context()

    if responding_agent is not None:
        r_name, r_role = responding_agent
        r_name = (r_name or "Agent").strip()
        r_role = (r_role or "").strip()
    else:
        r_name, r_role = _get_responding_agent()

    # agent_exchange is used by the consolidate template to represent the
    # full multi-agent response block.  It is set externally on cfg before
    # consolidation is triggered; we read it here with a safe fallback.
    agent_exchange = getattr(cfg, "agent_exchange", None) or (
        f"{r_name}: {cfg.agent_output}" if cfg.agent_output else ""
    )

    return {
        # Computed context strings
        "characters_context":    characters_ctx,
        "characters_visual":     characters_visual,
        "human_description":     human_desc,
        "scene_context":         scene_ctx,
        "responding_agent_name": r_name,
        "responding_agent_role": r_role,

        # Multi-agent exchange block (for consolidate / instance templates)
        "agent_exchange":        agent_exchange,

        # Legacy / convenience aliases
        "agent_name":     r_name,
        "agent_role":     r_role,
        "human_name":     cfg.human_name or "Human",
        "scene_location": cfg.scene_location or "Unknown Location",

        # Per-turn values
        "session_history":       cfg.session_history       or "No prior conversation.",
        "consolidated_instance": cfg.consolidated_instance or "",
        "human_input":           cfg.human_input           or "",
        "agent_output":          cfg.agent_output          or "",
    }


# -----------------------------------------------------------------------
# High-level conversation pipeline
# -----------------------------------------------------------------------
def prompt_response(task_name: str, responding_agent: tuple[str, str] | None = None) -> dict:
    """
    Format the prompt for *task_name*, run inference, and return a dict with
    either {'agent_response': str} or {'error': str}.

    If SYSTEM_PROMPTS contains an entry for *task_name*, it is sent as the
    ``system`` role message to Qwen3's chat template.

    Parameters
    ----------
    responding_agent : (name, role) tuple, optional
        When supplied, overrides Agent 1 as the character being prompted.
        Used by the multi-agent converse loop in displays.py so each active
        agent receives its own focused inference call.

    Returns {'cancelled': True} immediately if cfg.cancel_processing is set.
    """
    if cfg.cancel_processing:
        print(f"[{task_name}] Skipped — cancel requested.")
        return {"cancelled": True}

    settings = cfg.PROMPT_TO_SETTINGS.get(task_name)
    if not settings:
        return {"error": f"No settings for task '{task_name}'."}

    data = _build_runtime_data(responding_agent=responding_agent)
    formatted = format_prompt(task_name, data)
    if formatted is None:
        return {"error": f"Prompt template for '{task_name}' missing or invalid."}

    # Resolve system prompt (may be None → run_text_inference handles it)
    sys_prompt = SYSTEM_PROMPTS.get(task_name)

    agent_label = responding_agent[0] if responding_agent else data["responding_agent_name"]
    print(f"[{task_name}:{agent_label}] Sending prompt to model"
          f"{' (with system prompt)' if sys_prompt else ''}...")

    raw = run_text_inference(
        formatted,
        max_tokens=settings.get("max_tokens", 2000),
        temperature=settings.get("temperature", 0.7),
        repeat_penalty=settings.get("repeat_penalty", 1.1),
        system_prompt=sys_prompt,
    )

    if raw.startswith("Error:"):
        return {"error": raw}

    # Image prompts need extra cleaning (metadata stripping, etc.)
    if task_name == "image_prompt":
        parsed = _clean_image_prompt(raw)
    else:
        parsed = parse_agent_response(raw)

    print(f"[{task_name}:{agent_label}] Response received ({len(parsed)} chars).")
    return {"agent_response": parsed}


# -----------------------------------------------------------------------
# Image generation  (Z-Image-Turbo)
# -----------------------------------------------------------------------

# Module-level progress state — updated by a callback from sd.cpp during
# image generation.  Polled by the Gradio generator in displays.py.
_img_progress: dict = {"step": 0, "total": 0}


def _img_progress_callback(step: int, steps: int, time_taken: float) -> None:
    """Called by stable-diffusion-cpp-python at each denoising step."""
    _img_progress["step"] = step
    _img_progress["total"] = steps


def get_image_gen_progress() -> tuple[int, int]:
    """Return ``(current_step, total_steps)`` for the running image job."""
    return _img_progress["step"], _img_progress["total"]


def _ensure_session_folder() -> str:
    """
    Lazily create (or return) the per-session output folder.

    The folder name is derived from the current session_history text.
    Once created it is cached in cfg.session_folder for the remainder of
    the session.
    """
    if cfg.session_folder and os.path.isdir(cfg.session_folder):
        return cfg.session_folder

    history_text = cfg.session_history or cfg.default_history or "session"
    cfg.session_folder = generate_session_folder_name(history_text)
    return cfg.session_folder


def generate_image(scene_prompt: str | None = None) -> str | None:
    """
    Generate an image from the current session context using Z-Image-Turbo.

    Pipeline:
    1. If *scene_prompt* is ``None`` and the text model is loaded, distil
       the session into a visual prompt via the ``image_prompt`` task.
    2. Call stable-diffusion-cpp-python's ``generate_image()`` with
       Z-Image-Turbo params.
    3. Save the resulting PIL image to the per-session folder in ``./output/``.

    Z-Image-Turbo notes:
    - CFG scale must be **1.0** for distilled models (0.0 = unconditioned,
      the image will ignore the prompt entirely).
    - Negative prompts are mostly ignored but passed through if set.
    - Optimised for 8 NFEs (steps).
    - Resolutions should be multiples of 64.

    Returns the saved image path, or None on failure.
    """
    if not cfg.image_model_loaded or cfg.image_model is None:
        print("Image model not loaded, skipping generation.")
        return None

    if cfg.cancel_processing:
        print("Image generation skipped — cancel requested.")
        return None

    # --- Build a visual prompt ---------------------------------------------------
    visual_prompt = scene_prompt
    if visual_prompt is None and cfg.text_model_loaded:
        result = prompt_response("image_prompt")
        if "agent_response" in result:
            visual_prompt = result["agent_response"]

    if not visual_prompt:
        visual_prompt = f"A scene at {cfg.scene_location}."

    # --- Parse image dimensions --------------------------------------------------
    size_str = cfg.IMAGE_SIZE_OPTIONS.get("selected_size", "768x1024")
    try:
        width, height = map(int, size_str.split("x")[:2])
    except ValueError:
        width, height = 768, 1024

    # Z-Image-Turbo works well from 256 upward; round to nearest multiple of 64.
    MIN_SIZE = 256
    width  = max(MIN_SIZE, (width  // 64) * 64)
    height = max(MIN_SIZE, (height // 64) * 64)
    if width != int(size_str.split("x")[0]) or height != int(size_str.split("x")[1]):
        print(f"  Size adjusted from {size_str} → {width}x{height} "
              f"(minimum {MIN_SIZE}x{MIN_SIZE}; must be multiple of 64).")

    steps     = cfg.selected_steps if cfg.selected_steps in cfg.STEPS_OPTIONS else 8
    method    = cfg.selected_sample_method
    cfg_scale = cfg.selected_cfg_scale
    negative  = cfg.default_negative_prompt

    # Safety: cfg_scale=0 triggers "unconditioned mode" in sd.cpp,
    # which makes the image ignore the prompt completely.
    if cfg_scale < 0.5:
        print(f"  WARNING: cfg_scale={cfg_scale} would trigger unconditioned mode. "
              f"Overriding to 1.0 for Z-Image-Turbo.")
        cfg_scale = 1.0

    print(f"Generating image: {width}x{height}, {steps} steps, "
          f"cfg={cfg_scale}, method={method}")
    print(f"  Prompt: {visual_prompt[:150]}")
    if negative:
        print(f"  Negative: {negative[:80]}")

    # Reset progress tracking
    _img_progress["step"] = 0
    _img_progress["total"] = steps

    try:
        gen_kwargs = {
            "prompt": visual_prompt,
            "width": width,
            "height": height,
            "sample_steps": steps,
            "sample_method": method,
            "cfg_scale": cfg_scale,
            "progress_callback": _img_progress_callback,
        }
        # Only pass negative prompt if user provided one
        if negative:
            gen_kwargs["negative_prompt"] = negative

        output_images = cfg.image_model.generate_image(**gen_kwargs)
    except TypeError as te:
        # progress_callback might not be supported in older versions;
        # retry without it, then try txt_to_img fallback.
        if "progress_callback" in str(te):
            gen_kwargs.pop("progress_callback", None)
            try:
                output_images = cfg.image_model.generate_image(**gen_kwargs)
            except Exception as e2:
                print(f"Image generation failed: {e2}")
                return None
        else:
            # Fallback: older API versions may use txt_to_img() instead
            try:
                gen_kwargs.pop("progress_callback", None)
                output_images = cfg.image_model.txt_to_img(**gen_kwargs)
            except Exception as e:
                print(f"Image generation failed (txt_to_img fallback): {e}")
                return None
    except Exception as e:
        print(f"Image generation failed: {e}")
        return None

    # Mark progress as complete
    _img_progress["step"] = steps

    # --- Save the result to per-session folder -----------------------------------
    session_dir = _ensure_session_folder()
    image_path = os.path.join(session_dir, f"{uuid.uuid4().hex}.png")
    try:
        output_images[0].save(image_path)
        print(f"Image saved: {image_path}")
        # Track in session image list for Sequence panel
        cfg.session_image_paths.append(image_path)
        return image_path
    except Exception as e:
        print(f"Failed to save image: {e}")
        return None