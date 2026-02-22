# scripts/inference.py
# Model loading, text inference, prompt formatting, and image generation.
#
# Text model  : Qwen3-4B-abliterated (GGUF)  via llama-cpp-python  (Vulkan)
# Image model : WAI-NSFW-Illustrious SDXL     via stable-diffusion-cpp-python (Vulkan)
#
# GPU SELECTION
# -------------
# Both llama.cpp and stable-diffusion.cpp use the ggml Vulkan backend.
# The environment variable GGML_VK_VISIBLE_DEVICES controls which physical
# GPU the Vulkan backend binds to (same mechanism as CUDA_VISIBLE_DEVICES).
# We set it from cfg.selected_gpu before each model is initialised so the
# user's dropdown choice is respected.

import os
import re
import uuid

from scripts import configure as cfg
from scripts.utilities import (
    scan_for_gguf,
    calculate_optimal_threads,
    estimate_gpu_layers,
)


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
    import gc
    gc.collect()
    print("Image model unloaded.")


def ensure_models_loaded() -> tuple[bool, str]:
    """
    Lazy-load text and image models if they are not already in memory.

    Called by chat_with_model() in displays.py before each inference run
    so the user does not have to restart the application after changing
    model folders or after an idle-timeout unload.

    Returns
    -------
    (ok: bool, message: str)
        ok      — True if the text model is available (minimum requirement).
        message — Human-readable status suitable for the UI status bar.
    """
    messages: list[str] = []

    if not cfg.text_model_loaded:
        print("Lazy-loading text model...")
        messages.append("Loading text model…")
        if load_text_model():
            messages.append("Text model loaded.")
        else:
            messages.append("Text model failed to load — check the Model Folder path in Configuration.")
            return False, " ".join(messages)

    if not cfg.image_model_loaded:
        print("Lazy-loading image model...")
        messages.append("Loading image model…")
        if load_image_model():
            messages.append("Image model loaded.")
        else:
            # Image model is optional — warn but don't block chat
            messages.append("Image model unavailable (images will be skipped).")

    return True, " ".join(messages) if messages else "Models ready."


# -----------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------
def load_text_model() -> bool:
    """
    Scan cfg.text_model_folder for the first .gguf file and load it via
    llama-cpp-python (with Vulkan GPU offloading based on cfg.vram_assigned).

    Expected model: Qwen3-4B-abliterated Q4_K_M (~2-3 GB)
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python is not installed. Text model unavailable.")
        return False

    gguf_files = scan_for_gguf(cfg.text_model_folder)
    if not gguf_files:
        print(f"No .gguf text models found in {cfg.text_model_folder}.")
        return False

    model_path = gguf_files[0]
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
    n_gpu = estimate_gpu_layers(model_path, cfg.vram_assigned)

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
    stable-diffusion-cpp-python.

    Expected model: waiNSFWIllustrious v14.0 Q8_0 SDXL (~2.7 GB GGUF)
    - SDXL architecture: use model_path (not diffusion_model_path)
    - Integrated VAE: no separate vae_path needed
    - The library auto-detects SD1.x / SD2.x / SDXL from the model file
    """
    try:
        from stable_diffusion_cpp import StableDiffusion
    except ImportError:
        print("stable-diffusion-cpp-python is not installed. Image model unavailable.")
        return False

    gguf_files = scan_for_gguf(cfg.image_model_folder)
    if not gguf_files:
        print(f"No .gguf image models found in {cfg.image_model_folder}.")
        return False

    model_path = gguf_files[0]
    print(f"Loading image model: {model_path}")

    # Apply GPU device selection BEFORE the Vulkan instance is created
    _apply_gpu_selection()

    # Check for optional separate VAE file in the same folder.
    # SDXL models sometimes benefit from a fixed FP16 VAE
    # (e.g. sdxl_vae.safetensors from madebyollin/sdxl-vae-fp16-fix).
    # If present, pass it; otherwise let the integrated VAE handle it.
    vae_path = _find_vae_file(cfg.image_model_folder)
    if vae_path:
        print(f"  Using separate VAE: {vae_path}")

    try:
        init_kwargs = {
            "model_path": model_path,
            # vae_decode_only=True is intentionally omitted — it leaves the VAE
            # encoder uninitialized in stable-diffusion.cpp, causing a null-ptr
            # access violation even for txt2img-only use.
        }
        if vae_path:
            init_kwargs["vae_path"] = vae_path

        cfg.image_model = StableDiffusion(**init_kwargs)
        cfg.image_model_loaded = True
        print(f"Image model ready (SDXL via stable-diffusion.cpp, "
              f"gpu_index={cfg.selected_gpu}).")
        return True
    except Exception as e:
        print(f"Failed to load image model: {e}")
        return False


def _find_vae_file(directory: str) -> str | None:
    """
    Look for a VAE file (.safetensors or .gguf containing 'vae' in name)
    inside the image model folder.  Returns the path or None.
    """
    if not os.path.isdir(directory):
        return None
    for fname in os.listdir(directory):
        lower = fname.lower()
        if "vae" in lower and (lower.endswith(".safetensors") or lower.endswith(".gguf")):
            return os.path.join(directory, fname)
    return None


# -----------------------------------------------------------------------
# Text inference
# -----------------------------------------------------------------------
def run_text_inference(
    prompt: str,
    max_tokens: int = 2000,
    temperature: float = 0.7,
    repeat_penalty: float = 1.1,
) -> str:
    """Send a prompt to the loaded text model and return the generated text."""
    if not cfg.text_model_loaded or cfg.text_model is None:
        raise RuntimeError("Text model is not loaded.")

    try:
        # Use the chat completion API so the model's instruct template is applied.
        # Calling raw completion on an instruct-tuned model causes it to echo
        # instructions rather than respond to them.
        response = cfg.text_model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
        )
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                return msg.get("content", "").strip()
            return "No output from model."
        return "Unexpected response format."
    except Exception as e:
        print(f"Inference error: {e}")
        return f"Error: {e}"


# -----------------------------------------------------------------------
# Prompt templates (inline — no external files needed)
# -----------------------------------------------------------------------
PROMPT_TEMPLATES: dict[str, str] = {
    "converse": (
        "Your name is {agent_name}, and you are in the role as the '{agent_role}'. "
        "The location is '{scene_location}', where {agent_name} and {human_name} are present, "
        "and the events so far are, '{session_history}'. "
        "Just now, {human_name} just said '{human_input}' to {agent_name} (you). "
        "Your task is, in relevance to what {human_name} just said to you, "
        "respond to {human_name} with one sentence of dialogue followed by a one-sentence "
        "description of an action you take, "
        "for example, '\"I'm delighted to see you here, it's quite an unexpected pleasure\", "
        "{agent_name} says as he offers a warm smile to {human_name}.'. "
        "Only output {agent_name}'s response — no scene headers, no meta-commentary."
    ),
    "consolidate": (
        "Below is a roleplay conversation. Summarise it into a concise "
        "third-person narrative paragraph that preserves all important events, "
        "character actions, and scene details. Keep the summary under 200 words. "
        "Do not add new events.\n"
        "\n"
        "Conversation:\n"
        "{session_history}\n"
        "\n"
        "{human_name}: {human_input}\n"
        "{agent_name}: {agent_output}\n"
        "\n"
        "Summary:"
    ),
    "image_prompt": (
        "Read the scene description below and produce a single short visual "
        "prompt (under 30 words) suitable for an anime-style image generator. "
        "Focus on the setting, lighting, mood, and character poses. "
        "Do not include character dialogue. Use danbooru-style tags if possible.\n"
        "\n"
        "Scene:\n"
        "{session_history}\n"
        "\n"
        "Visual prompt:"
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
    """Clean up raw model output."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^###\s.*:\n", "", cleaned, flags=re.MULTILINE)
    # Take only the first meaningful paragraph to avoid runaway generation
    lines = [l for l in cleaned.split("\n") if l.strip()]
    if lines:
        cleaned = lines[0].strip()
    return cleaned


def _build_runtime_data() -> dict:
    """Assemble the placeholder dict used by prompt templates."""
    return {
        "agent_name":     cfg.agent_name,
        "agent_role":     cfg.agent_role,
        "human_name":     cfg.human_name,
        "session_history": cfg.session_history or "No prior conversation.",
        "human_input":    cfg.human_input or "",
        "agent_output":   cfg.agent_output or "",
        "scene_location": cfg.scene_location or "Unknown Location",
    }


# -----------------------------------------------------------------------
# High-level conversation pipeline
# -----------------------------------------------------------------------
def prompt_response(task_name: str) -> dict:
    """
    Format the prompt for *task_name*, run inference, and return a dict with
    either {'agent_response': str} or {'error': str}.
    """
    settings = cfg.PROMPT_TO_SETTINGS.get(task_name)
    if not settings:
        return {"error": f"No settings for task '{task_name}'."}

    data = _build_runtime_data()
    formatted = format_prompt(task_name, data)
    if formatted is None:
        return {"error": f"Prompt template for '{task_name}' missing or invalid."}

    print(f"[{task_name}] Sending prompt to model...")

    raw = run_text_inference(
        formatted,
        max_tokens=settings.get("max_tokens", 2000),
        temperature=settings.get("temperature", 0.7),
        repeat_penalty=settings.get("repeat_penalty", 1.1),
    )

    if raw.startswith("Error:"):
        return {"error": raw}

    parsed = parse_agent_response(raw)
    print(f"[{task_name}] Response received ({len(parsed)} chars).")
    return {"agent_response": parsed}


# -----------------------------------------------------------------------
# Image generation
# -----------------------------------------------------------------------
def generate_image(scene_prompt: str | None = None) -> str | None:
    """
    Generate an image from the current session context using the SDXL model.

    Pipeline:
    1. If the text model is loaded, distil the session into a visual prompt.
    2. Call stable-diffusion-cpp-python's generate_image() with SDXL params.
    3. Save the resulting PIL image to ./generated/.

    Returns the saved image path, or None on failure.
    """
    if not cfg.image_model_loaded or cfg.image_model is None:
        print("Image model not loaded, skipping generation.")
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
    size_str = cfg.IMAGE_SIZE_OPTIONS.get("selected_size", "768x768")
    try:
        width, height = map(int, size_str.split("x")[:2])
    except ValueError:
        width, height = 768, 768

    # SDXL uses a VAE with 8x downscaling.  At 256×256 the latent space is
    # only 32×32, which falls below SDXL's minimum attention resolution and
    # causes a null-pointer access violation inside stable-diffusion.cpp.
    # Enforce a safe floor of 512×512; also round to nearest multiple of 64
    # as required by the SDXL architecture.
    MIN_SIZE = 512
    width  = max(MIN_SIZE, (width  // 64) * 64)
    height = max(MIN_SIZE, (height // 64) * 64)
    if width != int(size_str.split("x")[0]) or height != int(size_str.split("x")[1]):
        print(f"  Size adjusted from {size_str} → {width}x{height} "
              f"(SDXL minimum is {MIN_SIZE}x{MIN_SIZE}; must be multiple of 64).")

    steps     = cfg.selected_steps if cfg.selected_steps in cfg.STEPS_OPTIONS else 8
    method    = cfg.selected_sample_method
    cfg_scale = cfg.selected_cfg_scale
    negative  = cfg.default_negative_prompt

    print(f"Generating image: {width}x{height}, {steps} steps, "
          f"cfg={cfg_scale}, method={method}")
    print(f"  Prompt: {visual_prompt[:120]}")
    if negative:
        print(f"  Negative: {negative[:80]}")

    try:
        output_images = cfg.image_model.generate_image(
            prompt=visual_prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            sample_steps=steps,
            sample_method=method,
            cfg_scale=cfg_scale,
        )
    except TypeError:
        # Fallback: older versions may use txt_to_img() instead
        try:
            output_images = cfg.image_model.txt_to_img(
                prompt=visual_prompt,
                negative_prompt=negative,
                width=width,
                height=height,
                sample_steps=steps,
                sample_method=method,
                cfg_scale=cfg_scale,
            )
        except Exception as e:
            print(f"Image generation failed (txt_to_img fallback): {e}")
            return None
    except Exception as e:
        print(f"Image generation failed: {e}")
        return None

    # --- Save the result ---------------------------------------------------------
    os.makedirs("./generated", exist_ok=True)
    image_path = os.path.join("./generated", f"{uuid.uuid4().hex}.png")
    try:
        output_images[0].save(image_path)
        print(f"Image saved: {image_path}")
        return image_path
    except Exception as e:
        print(f"Failed to save image: {e}")
        return None