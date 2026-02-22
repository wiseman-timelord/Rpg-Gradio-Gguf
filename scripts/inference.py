# scripts/inference.py
# Model loading, text inference, prompt formatting, and image generation.

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
# Model loading
# -----------------------------------------------------------------------
def load_text_model() -> bool:
    """
    Scan cfg.text_model_folder for the first .gguf file and load it via
    llama-cpp-python (with Vulkan GPU offloading based on cfg.vram_assigned).
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
        print(f"Text model ready (ctx={n_ctx}, gpu_layers={n_gpu}).")
        return True
    except Exception as e:
        print(f"Failed to load text model: {e}")
        return False


def load_image_model() -> bool:
    """
    Scan cfg.image_model_folder for the first .gguf file and load it via
    stable-diffusion-cpp-python.
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

    try:
        cfg.image_model = StableDiffusion(
            model_path=model_path,
            wtype="default",
        )
        cfg.image_model_loaded = True
        print("Image model ready.")
        return True
    except Exception as e:
        print(f"Failed to load image model: {e}")
        return False


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
        response = cfg.text_model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
        )
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("text", "").strip()
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
        "You are {agent_name}, {agent_role}.\n"
        "You are located at: {scene_location}.\n"
        "You are speaking with {human_name}.\n"
        "\n"
        "Conversation so far:\n"
        "{session_history}\n"
        "\n"
        "{human_name}: {human_input}\n"
        "\n"
        "Respond in character as {agent_name}. Stay in character and keep your "
        "response concise (2-4 sentences). Do not break character or add "
        "meta-commentary."
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
        "prompt (under 30 words) suitable for an image generator. Focus on the "
        "setting, lighting, and mood. Do not include character dialogue.\n"
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
        "agent_name": cfg.agent_name,
        "agent_role": cfg.agent_role,
        "human_name": cfg.human_name,
        "session_history": cfg.session_history or "No prior conversation.",
        "human_input": cfg.human_input or "",
        "agent_output": cfg.agent_output or "",
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
    Generate an image from the current session context.
    If the text model is loaded, it first creates a visual prompt from the
    session history.  Otherwise falls back to the raw scene_prompt or a
    placeholder sentence.
    Returns the saved image path, or None on failure.
    """
    if not cfg.image_model_loaded or cfg.image_model is None:
        print("Image model not loaded, skipping generation.")
        return None

    # --- Build a visual prompt ---------------------------------------------------
    visual_prompt = scene_prompt
    if visual_prompt is None and cfg.text_model_loaded:
        # Let the LLM distil the session into a short image description
        result = prompt_response("image_prompt")
        if "agent_response" in result:
            visual_prompt = result["agent_response"]

    if not visual_prompt:
        visual_prompt = f"A scene at {cfg.scene_location}."

    # --- Parse image dimensions --------------------------------------------------
    size_str = cfg.IMAGE_SIZE_OPTIONS.get("selected_size", "256x256")
    try:
        width, height = map(int, size_str.split("x")[:2])
    except ValueError:
        width, height = 256, 256

    steps = cfg.selected_steps if cfg.selected_steps in cfg.STEPS_OPTIONS else 8
    method = cfg.selected_sample_method

    print(f"Generating image: {width}x{height}, {steps} steps, method={method}")
    print(f"  Prompt: {visual_prompt[:120]}")

    try:
        output_images = cfg.image_model.txt_to_img(
            prompt=visual_prompt,
            width=width,
            height=height,
            sample_steps=steps,
            sample_method=method,
        )
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
