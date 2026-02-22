# scripts/configure.py
# Central configuration: runtime globals, option lists, JSON persistence.

import os
import json

# ---------------------------------------------------------------------------
# Runtime state  (mutable at runtime, not saved automatically)
# ---------------------------------------------------------------------------
agent_name: str = "Wise-Llama"
agent_role: str = "A wise oracle who speaks in riddles and metaphors"
human_name: str = "Adventurer"
scene_location: str = "A misty forest clearing at dawn"
session_history: str = "The conversation started."
human_input: str = ""
agent_output: str = ""
rotation_counter: int = 0
latest_image_path: str | None = None

# ---------------------------------------------------------------------------
# Shutdown callback — set by launcher.py at startup.
# displays.py calls this when the user clicks "Exit Program".
# Using a callback avoids circular imports (launcher → displays → launcher).
# ---------------------------------------------------------------------------
shutdown_fn = None

# ---------------------------------------------------------------------------
# Hardware / threading
# ---------------------------------------------------------------------------
CPU_THREADS: int = os.cpu_count() or 4          # Total system threads (constant)
threads_percent: int = 80                        # Legacy — kept for compat
optimal_threads: int = max(1, CPU_THREADS * 80 // 100)

# New hardware selection state
selected_gpu: int = 0
selected_cpu: int = 0
cpu_threads: int = max(1, CPU_THREADS * 80 // 100)  # Absolute thread count
auto_unload: bool = False
max_memory_percent: int = 85

# Populated at startup by utilities.detect_gpus / detect_cpus
DETECTED_GPUS: list[dict] = []
DETECTED_CPUS: list[dict] = []

# ---------------------------------------------------------------------------
# Model references  (set after loading)
# ---------------------------------------------------------------------------
text_model = None          # Llama instance
image_model = None         # StableDiffusion instance
text_model_loaded: bool = False
image_model_loaded: bool = False

# ---------------------------------------------------------------------------
# Paths (user-configurable)
# ---------------------------------------------------------------------------
text_model_folder: str = "./models/text"
image_model_folder: str = "./models/image"

# ---------------------------------------------------------------------------
# VRAM budget (MB) — user picks from dropdown, passed to model loader
# ---------------------------------------------------------------------------
vram_assigned: int = 8192
VRAM_OPTIONS: list[int] = [2048, 4096, 6144, 8192, 10240, 12288, 16384, 24576]

# ---------------------------------------------------------------------------
# Image generation options
# ---------------------------------------------------------------------------
# SDXL-appropriate sizes — native resolution is 1024x1024 but smaller
# sizes are provided for faster generation on limited hardware.
IMAGE_SIZE_OPTIONS: dict = {
    "available_sizes": [
        "256x256", "384x384", "512x512",
        "512x768", "768x512", "768x768",
        "768x1024", "1024x768", "1024x1024",
    ],
    "selected_size": "768x768",
}

STEPS_OPTIONS: list[int] = [2, 4, 6, 8, 10, 15, 20, 25, 30]
selected_steps: int = 8

SAMPLE_METHOD_OPTIONS: list[str] = [
    "euler", "euler_a", "heun", "dpm2",
    "dpm++2s_a", "dpm++2m", "dpm++2mv2", "lcm",
]
selected_sample_method: str = "euler_a"

# CFG scale — how strongly the image follows the prompt.
# SDXL typically uses 5–7 for best results.
CFG_SCALE_OPTIONS: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
selected_cfg_scale: float = 5.0

# Default negative prompt for image generation (SDXL benefits from this)
default_negative_prompt: str = (
    "low quality, blurry, distorted, deformed, ugly, bad anatomy, "
    "watermark, text, signature"
)

# ---------------------------------------------------------------------------
# Prompt-to-inference-settings map
# ---------------------------------------------------------------------------
PROMPT_TO_SETTINGS: dict = {
    "converse": {
        "temperature": 0.7,
        "repeat_penalty": 1.1,
        "max_tokens": 2000,
    },
    "consolidate": {
        "temperature": 0.5,
        "repeat_penalty": 1.0,
        "max_tokens": 1500,
    },
    "image_prompt": {
        "temperature": 0.6,
        "repeat_penalty": 1.0,
        "max_tokens": 80,
    },
}

# ---------------------------------------------------------------------------
# JSON persistence helpers
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(".", "data", "persistent.json")


def load_config(path: str | None = None) -> dict:
    """Load persistent.json and update module-level globals. Returns the dict."""
    global agent_name, agent_role, human_name, scene_location, session_history
    global threads_percent, optimal_threads, cpu_threads
    global selected_gpu, selected_cpu, auto_unload, max_memory_percent
    global text_model_folder, image_model_folder, vram_assigned
    global selected_steps, selected_sample_method, selected_cfg_scale
    global default_negative_prompt

    path = path or CONFIG_PATH
    if not os.path.exists(path):
        print(f"Config not found at {path}, using defaults.")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading config: {e}")
        return {}

    # --- Conversation / Personalize panel ---
    agent_name       = data.get("agent_name",       agent_name)
    agent_role       = data.get("agent_role",       agent_role)
    human_name       = data.get("human_name",       human_name)
    scene_location   = data.get("scene_location",   scene_location)
    session_history  = data.get("session_history",  session_history)

    # --- Model paths ---
    text_model_folder  = data.get("text_model_folder",  text_model_folder)
    image_model_folder = data.get("image_model_folder", image_model_folder)

    # --- VRAM budget ---
    vram_assigned = data.get("vram_assigned", vram_assigned)

    # --- Image generation settings ---
    IMAGE_SIZE_OPTIONS["selected_size"] = data.get(
        "image_size", IMAGE_SIZE_OPTIONS["selected_size"]
    )
    selected_steps         = data.get("image_steps",    selected_steps)
    selected_sample_method = data.get("sample_method",  selected_sample_method)
    selected_cfg_scale     = data.get("cfg_scale",      selected_cfg_scale)
    default_negative_prompt = data.get("negative_prompt", default_negative_prompt)

    # --- Hardware / threading ---
    selected_gpu       = data.get("selected_gpu",       selected_gpu)
    selected_cpu       = data.get("selected_cpu",       selected_cpu)
    auto_unload        = data.get("auto_unload",        auto_unload)
    max_memory_percent = data.get("max_memory_percent", max_memory_percent)

    # cpu_threads is the authoritative absolute thread count.
    # A stored value of 0 means "not yet set" (installer default), so we
    # fall back to deriving from threads_percent in that case.
    # Legacy configs that only have threads_percent also use the fallback.
    threads_percent = data.get("threads_percent", threads_percent)
    raw_cpu_threads = data.get("cpu_threads", 0)
    if raw_cpu_threads and raw_cpu_threads > 0:
        cpu_threads = max(1, int(raw_cpu_threads))
    else:
        cpu_threads = max(1, (CPU_THREADS * threads_percent) // 100)

    # Keep optimal_threads in sync with cpu_threads
    optimal_threads = cpu_threads

    print(f"Config loaded from {path}.")
    return data


def save_config(path: str | None = None) -> None:
    """Persist current globals to persistent.json."""
    path = path or CONFIG_PATH
    data = {
        # --- Conversation / Personalize panel ---
        "agent_name":      agent_name,
        "agent_role":      agent_role,
        "human_name":      human_name,
        "scene_location":  scene_location,
        "session_history": session_history,

        # --- Model paths ---
        "text_model_folder":  text_model_folder,
        "image_model_folder": image_model_folder,

        # --- VRAM budget ---
        "vram_assigned": vram_assigned,

        # --- Image generation settings ---
        "image_size":      IMAGE_SIZE_OPTIONS["selected_size"],
        "image_steps":     selected_steps,
        "sample_method":   selected_sample_method,
        "cfg_scale":       selected_cfg_scale,
        "negative_prompt": default_negative_prompt,

        # --- Hardware / threading ---
        "selected_gpu":       selected_gpu,
        "selected_cpu":       selected_cpu,
        "cpu_threads":        cpu_threads,
        "threads_percent":    threads_percent,    # kept for legacy round-trips
        "auto_unload":        auto_unload,
        "max_memory_percent": max_memory_percent,
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print("Configuration saved.")
    except Exception as e:
        print(f"Error saving config: {e}")