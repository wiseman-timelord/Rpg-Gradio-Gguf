# scripts/configure.py
# Central configuration: runtime globals, option lists, JSON persistence.
#
# UPDATED for Z-Image-Turbo + Qwen3-4b-Z-Image-Turbo-AbliteratedV1:
#   - cfg_scale default → 1.0 (sd.cpp distilled models need cfg=1.0;
#     0.0 triggers unconditioned mode which ignores the prompt entirely)
#   - Negative prompt default → "" (Z-Image-Turbo ignores negatives)
#   - Steps default → 8 (Turbo is optimised for 8 NFEs)
#   - Sample method default → "euler" (stable for Z-Image-Turbo)
#   - Image sizes include larger options suitable for Z-Image
#
# UPDATED: default_history / session_history separation
#   - default_history  → the starting-point template saved in JSON / Personalize
#   - session_history   → the running consolidated narrative (mutated each turn)
#   - Session image tracking and per-session output folders

import os
import json
import time

# ---------------------------------------------------------------------------
# Runtime state  (mutable at runtime, not saved automatically)
# ---------------------------------------------------------------------------

# Idle-unload settings
# user_turn_start_time is set (to time.time()) every time control returns to
# the user after a model response.  The idle-watcher thread in launcher.py
# checks this value and unloads the models if the user has been idle for
# longer than IDLE_UNLOAD_SECONDS.  It is reset to None while the model is
# processing (not user's turn) so we never unload mid-generation.
IDLE_UNLOAD_SECONDS: int = 15 * 60          # 15 minutes
user_turn_start_time: float | None = None   # None  → not user's turn yet

agent_name: str = "Wise-Llama"
agent_role: str = "A wise oracle who speaks in riddles and metaphors"
human_name: str = "Adventurer"
scene_location: str = "A misty forest clearing at dawn"

# default_history is the starting-point template.  It is saved to / loaded
# from persistent.json and shown in the "Default History" box on the
# Personalize panel.  It is NOT mutated during conversation.
default_history: str = (
    "The two roleplayers approached one another, and the conversation started."
)

# session_history is the running consolidated narrative.  Each turn the
# consolidation prompt rewrites it.  Shown in "Consolidated History".
session_history: str = ""

scenario_log: str = ""   # Running per-session dialogue log shown in Scenario Log box
human_input: str = ""
agent_output: str = ""
rotation_counter: int = 0
latest_image_path: str | None = None

# Per-session image tracking  (for Sequence panel in Chronicler)
session_image_paths: list[str] = []    # all images generated this session
session_folder: str | None = None      # e.g. "./output/misty_forest_cle"

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
# Image generation options  (tuned for Z-Image-Turbo)
# ---------------------------------------------------------------------------
# Z-Image-Turbo supports a wide range of resolutions.  Multiples of 64.
# It works well at 512-1024 range; 768x1024 is a sweet spot for portraits.
IMAGE_SIZE_OPTIONS: dict = {
    "available_sizes": [
        "512x512",
        "512x768", "768x512", "768x768",
        "768x1024", "1024x768", "1024x1024",
    ],
    "selected_size": "768x1024",
}

# Z-Image-Turbo is a distilled model optimised for low step counts (8 NFEs).
STEPS_OPTIONS: list[int] = [4, 6, 8, 10, 12, 15, 20]
selected_steps: int = 8

SAMPLE_METHOD_OPTIONS: list[str] = [
    "euler", "euler_a", "heun", "dpm2",
    "dpm++2s_a", "dpm++2m", "dpm++2mv2", "lcm",
]
selected_sample_method: str = "euler"

# CFG scale — Z-Image-Turbo (distilled/Turbo models) should use 0.0.
# Higher values are kept as options for experimentation.
CFG_SCALE_OPTIONS: list[float] = [1.0, 1.5, 2.0, 3.0, 5.0, 7.0]
selected_cfg_scale: float = 1.0

# Z-Image-Turbo mostly ignores negative prompts, so default is empty.
# Users can still enter one for experimentation.
default_negative_prompt: str = ""

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
        "max_tokens": 350,
    },
}

# ---------------------------------------------------------------------------
# Default RP settings — used by "Restore RP Defaults" to reset to
# the values the installer would have written to persistent.json.
# ---------------------------------------------------------------------------
DEFAULT_RP_SETTINGS: dict = {
    "agent_name":     "Wise-Llama",
    "agent_role":     "A wise oracle who speaks in riddles and metaphors",
    "human_name":     "Adventurer",
    "scene_location": "A misty forest clearing at dawn",
    "default_history": (
        "The two roleplayers approached one another, "
        "and the conversation started."
    ),
}

# ---------------------------------------------------------------------------
# JSON persistence helpers
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(".", "data", "persistent.json")


def load_config(path: str | None = None) -> dict:
    """Load persistent.json and update module-level globals. Returns the dict."""
    global agent_name, agent_role, human_name, scene_location
    global default_history, session_history
    global threads_percent, optimal_threads, cpu_threads
    global selected_gpu, selected_cpu, auto_unload, max_memory_percent
    global text_model_folder, image_model_folder, vram_assigned
    global selected_steps, selected_sample_method, selected_cfg_scale
    global default_negative_prompt, user_turn_start_time

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

    # default_history is the saved starting template.
    # Legacy configs may still use the old "session_history" key.
    default_history  = data.get(
        "default_history",
        data.get("session_history", default_history),
    )

    # On a fresh load (startup / new-session), seed session_history from the
    # saved default.  During a running session session_history is managed by
    # the consolidation prompt and should not be overwritten here — the
    # caller (reset_session_state) explicitly sets it when needed.
    if not session_history:
        session_history = default_history

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
        "default_history": default_history,

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