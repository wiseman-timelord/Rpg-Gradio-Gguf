# scripts/configure.py
# Central configuration: runtime globals, option lists, JSON persistence.
#
# UPDATED for multi-agent support:
#   - agent_name / agent_role replaced by agent1/2/3 name and role
#   - human_age, human_gender added to human description
#   - event_time added alongside scene_location
#   - GENDER_OPTIONS constant used by human gender dropdown
#
# UPDATED for Z-Image-Turbo + Qwen3-4b-Z-Image-Turbo-AbliteratedV1:
#   - cfg_scale default → 1.0
#   - Negative prompt default → ""
#   - Steps default → 8
#   - Sample method default → "euler"

import os
import json
import time

# ---------------------------------------------------------------------------
# Runtime state  (mutable at runtime, not saved automatically)
# ---------------------------------------------------------------------------

# Idle-unload settings
IDLE_UNLOAD_SECONDS: int = 15 * 60          # 15 minutes
user_turn_start_time: float | None = None   # None  → not user's turn yet

# Cancel flag — set True by the "Cancel Response" button to abort the
# current inference pipeline between steps.  Reset to False at the
# start of every new chat_with_model() call.
cancel_processing: bool = False

# ---------------------------------------------------------------------------
# Agent fields  (up to 3 agents; empty/None means agent is inactive)
# ---------------------------------------------------------------------------
GENDER_OPTIONS: list[str] = ["None", "Male", "Female"]

agent1_name: str = "Wise-Llama"
agent1_role: str = "A wise oracle llama"

agent2_name: str = "Blue-Bird"
agent2_role: str = "A jovial song bird"

agent3_name: str = ""
agent3_role: str = ""

# ---------------------------------------------------------------------------
# Human / Player fields
# ---------------------------------------------------------------------------
human_name: str   = "Benevolent-Adventurer"
human_age: str    = ""           # blank / "None" -> omit from prompts
human_gender: str = "None"       # "None" -> omit from prompts

# ---------------------------------------------------------------------------
# Scene / setting
# ---------------------------------------------------------------------------
scene_location: str = "A misty forest clearing"
event_time: str     = "07:13"    # blank / "None" -> omit from prompts

# ---------------------------------------------------------------------------
# Session narrative
# ---------------------------------------------------------------------------
default_history: str = (
    "The three roleplayers approached one another, and the conversation started."
)
session_history: str = ""

scenario_log: str = ""
human_input: str = ""
agent_output: str = ""
agent_exchange: str = ""        # full multi-agent response block, e.g. "Agent1: ...\nAgent2: ..."
consolidated_instance: str = "" # LLM summary of the most recent single rotation (used for image prompts)
rotation_counter: int = 0
latest_image_path: str | None = None

# Per-session image tracking
session_image_paths: list[str] = []
session_folder: str | None = None

# ---------------------------------------------------------------------------
# Shutdown callback -- set by launcher.py at startup.
# ---------------------------------------------------------------------------
shutdown_fn = None

# ---------------------------------------------------------------------------
# Hardware / threading
# ---------------------------------------------------------------------------
CPU_THREADS: int = os.cpu_count() or 4
threads_percent: int = 80
optimal_threads: int = max(1, CPU_THREADS * 80 // 100)

selected_gpu: int = 0
selected_cpu: int = 0
cpu_threads: int = max(1, CPU_THREADS * 80 // 100)
auto_unload: bool = False
max_memory_percent: int = 85

DETECTED_GPUS: list[dict] = []
DETECTED_CPUS: list[dict] = []

# ---------------------------------------------------------------------------
# Model references
# ---------------------------------------------------------------------------
text_model = None
image_model = None
text_model_loaded: bool = False
image_model_loaded: bool = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
text_model_folder: str = "./models/text"
image_model_folder: str = "./models/image"

# ---------------------------------------------------------------------------
# VRAM budget (MB)
# ---------------------------------------------------------------------------
vram_assigned: int = 8192
VRAM_OPTIONS: list[int] = [2048, 4096, 6144, 8192, 10240, 12288, 16384, 24576]

# ---------------------------------------------------------------------------
# Image generation options  (tuned for Z-Image-Turbo)
# ---------------------------------------------------------------------------
IMAGE_SIZE_OPTIONS: dict = {
    "available_sizes": [
        "256x256", "256x512", "512x256",
        "512x512",
        "512x768", "768x512", "768x768",
        "768x1024", "1024x768", "1024x1024",
    ],
    "selected_size": "512x256",
}

STEPS_OPTIONS: list[int] = [4, 6, 8, 10, 12, 15, 20]
selected_steps: int = 4

SAMPLE_METHOD_OPTIONS: list[str] = [
    "euler", "euler_a", "heun", "dpm2",
    "dpm++2s_a", "dpm++2m", "dpm++2mv2", "lcm",
]
selected_sample_method: str = "euler"

CFG_SCALE_OPTIONS: list[float] = [1.0, 1.5, 2.0, 3.0, 5.0, 7.0]
selected_cfg_scale: float = 1.0

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
    "instance": {
        "temperature": 0.5,
        "repeat_penalty": 1.0,
        "max_tokens": 400,
    },
    "image_prompt": {
        "temperature": 0.6,
        "repeat_penalty": 1.0,
        "max_tokens": 350,
    },
}

# ---------------------------------------------------------------------------
# Default RP settings -- used by "Restore RP Defaults"
# ---------------------------------------------------------------------------
DEFAULT_RP_SETTINGS: dict = {
    "agent1_name": "Wise-Llama",
    "agent1_role": "A wise oracle llama",

    "agent2_name": "Blue-Bird",
    "agent2_role": "A bird speaking in songs",

    "agent3_name": "",
    "agent3_role": "",

    "human_name":   "Benevolent-Adventurer",
    "human_age":    "",
    "human_gender": "None",

    "scene_location": "A misty forest clearing",
    "event_time":     "07:13",

    "default_history": (
        "The three roleplayers approached one another, "
        "and the conversation started."
    ),
}

# ---------------------------------------------------------------------------
# JSON persistence helpers
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(".", "data", "persistent.json")


def load_config(path: str | None = None) -> dict:
    """Load persistent.json and update module-level globals. Returns the dict."""
    global agent1_name, agent1_role
    global agent2_name, agent2_role
    global agent3_name, agent3_role
    global human_name, human_age, human_gender
    global scene_location, event_time
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

    # --- Agent fields (support legacy single-agent keys for old configs) ---
    legacy_name = data.get("agent_name", "")
    legacy_role = data.get("agent_role", "")

    agent1_name = data.get("agent1_name", legacy_name or agent1_name)
    agent1_role = data.get("agent1_role", legacy_role or agent1_role)

    agent2_name = data.get("agent2_name", agent2_name)
    agent2_role = data.get("agent2_role", agent2_role)

    agent3_name = data.get("agent3_name", agent3_name)
    agent3_role = data.get("agent3_role", agent3_role)

    # --- Human fields ---
    human_name   = data.get("human_name",   human_name)
    human_age    = data.get("human_age",    human_age)
    human_gender = data.get("human_gender", human_gender)

    # --- Scene / setting ---
    scene_location = data.get("scene_location", scene_location)
    event_time     = data.get("event_time",     event_time)

    # --- Narrative history ---
    default_history = data.get(
        "default_history",
        data.get("session_history", default_history),
    )
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
    selected_steps          = data.get("image_steps",    selected_steps)
    selected_sample_method  = data.get("sample_method",  selected_sample_method)
    selected_cfg_scale      = data.get("cfg_scale",      selected_cfg_scale)
    default_negative_prompt = data.get("negative_prompt", default_negative_prompt)

    # --- Hardware / threading ---
    selected_gpu       = data.get("selected_gpu",       selected_gpu)
    selected_cpu       = data.get("selected_cpu",       selected_cpu)
    auto_unload        = data.get("auto_unload",        auto_unload)
    max_memory_percent = data.get("max_memory_percent", max_memory_percent)

    threads_percent = data.get("threads_percent", threads_percent)
    raw_cpu_threads = data.get("cpu_threads", 0)
    if raw_cpu_threads and raw_cpu_threads > 0:
        cpu_threads = max(1, int(raw_cpu_threads))
    else:
        cpu_threads = max(1, (CPU_THREADS * threads_percent) // 100)

    optimal_threads = cpu_threads

    print(f"Config loaded from {path}.")
    return data


def save_config(path: str | None = None) -> None:
    """Persist current globals to persistent.json."""
    path = path or CONFIG_PATH
    data = {
        # --- Agent fields ---
        "agent1_name": agent1_name,
        "agent1_role": agent1_role,

        "agent2_name": agent2_name,
        "agent2_role": agent2_role,

        "agent3_name": agent3_name,
        "agent3_role": agent3_role,

        # --- Human fields ---
        "human_name":   human_name,
        "human_age":    human_age,
        "human_gender": human_gender,

        # --- Scene / setting ---
        "scene_location": scene_location,
        "event_time":     event_time,

        # --- Narrative ---
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
        "threads_percent":    threads_percent,
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