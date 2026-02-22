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
# Hardware / threading
# ---------------------------------------------------------------------------
threads_percent: int = 80
optimal_threads: int = max(1, (os.cpu_count() or 4) * 80 // 100)

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
VRAM_OPTIONS: list[int] = [4096, 8192, 12288, 16384]

# ---------------------------------------------------------------------------
# Image generation options
# ---------------------------------------------------------------------------
IMAGE_SIZE_OPTIONS: dict = {
    "available_sizes": [
        "64x128", "128x128", "128x256",
        "256x256", "256x512", "512x512",
    ],
    "selected_size": "256x256",
}

STEPS_OPTIONS: list[int] = [2, 4, 6, 8, 10, 15, 20]
selected_steps: int = 8

SAMPLE_METHOD_OPTIONS: list[str] = [
    "euler", "euler_a", "heun", "dpm2",
    "dpm++2s_a", "dpm++2m", "dpm++2mv2", "lcm",
]
selected_sample_method: str = "euler_a"

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
    global threads_percent, optimal_threads
    global text_model_folder, image_model_folder, vram_assigned
    global selected_steps, selected_sample_method

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

    agent_name = data.get("agent_name", agent_name)
    agent_role = data.get("agent_role", agent_role)
    human_name = data.get("human_name", human_name)
    scene_location = data.get("scene_location", scene_location)
    session_history = data.get("session_history", session_history)
    threads_percent = data.get("threads_percent", threads_percent)
    text_model_folder = data.get("text_model_folder", text_model_folder)
    image_model_folder = data.get("image_model_folder", image_model_folder)
    vram_assigned = data.get("vram_assigned", vram_assigned)
    IMAGE_SIZE_OPTIONS["selected_size"] = data.get(
        "image_size", IMAGE_SIZE_OPTIONS["selected_size"]
    )
    selected_steps = data.get("image_steps", selected_steps)
    selected_sample_method = data.get("sample_method", selected_sample_method)

    total_cores = os.cpu_count() or 4
    optimal_threads = max(1, (total_cores * threads_percent) // 100)

    print(f"Config loaded from {path}.")
    return data


def save_config(path: str | None = None) -> None:
    """Persist current globals to persistent.json."""
    path = path or CONFIG_PATH
    data = {
        "agent_name": agent_name,
        "agent_role": agent_role,
        "human_name": human_name,
        "scene_location": scene_location,
        "session_history": session_history,
        "threads_percent": threads_percent,
        "text_model_folder": text_model_folder,
        "image_model_folder": image_model_folder,
        "vram_assigned": vram_assigned,
        "image_size": IMAGE_SIZE_OPTIONS["selected_size"],
        "image_steps": selected_steps,
        "sample_method": selected_sample_method,
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print("Configuration saved.")
    except Exception as e:
        print(f"Error saving config: {e}")
