# launcher.py
# Entry point for Rpg-Gradio-Gguf.
# Loads persistent configuration, initializes models, and launches the UI.

import os
import sys
import time
import threading

# Ensure project root is on sys.path so `scripts` package resolves
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts import configure as cfg
from scripts.inference import load_text_model, load_image_model
from scripts.displays import launch_gradio_interface


def load_persistent_settings() -> None:
    """Load settings from persistent.json into the configure module."""
    print("Loading persistent settings...")
    cfg.load_config()
    print(
        f"  Agent: {cfg.agent_name} | Role: {cfg.agent_role} | "
        f"Threads: {cfg.optimal_threads} | VRAM: {cfg.vram_assigned} MB"
    )


def initialize_models() -> None:
    """Load text and image GGUF models."""
    print("Initializing models...")

    if load_text_model():
        print("Text model loaded successfully.")
    else:
        print("WARNING: Text model not loaded. Chat will be unavailable.")

    if load_image_model():
        print("Image model loaded successfully.")
    else:
        print("WARNING: Image model not loaded. Image generation will be skipped.")


def background_engine() -> None:
    """
    Watch persistent.json for external changes (e.g. manual edits) and
    hot-reload settings when the file is modified.
    """
    config_path = cfg.CONFIG_PATH
    last_mtime: float | None = None
    print("Background config watcher started.")

    while True:
        try:
            if os.path.exists(config_path):
                current_mtime = os.path.getmtime(config_path)
                if last_mtime is None:
                    last_mtime = current_mtime
                elif current_mtime > last_mtime:
                    print("Config file changed on disk. Reloading settings...")
                    cfg.load_config()
                    last_mtime = current_mtime
            time.sleep(5)
        except Exception as e:
            print(f"Background engine error: {e}")
            time.sleep(10)


def main() -> None:
    """Startup sequence: settings → models → background watcher → UI."""
    print("=" * 50)
    print("  Rpg-Gradio-Gguf  —  Starting Up")
    print("=" * 50)

    load_persistent_settings()
    initialize_models()

    # Background watcher (daemon thread dies with main process)
    watcher = threading.Thread(target=background_engine, daemon=True)
    watcher.start()

    # Gradio blocks the main thread
    launch_gradio_interface()


if __name__ == "__main__":
    main()
