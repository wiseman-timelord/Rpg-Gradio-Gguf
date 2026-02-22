# scripts/utilities.py
# General-purpose helper functions.

import os
from scripts import configure as cfg


def calculate_optimal_threads(percent: int = 80) -> int:
    """Return thread count based on a percentage of available CPU cores."""
    cores = os.cpu_count() or 4
    threads = max(1, (cores * percent) // 100)
    print(f"Threads: {threads} ({percent}% of {cores} cores).")
    return threads


def reset_session_state() -> None:
    """Reset conversation state to defaults (does not touch model refs)."""
    cfg.session_history = "The conversation started."
    cfg.agent_output = ""
    cfg.human_input = ""
    cfg.rotation_counter = 0
    cfg.latest_image_path = None
    print("Session state reset.")


def scan_for_gguf(directory: str) -> list[str]:
    """Return a list of .gguf file paths found in *directory*."""
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return []
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".gguf")
    ]
    return files


def browse_folder(current_path: str) -> str:
    """
    Open a native Windows folder-picker dialog.
    Returns the chosen path, or *current_path* if the user cancels.
    Falls back gracefully if tkinter is unavailable.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(
            initialdir=current_path,
            title="Select Model Folder",
        )
        root.destroy()
        if chosen:
            return chosen.replace("/", os.sep)
    except Exception as e:
        print(f"Folder browser unavailable: {e}")
    return current_path


def estimate_gpu_layers(model_path: str, vram_mb: int) -> int:
    """
    Rough heuristic: decide how many layers to offload to the GPU based on
    file size and the user-specified VRAM budget.

    Returns -1 (all layers) when the budget comfortably exceeds model size,
    otherwise a proportional layer count.
    """
    if not os.path.isfile(model_path):
        return 0

    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    overhead_mb = 512  # reserve for KV-cache, scratch buffers, etc.
    usable_vram = max(0, vram_mb - overhead_mb)

    if usable_vram >= file_size_mb:
        print(f"VRAM budget ({vram_mb} MB) covers model ({file_size_mb:.0f} MB). Full GPU offload.")
        return -1  # offload everything

    # Conservative proportional estimate assuming ~60 layers for a 14B model
    estimated_layers = 60
    ratio = usable_vram / max(file_size_mb, 1)
    layers = max(0, int(estimated_layers * ratio))
    print(f"VRAM budget ({vram_mb} MB) < model ({file_size_mb:.0f} MB). Offloading ~{layers} layers.")
    return layers
