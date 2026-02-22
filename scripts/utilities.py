# scripts/utilities.py
# General-purpose helper functions.
# UPDATED: detect_gpus() now enumerates ALL video adapters on Windows via
# WMIC (primary) and supplements NVIDIA VRAM with nvidia-smi. This ensures
# secondary / non-primary GPUs (AMD, Intel, etc.) are listed alongside the
# primary NVIDIA card in the Selected GPU dropdown.

import os
import platform
import subprocess
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


# ---------------------------------------------------------------------------
# Hardware detection — internal helpers
# ---------------------------------------------------------------------------

def _run_cmd(cmd: list[str], timeout: int = 10) -> str | None:
    """Run a subprocess and return its stdout, or None on any failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _detect_gpus_wmic() -> list[dict]:
    """
    Enumerate ALL video adapters via WMIC (Windows only).

    Command used:
        wmic path win32_VideoController get AdapterRAM,Name /format:csv

    WMIC returns properties in alphabetical order after the Node column,
    giving columns: Node | AdapterRAM | Name.

    Known limitation: WMIC caps AdapterRAM at 4 294 967 295 bytes (~4 GB)
    even for cards with more VRAM.  Call _enrich_with_nvidia_smi() afterwards
    to correct NVIDIA entries.
    """
    if platform.system() != "Windows":
        return []

    output = _run_cmd(
        ["wmic", "path", "win32_VideoController",
         "get", "AdapterRAM,Name", "/format:csv"]
    )
    if not output:
        return []

    gpus: list[dict] = []
    idx = 0
    for line in output.splitlines():
        line = line.strip()
        # Skip header row and blank lines
        if not line or line.lower().startswith("node"):
            continue
        parts = [p.strip() for p in line.split(",")]
        # Expected: [Node, AdapterRAM, Name]
        if len(parts) < 3:
            continue
        name = parts[2]
        if not name or name.lower() in ("", "name"):
            continue
        try:
            vram_mb = int(parts[1]) // (1024 * 1024) if parts[1] else 0
        except ValueError:
            vram_mb = 0
        gpus.append({"index": idx, "name": name, "vram_mb": vram_mb})
        idx += 1

    if gpus:
        print(f"Detected {len(gpus)} GPU(s) via WMIC: "
              + ", ".join(g["name"] for g in gpus))
    return gpus


def _enrich_with_nvidia_smi(gpus: list[dict]) -> None:
    """
    Query nvidia-smi for accurate VRAM figures and update matching GPU
    entries in *gpus* in-place.

    Matching is done by checking whether either name string is a substring
    of the other (case-insensitive), which handles WMIC shortening names
    like "NVIDIA GeForce RTX 3080 Ti" vs nvidia-smi's full string.
    """
    output = _run_cmd(
        ["nvidia-smi",
         "--query-gpu=index,name,memory.total",
         "--format=csv,noheader,nounits"]
    )
    if not output:
        return

    nvidia_entries: list[dict] = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                nvidia_entries.append({
                    "name":    parts[1],
                    "vram_mb": int(float(parts[2])),
                })
            except ValueError:
                continue

    for nv in nvidia_entries:
        nv_lower = nv["name"].lower()
        for gpu in gpus:
            wmic_lower = gpu["name"].lower()
            # Accept if either name is contained within the other
            if nv_lower in wmic_lower or wmic_lower in nv_lower:
                old = gpu["vram_mb"]
                gpu["vram_mb"] = nv["vram_mb"]
                print(f"  Corrected VRAM for '{gpu['name']}': "
                      f"{old} MB → {nv['vram_mb']} MB (from nvidia-smi).")
                break


def _detect_gpus_nvidia_smi_only() -> list[dict]:
    """Build GPU list solely from nvidia-smi (no WMIC)."""
    output = _run_cmd(
        ["nvidia-smi",
         "--query-gpu=index,name,memory.total",
         "--format=csv,noheader,nounits"]
    )
    if not output:
        return []

    gpus: list[dict] = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                gpus.append({
                    "index":   int(parts[0]),
                    "name":    parts[1],
                    "vram_mb": int(float(parts[2])),
                })
            except ValueError:
                continue

    if gpus:
        print(f"Detected {len(gpus)} NVIDIA GPU(s) via nvidia-smi.")
    return gpus


def _detect_gpus_vulkaninfo() -> list[dict]:
    """Build GPU list from vulkaninfo --summary (AMD / Intel / any Vulkan)."""
    output = _run_cmd(["vulkaninfo", "--summary"])
    if not output:
        return []

    gpus: list[dict] = []
    idx = 0
    for line in output.splitlines():
        ll = line.lower()
        if "devicename" in ll or "device name" in ll:
            name = (
                line.split("=")[-1].strip()
                if "=" in line
                else line.split(":")[-1].strip()
            )
            gpus.append({"index": idx, "name": name, "vram_mb": 0})
            idx += 1

    if gpus:
        print(f"Detected {len(gpus)} GPU(s) via vulkaninfo.")
    return gpus


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_gpus() -> list[dict]:
    """
    Detect ALL available GPUs and return a list of dicts:
        [{"index": 0, "name": "...", "vram_mb": 8192}, ...]

    Detection strategy
    ------------------
    Windows (primary path):
      1. WMIC win32_VideoController — enumerates every adapter registered
         with Windows, including integrated GPUs, secondary discrete cards,
         and any GPU not supported by nvidia-smi (AMD, Intel Arc, etc.).
      2. nvidia-smi — used only to correct the VRAM figures for NVIDIA
         entries, because WMIC hard-caps AdapterRAM at ~4 GB regardless of
         actual VRAM.

    Fallback (non-Windows or WMIC failure):
      3. nvidia-smi alone
      4. vulkaninfo --summary
    """
    if platform.system() == "Windows":
        # Primary: WMIC gives us the full picture
        gpus = _detect_gpus_wmic()
        if gpus:
            _enrich_with_nvidia_smi(gpus)   # fix up NVIDIA VRAM in-place
            return gpus
        # WMIC produced nothing — try NVIDIA-only path
        gpus = _detect_gpus_nvidia_smi_only()
        if gpus:
            return gpus
        # Last resort
        gpus = _detect_gpus_vulkaninfo()
        if gpus:
            return gpus
    else:
        # Linux / macOS
        gpus = _detect_gpus_nvidia_smi_only()
        if gpus:
            return gpus
        gpus = _detect_gpus_vulkaninfo()
        if gpus:
            return gpus

    print("No GPUs detected.")
    return []


def detect_cpus() -> list[dict]:
    """
    Detect available CPU(s).  Returns a list of dicts:
        [{"index": 0, "name": "...", "threads": 16}, ...]
    """
    total_threads = os.cpu_count() or 4
    cpu_name = platform.processor()

    # platform.processor() can return empty string on some systems
    if not cpu_name or cpu_name.strip() == "":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_name = line.split(":")[1].strip()
                        break
        except Exception:
            pass

    if (not cpu_name or cpu_name.strip() == "") and platform.system() == "Windows":
        output = _run_cmd(["wmic", "cpu", "get", "Name", "/format:csv"])
        if output:
            for line in output.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[1] and "Name" not in parts[1]:
                    cpu_name = parts[1]
                    break

    cpu_name = (cpu_name or "Unknown CPU").strip()
    cpus = [{"index": 0, "name": cpu_name, "threads": total_threads}]
    print(f"Detected CPU: {cpu_name} ({total_threads} threads).")
    return cpus