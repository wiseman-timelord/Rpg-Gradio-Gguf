# scripts/utilities.py
# General-purpose helper functions.
# UPDATED: Smart dual-model VRAM allocation for Z-Image-Turbo + Qwen3 combo.
# detect_gpus() enumerates ALL video adapters on Windows via WMIC (primary)
# and supplements NVIDIA VRAM with nvidia-smi.
#
# UPDATED: Per-session output folders in ./output with keyword-derived names.

import os
import re
import shutil
import platform
import subprocess
import struct
from scripts import configure as cfg


def calculate_optimal_threads(percent: int = 80) -> int:
    """Return thread count based on a percentage of available CPU cores."""
    cores = os.cpu_count() or 4
    threads = max(1, (cores * percent) // 100)
    print(f"Threads: {threads} ({percent}% of {cores} cores).")
    return threads


# ---------------------------------------------------------------------------
# Session folder naming
# ---------------------------------------------------------------------------
# Stop words excluded when extracting keywords from the session history.
_STOP_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "is", "it", "was", "were", "are", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "shall", "should",
    "may", "might", "can", "could", "not", "no", "so", "if", "then", "than",
    "that", "this", "these", "those", "with", "from", "by", "as", "into",
    "about", "up", "out", "off", "over", "under", "between", "through",
    "after", "before", "during", "its", "their", "his", "her", "my", "your",
    "our", "he", "she", "they", "we", "you", "i", "me", "him", "us", "them",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "just", "also", "very", "too", "each", "every", "all", "some", "any",
    "two", "one", "started", "conversation", "roleplayers", "approached",
    "another", "said", "says", "told", "asked",
}


def _extract_keywords(text: str, max_keywords: int = 4) -> list[str]:
    """
    Extract the most meaningful words from *text* for folder naming.

    Uses simple frequency-weighted keyword extraction:
    1. Tokenise to lowercase alpha words.
    2. Remove stop words and very short words.
    3. Return the most frequent remaining words (up to *max_keywords*).
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    filtered = [w for w in words if len(w) > 2 and w not in _STOP_WORDS]

    if not filtered:
        return ["session"]

    # Frequency count — prefer words that appear more often
    freq: dict[str, int] = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1

    # Sort by frequency (desc), then alphabetically for tie-breaking
    ranked = sorted(freq.keys(), key=lambda w: (-freq[w], w))
    return ranked[:max_keywords]


def generate_session_folder_name(history_text: str) -> str:
    """
    Create a session output folder inside ``./output/`` with a name derived
    from the session history.

    Rules:
    - The base name is up to 16 characters, built from keywords in the
      history text joined with underscores.
    - If a folder with that name already exists, a serial suffix ``_001``,
      ``_002``, … is appended (before the folder is created).
    - The folder is created on disk and the full path is returned.
    """
    keywords = _extract_keywords(history_text)

    # Build a base name from keywords, truncating to 16 chars
    base = "_".join(keywords)
    # Keep only filesystem-safe characters
    base = re.sub(r"[^a-zA-Z0-9_]", "", base)
    base = base[:16].rstrip("_") or "session"

    output_root = os.path.join(".", "output")
    os.makedirs(output_root, exist_ok=True)

    candidate = os.path.join(output_root, base)
    if not os.path.exists(candidate):
        os.makedirs(candidate, exist_ok=True)
        print(f"Session folder created: {candidate}")
        return candidate

    # Collision — append serial number _001, _002, …
    serial = 1
    while True:
        suffixed = os.path.join(output_root, f"{base}_{serial:03d}")
        if not os.path.exists(suffixed):
            os.makedirs(suffixed, exist_ok=True)
            print(f"Session folder created: {suffixed}")
            return suffixed
        serial += 1
        if serial > 999:
            # Extreme edge case — fall back to timestamp
            import time
            ts = str(int(time.time()))[-8:]
            fallback = os.path.join(output_root, f"{base[:7]}_{ts}")
            os.makedirs(fallback, exist_ok=True)
            return fallback


# ---------------------------------------------------------------------------
# Session state management
# ---------------------------------------------------------------------------
def reset_session_state() -> None:
    """Reset conversation state to defaults (does not touch model refs).

    Seeds session_history from default_history (the saved starting template).
    Creates a new session output folder based on the history text.
    """
    cfg.session_history = cfg.default_history
    cfg.scenario_log = ""
    cfg.agent_output = ""
    cfg.human_input = ""
    cfg.rotation_counter = 0
    cfg.latest_image_path = None
    cfg.session_image_paths = []
    cfg.session_folder = None          # will be created on first image
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


# ---------------------------------------------------------------------------
# ae.safetensors management
# ---------------------------------------------------------------------------
# The installer downloads ae.safetensors to ./models/ae.safetensors.
# The image model (Z-Image-Turbo) requires it in the same folder as the
# diffusion GGUF.  ensure_vae_in_image_folder() copies it there if missing.

AE_INSTALLER_PATH = os.path.join(".", "models", "ae.safetensors")


def ensure_vae_in_image_folder(image_folder: str) -> str | None:
    """
    Ensure ae.safetensors exists in *image_folder*.

    If the file already exists there, return its path immediately.
    Otherwise, copy it from the installer download location
    (./models/ae.safetensors).

    Returns the full path to ae.safetensors in the image folder,
    or None if it could not be provided.
    """
    target = os.path.join(image_folder, "ae.safetensors")

    if os.path.isfile(target):
        print(f"  VAE already present: {target}")
        return target

    # Try to copy from installer location
    if os.path.isfile(AE_INSTALLER_PATH):
        try:
            os.makedirs(image_folder, exist_ok=True)
            shutil.copy2(AE_INSTALLER_PATH, target)
            print(f"  Copied ae.safetensors → {target}")
            return target
        except Exception as e:
            print(f"  Failed to copy ae.safetensors: {e}")
            return None

    print(f"  WARNING: ae.safetensors not found at {AE_INSTALLER_PATH} or {target}.")
    print("  Re-run the installer (option 3) to download it.")
    return None


# ---------------------------------------------------------------------------
# GGUF layer count estimation
# ---------------------------------------------------------------------------

def _read_gguf_layer_count(model_path: str) -> int | None:
    """
    Try to read the number of layers from GGUF metadata.

    Looks for common metadata keys that store layer count:
      - llama.block_count  (llama-family models)
      - general.block_count

    Returns the layer count, or None if it cannot be determined.
    """
    try:
        from gguf_parser import GGUFParser
        parser = GGUFParser(model_path)
        parser.parse()
        meta = parser.metadata
        for key in ("llama.block_count", "general.block_count",
                     "qwen2.block_count", "block_count"):
            if key in meta:
                return int(meta[key])
    except Exception:
        pass
    return None


def _estimate_layers_from_size(file_size_mb: float) -> int:
    """
    Heuristic: estimate layer count from file size.
    Based on typical GGUF quant sizes:
      ~1-2 GB  → ~24-32 layers  (small models)
      ~2-4 GB  → ~32-40 layers  (4B models)
      ~4-8 GB  → ~40-60 layers  (7B-14B models)
    """
    if file_size_mb < 1500:
        return 28
    elif file_size_mb < 3000:
        return 36
    elif file_size_mb < 6000:
        return 48
    else:
        return 60


# ---------------------------------------------------------------------------
# Smart dual-model VRAM allocation
# ---------------------------------------------------------------------------

def compute_vram_allocation(
    text_model_path: str | None,
    image_model_path: str | None,
    vram_budget_mb: int,
) -> dict:
    """
    Compute optimal VRAM allocation for loading BOTH text and image models
    simultaneously.

    The text model (Qwen3-4B, ~2.5 GB) also serves as the image model's
    text encoder (llm_path).  The image model (Z-Image-Turbo, ~4.6 GB) is
    the diffusion backbone.

    Strategy
    --------
    1. Measure file sizes as a proxy for full-GPU memory cost.
    2. Reserve overhead for KV-cache, VAE decode, scratch buffers.
    3. If both fit entirely → full GPU offload for both.
    4. If tight → prioritise text model layers (it's used more frequently),
       enable CPU offload for the image model's heavier params.
    5. Compute n_gpu_layers for the text model proportionally.

    Returns
    -------
    dict with keys:
        text_gpu_layers : int   (-1 = all, 0 = none, >0 = partial)
        image_offload_cpu : bool  (True = offload diffusion params to CPU)
        info : str               (human-readable summary)
    """
    OVERHEAD_MB = 600  # KV-cache, VAE, Vulkan context, scratch

    text_size_mb = 0.0
    image_size_mb = 0.0
    text_layers = 0

    if text_model_path and os.path.isfile(text_model_path):
        text_size_mb = os.path.getsize(text_model_path) / (1024 * 1024)
        text_layers = _read_gguf_layer_count(text_model_path) or \
                      _estimate_layers_from_size(text_size_mb)

    if image_model_path and os.path.isfile(image_model_path):
        image_size_mb = os.path.getsize(image_model_path) / (1024 * 1024)

    total_model_mb = text_size_mb + image_size_mb
    usable_vram = max(0, vram_budget_mb - OVERHEAD_MB)

    info_parts = [
        f"VRAM budget: {vram_budget_mb} MB, usable: {usable_vram} MB "
        f"(overhead reserve: {OVERHEAD_MB} MB)",
        f"Text model: {text_size_mb:.0f} MB, ~{text_layers} layers",
        f"Image model: {image_size_mb:.0f} MB",
    ]

    # Case 1: Everything fits comfortably
    if usable_vram >= total_model_mb * 1.1:
        info_parts.append("Both models fit in VRAM → full GPU offload.")
        return {
            "text_gpu_layers": -1,
            "image_offload_cpu": False,
            "info": " | ".join(info_parts),
        }

    # Case 2: Tight but possible with image CPU offload
    # With offload_params_to_cpu, the image model uses ~30-40% of its
    # file size in VRAM (activations + currently-executing layers only).
    image_offloaded_vram = image_size_mb * 0.35
    if usable_vram >= text_size_mb + image_offloaded_vram:
        info_parts.append(
            "Tight fit → image model offloads params to CPU, "
            "text model gets full GPU offload."
        )
        return {
            "text_gpu_layers": -1,
            "image_offload_cpu": True,
            "info": " | ".join(info_parts),
        }

    # Case 3: Need to partially offload text model too
    # Give image model its offloaded share, rest goes to text layers.
    vram_for_text = max(0, usable_vram - image_offloaded_vram)
    if text_size_mb > 0 and text_layers > 0:
        ratio = min(1.0, vram_for_text / text_size_mb)
        gpu_layers = max(0, int(text_layers * ratio))
    else:
        gpu_layers = 0

    info_parts.append(
        f"Constrained → text model: {gpu_layers}/{text_layers} layers on GPU, "
        f"image model offloads to CPU."
    )
    return {
        "text_gpu_layers": gpu_layers,
        "image_offload_cpu": True,
        "info": " | ".join(info_parts),
    }


def estimate_gpu_layers(model_path: str, vram_mb: int) -> int:
    """
    Rough heuristic: decide how many layers to offload to the GPU based on
    file size and the user-specified VRAM budget.

    Returns -1 (all layers) when the budget comfortably exceeds model size,
    otherwise a proportional layer count.

    NOTE: For dual-model setups, prefer compute_vram_allocation() which
    accounts for both models sharing the same VRAM.  This function is kept
    as a simpler fallback for single-model scenarios.
    """
    if not os.path.isfile(model_path):
        return 0

    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    overhead_mb = 512  # reserve for KV-cache, scratch buffers, etc.
    usable_vram = max(0, vram_mb - overhead_mb)

    if usable_vram >= file_size_mb:
        print(f"VRAM budget ({vram_mb} MB) covers model ({file_size_mb:.0f} MB). Full GPU offload.")
        return -1  # offload everything

    # Use actual or estimated layer count
    layers = _read_gguf_layer_count(model_path) or \
             _estimate_layers_from_size(file_size_mb)
    ratio = usable_vram / max(file_size_mb, 1)
    gpu_layers = max(0, int(layers * ratio))
    print(f"VRAM budget ({vram_mb} MB) < model ({file_size_mb:.0f} MB). "
          f"Offloading ~{gpu_layers}/{layers} layers.")
    return gpu_layers


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