# installer.py
# Standalone installer for Rpg-Gradio-Gguf
# Run with system Python: python installer.py
#
# UPDATED for Z-Image-Turbo + Qwen3-4b-Z-Image-Turbo-AbliteratedV1:
#   - Downloads ae.safetensors VAE (~350 MB) required by Z-Image-Turbo
#   - Updated default config for Z-Image-Turbo parameters
#   - Updated model download instructions in summary
#
# UPDATED: default_history replaces session_history in persistent.json.
#          Output directory renamed from ./generated to ./output.

import subprocess
import sys
import os
import json
import zipfile
import urllib.request
import urllib.error
import time
import shutil

# ---------------------------------------------------------------------------
# Package list (replaces requirements.txt)
# ---------------------------------------------------------------------------
PACKAGES = [
    "gradio>=4.0",
    "Pillow",
    "gguf-parser",
    "pywebview>=5.0",
]

# llama-cpp-python with Vulkan is installed separately via --extra-index-url
LLAMA_CPP_PACKAGE = "llama-cpp-python"
LLAMA_CPP_VULKAN_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/vulkan"

# stable-diffusion-cpp-python for GGUF image model inference
# Built with Vulkan backend via CMAKE_ARGS
SD_CPP_PACKAGE = "stable-diffusion-cpp-python"

# Standalone llama.cpp Vulkan binaries — change version here to upgrade
LLAMA_CPP_BIN_VERSION = "b8123"
VULKAN_BIN_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/"
    f"{LLAMA_CPP_BIN_VERSION}/llama-{LLAMA_CPP_BIN_VERSION}-bin-win-vulkan-x64.zip"
)
VULKAN_BIN_DIR = os.path.join(".", "data", "llama_cpp-vulkan")
VULKAN_ZIP_PATH = os.path.join(".", "data", "llama_cpp_vulkan.zip")

# ae.safetensors VAE required by Z-Image-Turbo
# Primary: Comfy-Org mirror (smaller, more reliable)
# Fallback: Black Forest Labs original
AE_SAFETENSORS_URLS = [
    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
]
AE_SAFETENSORS_PATH = os.path.join(".", "models", "ae.safetensors")

# Download settings
DOWNLOAD_MAX_RETRIES = 10        # Max resume attempts before giving up
DOWNLOAD_RETRY_DELAY = 5         # Seconds to wait between retries
DOWNLOAD_CHUNK_SIZE = 1024 * 512 # 512 KB read chunks

# Default persistent configuration
# Keys must stay in sync with configure.save_config() / configure.load_config()
# UPDATED: defaults tuned for Z-Image-Turbo (cfg_scale=1.0, no negatives, 8 steps)
# UPDATED: "default_history" replaces "session_history"
DEFAULT_CONFIG = {
    # --- Conversation / Personalize panel ---
    "agent_name":      "Wise-Llama",
    "agent_role":      "A wise oracle who speaks in riddles and metaphors",
    "human_name":      "Adventurer",
    "scene_location":  "A misty forest clearing at dawn",
    "default_history": "The two roleplayers approached one another, and the conversation started.",

    # --- Model paths ---
    "text_model_folder":  "./models/text",
    "image_model_folder": "./models/image",

    # --- VRAM budget ---
    "vram_assigned": 8192,

    # --- Image generation settings (Z-Image-Turbo) ---
    "image_size":      "512x512",
    "image_steps":     4,
    "sample_method":   "euler",
    "cfg_scale":       2.0,
    "negative_prompt": "",

    # --- Hardware / threading ---
    "selected_gpu":       0,
    "selected_cpu":       0,
    "cpu_threads":        0,      # 0 = derive from threads_percent at first load
    "threads_percent":    85,
    "auto_unload":        False,
    "max_memory_percent": 85,
}



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_cmd(cmd, description, check=True, env=None):
    """Run a shell command and print status."""
    print(f"  -> {description}")
    result = subprocess.run(cmd, shell=True, env=env)
    if check and result.returncode != 0:
        print(f"  !! FAILED: {description}")
        return False
    return True


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def download_with_resume(url, dest_path, max_retries=DOWNLOAD_MAX_RETRIES,
                         retry_delay=DOWNLOAD_RETRY_DELAY):
    """
    Download a file to dest_path with support for resuming incomplete downloads.

    If dest_path already exists and is a partial download, an HTTP Range
    request is sent so only the remaining bytes are fetched and appended.
    Retries up to max_retries times on connection errors before giving up.

    Returns True on success, False on permanent failure.
    """
    for attempt in range(1, max_retries + 1):
        # Determine how many bytes we already have
        existing_bytes = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0

        try:
            req = urllib.request.Request(url)

            if existing_bytes:
                req.add_header("Range", f"bytes={existing_bytes}-")
                print(f"  Resuming from byte {existing_bytes:,} (attempt {attempt}/{max_retries})...")
            else:
                print(f"  Starting download (attempt {attempt}/{max_retries})...")

            with urllib.request.urlopen(req, timeout=60) as response:
                # Determine total file size from headers if available
                content_range = response.headers.get("Content-Range", "")
                content_length = response.headers.get("Content-Length", "")

                if content_range:
                    # e.g. "bytes 12345-999999/1000000"
                    total_bytes = int(content_range.split("/")[-1])
                elif content_length:
                    total_bytes = existing_bytes + int(content_length)
                else:
                    total_bytes = None  # Unknown total size

                status_code = response.status
                # 206 = Partial Content (resume accepted), 200 = full file
                if status_code == 200 and existing_bytes:
                    # Server ignored Range header; restart from scratch
                    print("  Server does not support resume; restarting download...")
                    existing_bytes = 0

                mode = "ab" if existing_bytes else "wb"
                downloaded = existing_bytes

                with open(dest_path, mode) as f:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_bytes:
                            pct = downloaded / total_bytes * 100
                            print(f"\r  Progress: {downloaded:,} / {total_bytes:,} bytes"
                                  f"  ({pct:.1f}%)", end="", flush=True)

            print()  # newline after progress line

            # Verify completeness when total size is known
            final_size = os.path.getsize(dest_path)
            if total_bytes and final_size < total_bytes:
                print(f"  Incomplete download ({final_size:,} of {total_bytes:,} bytes)."
                      f" Will retry in {retry_delay}s...")
                time.sleep(retry_delay)
                continue  # retry — existing partial file will be resumed

            print("  Download complete.")
            return True

        except (urllib.error.URLError, OSError, EOFError) as e:
            print(f"\n  !! Connection error: {e}")
            if attempt < max_retries:
                print(f"  Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print("  Max retries reached. Download failed.")
                return False

    return False  # exhausted retries



# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def step_create_directories():
    print("\n[1/6] Creating directory structure...")
    for d in ["data", "models", "models/text", "models/image",
              "scripts", "output", "logs"]:
        ensure_directory(os.path.join(".", d))
        print(f"  OK: ./{d}/")


def step_create_venv():
    print("\n[2/6] Creating virtual environment...")
    venv_dir = os.path.join(".", "venv")
    if os.path.exists(os.path.join(venv_dir, "Scripts", "python.exe")):
        print("  Virtual environment already exists, skipping creation.")
        return True
    return run_cmd(
        f'"{sys.executable}" -m venv "{venv_dir}"',
        "Creating venv with current Python interpreter"
    )


def step_install_packages():
    print("\n[3/6] Installing Python packages into venv...")
    # Use python.exe -m pip rather than pip.exe directly.
    # Calling pip.exe to upgrade itself causes a file-lock error on Windows
    # because pip cannot replace its own running executable.
    python_exe = os.path.join(".", "venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        print("  !! python.exe not found in venv. Venv creation may have failed.")
        return False

    pip_base = f'"{python_exe}" -m pip'

    # Upgrade pip via the interpreter, not pip.exe itself
    run_cmd(f'{pip_base} install --upgrade pip', "Upgrading pip", check=False)

    # Install standard packages (includes pywebview for app-style GUI window)
    pkg_string = " ".join(f'"{p}"' for p in PACKAGES)
    if not run_cmd(f'{pip_base} install {pkg_string}', "Installing core packages"):
        return False

    # Install llama-cpp-python with Vulkan wheel
    print("  -> Installing llama-cpp-python (Vulkan build)...")
    vulkan_ok = run_cmd(
        f'{pip_base} install "{LLAMA_CPP_PACKAGE}" '
        f'--prefer-binary '
        f'--extra-index-url "{LLAMA_CPP_VULKAN_INDEX}"',
        "llama-cpp-python (Vulkan)",
        check=False
    )
    if not vulkan_ok:
        print("  !! Vulkan wheel not available. Trying standard llama-cpp-python...")
        run_cmd(
            f'{pip_base} install "{LLAMA_CPP_PACKAGE}"',
            "llama-cpp-python (CPU fallback)",
            check=False
        )

    # -----------------------------------------------------------------------
    # Install stable-diffusion-cpp-python with Vulkan backend
    # -----------------------------------------------------------------------
    # The Vulkan SDK must be installed on the system for this to compile.
    # On Windows: https://www.lunarg.com/vulkan-sdk/
    # The CMAKE_ARGS env var tells the build to enable Vulkan in sd.cpp.
    # -----------------------------------------------------------------------
    print("  -> Installing stable-diffusion-cpp-python (Vulkan build)...")
    print("     NOTE: Requires Vulkan SDK. Install from https://www.lunarg.com/vulkan-sdk/")

    # Build an environment with CMAKE_ARGS set for Vulkan
    sd_env = os.environ.copy()
    sd_env["CMAKE_ARGS"] = "-DSD_VULKAN=ON"

    sd_vulkan_ok = run_cmd(
        f'{pip_base} install "{SD_CPP_PACKAGE}" --no-binary :all:',
        "stable-diffusion-cpp-python (Vulkan)",
        check=False,
        env=sd_env,
    )
    if not sd_vulkan_ok:
        print("  !! Vulkan build failed (Vulkan SDK may not be installed).")
        print("  Trying standard CPU-only stable-diffusion-cpp-python...")
        run_cmd(
            f'{pip_base} install "{SD_CPP_PACKAGE}"',
            "stable-diffusion-cpp-python (CPU fallback)",
            check=False
        )

    return True


def step_download_vulkan_binaries():
    print("\n[4/6] Downloading llama.cpp Vulkan binaries...")
    if os.path.isdir(VULKAN_BIN_DIR) and os.listdir(VULKAN_BIN_DIR):
        print("  Vulkan binaries already present, skipping download.")
        return True

    ensure_directory(VULKAN_BIN_DIR)

    print(f"  Source: {VULKAN_BIN_URL}")
    print("  Large file — will resume automatically if the connection drops.")

    success = download_with_resume(VULKAN_BIN_URL, VULKAN_ZIP_PATH)
    if not success:
        print("  !! Download failed after all retries.")
        print("  You can manually download the zip and extract to ./data/llama_cpp-vulkan/")
        # Leave any partial zip so the next run can resume it
        return False

    print("  Extracting...")
    try:
        with zipfile.ZipFile(VULKAN_ZIP_PATH, "r") as zf:
            zf.extractall(VULKAN_BIN_DIR)
        print(f"  Extracted to {VULKAN_BIN_DIR}")
    except Exception as e:
        print(f"  !! Extraction failed: {e}")
        # Remove corrupt zip so a fresh download starts next run
        if os.path.exists(VULKAN_ZIP_PATH):
            os.remove(VULKAN_ZIP_PATH)
        return False

    # Clean up zip only after successful extraction
    if os.path.exists(VULKAN_ZIP_PATH):
        os.remove(VULKAN_ZIP_PATH)

    return True


def step_download_vae():
    """
    Download ae.safetensors (~350 MB) required by Z-Image-Turbo.

    The file is saved to ./models/ae.safetensors.  When the user sets
    an image model folder in the program, it is automatically copied
    there if not already present.

    Tries the primary URL first, falls back to the secondary if needed.
    Supports resume for interrupted downloads.
    """
    print("\n[5/6] Downloading ae.safetensors (VAE for Z-Image-Turbo)...")

    if os.path.isfile(AE_SAFETENSORS_PATH):
        file_size = os.path.getsize(AE_SAFETENSORS_PATH)
        # A valid ae.safetensors should be roughly 300-400 MB.
        # If the file is very small, it's probably a failed/partial download.
        if file_size > 100 * 1024 * 1024:  # > 100 MB
            print(f"  ae.safetensors already present ({file_size / (1024*1024):.0f} MB), "
                  "skipping download.")
            return True
        else:
            print(f"  Existing file is only {file_size:,} bytes (likely incomplete). "
                  "Re-downloading...")
            os.remove(AE_SAFETENSORS_PATH)

    ensure_directory(os.path.dirname(AE_SAFETENSORS_PATH))

    for i, url in enumerate(AE_SAFETENSORS_URLS):
        source_label = "primary" if i == 0 else "fallback"
        print(f"  Trying {source_label} source: {url}")
        print("  ~350 MB — will resume automatically if the connection drops.")

        success = download_with_resume(url, AE_SAFETENSORS_PATH)
        if success:
            final_size = os.path.getsize(AE_SAFETENSORS_PATH)
            print(f"  ae.safetensors downloaded successfully ({final_size / (1024*1024):.0f} MB).")
            return True

        # Clean up failed attempt before trying fallback
        if os.path.exists(AE_SAFETENSORS_PATH):
            os.remove(AE_SAFETENSORS_PATH)
        print(f"  {source_label.capitalize()} source failed.")

    print("  !! ae.safetensors download failed from all sources.")
    print("  You can manually download it from:")
    print("    https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors")
    print(f"  and place it at: {os.path.abspath(AE_SAFETENSORS_PATH)}")
    return False


def step_create_default_config():
    print("\n[6/6] Creating default configuration and assets...")
    config_path = os.path.join(".", "data", "persistent.json")
    if os.path.exists(config_path):
        print(f"  Config already exists: {config_path}")
    else:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        print(f"  Created: {config_path}")

    # Create a placeholder startup image if missing
    placeholder = os.path.join(".", "data", "new_session.jpg")
    if not os.path.exists(placeholder):
        try:
            from PIL import Image
            img = Image.new("RGB", (256, 256), color=(30, 30, 40))
            img.save(placeholder)
            print(f"  Created placeholder image: {placeholder}")
        except ImportError:
            print("  Pillow not available yet; placeholder image skipped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Check Python version
    v = sys.version_info
    print(f"\nPython version: {v.major}.{v.minor}.{v.micro}")
    if v.major != 3 or v.minor < 10:
        print("WARNING: Python 3.10+ recommended. You have {}.{}.".format(v.major, v.minor))

    # Run each step and record pass/fail
    steps = [
        ("Directories",           step_create_directories),
        ("Virtual Environment",   step_create_venv),
        ("Python Packages",       step_install_packages),
        ("Vulkan Binaries",       step_download_vulkan_binaries),
        ("VAE (ae.safetensors)",  step_download_vae),
        ("Config & Assets",       step_create_default_config),
    ]

    results = []
    for label, fn in steps:
        ret = fn()
        # Steps that return None (no explicit return) are treated as success
        passed = (ret is None) or (ret is True)
        results.append((label, passed))

    # -----------------------------------------------------------------------
    # Results summary
    # -----------------------------------------------------------------------
    print("\n")
    print("=" * 60)
    print("  Installation Summary")
    print("=" * 60)
    all_passed = True
    for label, passed in results:
        status = "OK  " if passed else "FAIL"
        marker = "+" if passed else "!"
        print(f"  [{marker}] {status}  {label}")
        if not passed:
            all_passed = False
    print("=" * 60)

    if all_passed:
        print("  All steps completed successfully.")
        print()
        print("  MODELS TO DOWNLOAD:")
        print("  ---")
        print("  Text:  Qwen3-4b-Z-Image-Turbo-AbliteratedV1.Q4_K_M.gguf")
        print("         -> ./models/text/")
        print("         (also serves as image encoder)")
        print()
        print("  Image: z_image_turbo-Q4_0.gguf")
        print("         -> ./models/image/")
        print("         (ae.safetensors auto-copied here on first use)")
        print("  ---")
        print("  Then launch from the batch menu (option 1 or 2).")
    else:
        print("  One or more steps failed — see output above for details.")
        print("  Re-run this installer to retry failed steps.")

    print("=" * 60)

    # -----------------------------------------------------------------------
    # Pause before returning to batch menu
    # -----------------------------------------------------------------------
    print()
    print("-" * 70)
    print("  Press any key to return to Batch Menu...")
    print("-" * 70)

    if os.name == "nt":
        # Windows: use msvcrt for a true "any key" pause
        import msvcrt
        msvcrt.getch()
    else:
        # Fallback for non-Windows (shouldn't occur in normal use)
        input()


if __name__ == "__main__":
    main()