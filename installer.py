# installer.py
# Standalone installer for Rpg-Gradio-Gguf
# Run with system Python: python installer.py

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

# Download settings
DOWNLOAD_MAX_RETRIES = 10        # Max resume attempts before giving up
DOWNLOAD_RETRY_DELAY = 5         # Seconds to wait between retries
DOWNLOAD_CHUNK_SIZE = 1024 * 512 # 512 KB read chunks

# Default persistent configuration
DEFAULT_CONFIG = {
    "agent_name": "Wise-Llama",
    "agent_role": "A wise oracle who speaks in riddles and metaphors",
    "human_name": "Adventurer",
    "scene_location": "A misty forest clearing at dawn",
    "session_history": "The conversation started.",
    "threads_percent": 80,
    "text_model_folder": "./models/text",
    "image_model_folder": "./models/image",
    "vram_assigned": 8192,
    "image_size": "768x768",
    "image_steps": 8,
    "cfg_scale": 5.0,
    "sample_method": "euler_a",
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
    print("\n[1/5] Creating directory structure...")
    for d in ["data", "models", "models/text", "models/image",
              "scripts", "generated"]:
        ensure_directory(os.path.join(".", d))
        print(f"  OK: ./{d}/")


def step_create_venv():
    print("\n[2/5] Creating virtual environment...")
    venv_dir = os.path.join(".", "venv")
    if os.path.exists(os.path.join(venv_dir, "Scripts", "python.exe")):
        print("  Virtual environment already exists, skipping creation.")
        return True
    return run_cmd(
        f'"{sys.executable}" -m venv "{venv_dir}"',
        "Creating venv with current Python interpreter"
    )


def step_install_packages():
    print("\n[3/5] Installing Python packages into venv...")
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

    # Install standard packages
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
    print("\n[4/5] Downloading llama.cpp Vulkan binaries...")
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


def step_create_default_config():
    print("\n[5/5] Creating default configuration and assets...")
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
        print("  Text:  Qwen3-4B-abliterated Q4_K_M .gguf  -> ./models/text/")
        print("  Image: waiNSFWIllustrious v14.0 Q8_0 .gguf -> ./models/image/")
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