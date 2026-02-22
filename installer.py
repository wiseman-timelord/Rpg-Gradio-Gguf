# installer.py
# Standalone installer for Rpg-Gradio-Gguf
# Run with system Python: python installer.py

import subprocess
import sys
import os
import json
import zipfile
import urllib.request
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
SD_CPP_PACKAGE = "stable-diffusion-cpp-python"

# Standalone llama.cpp Vulkan binaries — change version here to upgrade
LLAMA_CPP_BIN_VERSION = "b8123"
VULKAN_BIN_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/"
    f"{LLAMA_CPP_BIN_VERSION}/llama-{LLAMA_CPP_BIN_VERSION}-bin-win-vulkan-x64.zip"
)
VULKAN_BIN_DIR = os.path.join(".", "data", "llama_cpp-vulkan")
VULKAN_ZIP_PATH = os.path.join(".", "data", "llama_cpp_vulkan.zip")

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
    "image_size": "256x256",
    "image_steps": 8,
    "sample_method": "euler_a",
}



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_cmd(cmd, description, check=True):
    """Run a shell command and print status."""
    print(f"  -> {description}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"  !! FAILED: {description}")
        return False
    return True


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)



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
    pip_exe = os.path.join(".", "venv", "Scripts", "pip.exe")
    if not os.path.exists(pip_exe):
        print("  !! pip not found in venv. Venv creation may have failed.")
        return False

    # Upgrade pip first
    run_cmd(f'"{pip_exe}" install --upgrade pip', "Upgrading pip", check=False)

    # Install standard packages
    pkg_string = " ".join(f'"{p}"' for p in PACKAGES)
    if not run_cmd(f'"{pip_exe}" install {pkg_string}', "Installing core packages"):
        return False

    # Install llama-cpp-python with Vulkan wheel
    print("  -> Installing llama-cpp-python (Vulkan build)...")
    vulkan_ok = run_cmd(
        f'"{pip_exe}" install "{LLAMA_CPP_PACKAGE}" '
        f'--prefer-binary '
        f'--extra-index-url "{LLAMA_CPP_VULKAN_INDEX}"',
        "llama-cpp-python (Vulkan)",
        check=False
    )
    if not vulkan_ok:
        print("  !! Vulkan wheel not available. Trying standard llama-cpp-python...")
        run_cmd(
            f'"{pip_exe}" install "{LLAMA_CPP_PACKAGE}"',
            "llama-cpp-python (CPU fallback)",
            check=False
        )

    # Install stable-diffusion-cpp-python
    run_cmd(
        f'"{pip_exe}" install "{SD_CPP_PACKAGE}"',
        "stable-diffusion-cpp-python",
        check=False
    )

    return True


def step_download_vulkan_binaries():
    print("\n[4/5] Downloading llama.cpp Vulkan binaries...")
    if os.path.isdir(VULKAN_BIN_DIR) and os.listdir(VULKAN_BIN_DIR):
        print("  Vulkan binaries already present, skipping download.")
        return True

    ensure_directory(VULKAN_BIN_DIR)

    print(f"  Downloading from: {VULKAN_BIN_URL}")
    print("  This may take a minute...")
    try:
        urllib.request.urlretrieve(VULKAN_BIN_URL, VULKAN_ZIP_PATH)
        print("  Download complete. Extracting...")
    except Exception as e:
        print(f"  !! Download failed: {e}")
        print("  You can manually download the zip and extract to ./data/llama_cpp-vulkan/")
        return False

    try:
        with zipfile.ZipFile(VULKAN_ZIP_PATH, "r") as zf:
            zf.extractall(VULKAN_BIN_DIR)
        print(f"  Extracted to {VULKAN_BIN_DIR}")
    except Exception as e:
        print(f"  !! Extraction failed: {e}")
        return False
    finally:
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
    print("=" * 60)
    print("  Rpg-Gradio-Gguf  —  Installer / Repair")
    print("=" * 60)

    # Check Python version
    v = sys.version_info
    print(f"\nPython version: {v.major}.{v.minor}.{v.micro}")
    if v.major != 3 or v.minor < 10:
        print("WARNING: Python 3.10+ recommended. You have {}.{}.".format(v.major, v.minor))

    step_create_directories()
    step_create_venv()
    step_install_packages()
    step_download_vulkan_binaries()
    step_create_default_config()

    print("\n" + "=" * 60)
    print("  Installation complete.")
    print("  Place your GGUF text model in ./models/text/")
    print("  Place your GGUF image model in ./models/image/")
    print("  Then launch from the batch menu (option 1 or 2).")
    print("=" * 60)


if __name__ == "__main__":
    main()
