# installer.py
# Standalone installer for Rpg-Gradio-Gguf
#
# UPDATED:
#   - Main menu with system detections (CPU, Vulkan, CMake, OS)
#   - Clean install / Check install / Refresh configs
#   - Robust pip retry with exponential backoff
#   - Pre‑install cmake binary wheel to avoid build failures
#   - critical_fail() stops installation on any error
#   - consistent UI with print_header() and clear_screen()
#   - CMake detection via PATH and VS Build Tools 2019/2022 install paths (vswhere + hard-coded fallback)

import subprocess
import sys
import os
import json
import zipfile
import urllib.request
import urllib.error
import time
import shutil
import re
import tempfile
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------
PACKAGES = [
    "gradio>=4.0",
    "Pillow",
    "gguf-parser",
    "pywebview>=5.0",
]

# llama-cpp-python versions
LLAMACPP_PYTHON_PREBUILT_VERSION = "v0.3.16"      # eswarthammana wheels
LLAMACPP_PYTHON_COMPILE_VERSION_FALLBACK = "v0.3.26"

# stable-diffusion-cpp-python – no pre‑built Vulkan wheels, so we fall back to source
SD_CPP_PACKAGE = "stable-diffusion-cpp-python"

# Standalone llama.cpp Vulkan binaries (for bundled server, not used for Python wheel)
LLAMA_CPP_BIN_VERSION = "b8123"
VULKAN_BIN_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/"
    f"{LLAMA_CPP_BIN_VERSION}/llama-{LLAMA_CPP_BIN_VERSION}-bin-win-vulkan-x64.zip"
)
VULKAN_BIN_DIR = os.path.join(".", "data", "llama_cpp-vulkan")
VULKAN_ZIP_PATH = os.path.join(".", "data", "llama_cpp_vulkan.zip")

# ae.safetensors VAE (required by Z-Image-Turbo)
AE_SAFETENSORS_URLS = [
    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
]
AE_SAFETENSORS_PATH = os.path.join(".", "models", "ae.safetensors")

# Download settings
DOWNLOAD_MAX_RETRIES = 10
DOWNLOAD_RETRY_DELAY = 5
DOWNLOAD_CHUNK_SIZE = 1024 * 512

# Build retry settings
BUILD_MAX_RETRIES = 3
BUILD_RETRY_DELAY = 10

# Default persistent config (unchanged)
DEFAULT_CONFIG = {
    "agent1_name": "Wise-Llama",
    "agent1_role": "A wise oracle llama",
    "agent2_name": "Blue-Bird",
    "agent2_role": "A jovial song bird",
    "agent3_name": "",
    "agent3_role": "",
    "human_name": "Benevolent-Adventurer",
    "human_age": "",
    "human_gender": "None",
    "scene_location": "A misty forest clearing",
    "event_time": "16:20",
    "default_history": "The three roleplayers approached one another, and the conversation started.",
    "text_model_folder": "./models/text",
    "image_model_folder": "./models/image",
    "vram_assigned": 8192,
    "image_size": "512x256",
    "image_steps": 4,
    "sample_method": "euler",
    "cfg_scale": 1.0,
    "negative_prompt": "",
    "selected_gpu": 0,
    "selected_cpu": 0,
    "cpu_threads": 0,
    "threads_percent": 85,
    "auto_unload": False,
    "max_memory_percent": 85,
}

# Platform detection
PLATFORM = "windows" if sys.platform == "win32" else "linux"
PY_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"

# Global detection cache
_DETECTED_CPU_FEATURES = None
_DETECTED_VULKAN = None
_DETECTED_BUILD_TOOLS = None   # will hold {"Git": bool, "CMake": bool, "MSVC": bool, "MSBuild": bool}
_DETECTIONS_RUN = False

# ---------------------------------------------------------------------------
# UI Functions
# ---------------------------------------------------------------------------
def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    """Clear screen and print a formatted header with the given title."""
    clear_screen()
    print("=" * 77)
    print(f"    {title}")
    print("=" * 77)
    print()

# ---------------------------------------------------------------------------
# Detection Functions (cached)
# ---------------------------------------------------------------------------
def detect_cpu_features():
    """Detect CPU SIMD features (simplified version)."""
    global _DETECTED_CPU_FEATURES
    if _DETECTED_CPU_FEATURES is not None:
        return _DETECTED_CPU_FEATURES
    features = {
        "AVX": False, "AVX2": False, "AVX512": False, "FMA": False,
        "F16C": False, "SSE3": False, "SSSE3": False, "SSE4_1": False, "SSE4_2": False
    }
    if PLATFORM == "windows":
        try:
            import ctypes
            _ipfp = ctypes.windll.kernel32.IsProcessorFeaturePresent
            _ipfp.restype = ctypes.c_bool
            _ipfp.argtypes = [ctypes.c_uint]
            features["SSE3"]   = bool(_ipfp(13))
            features["SSSE3"]  = bool(_ipfp(36))
            features["SSE4_1"] = bool(_ipfp(37))
            features["SSE4_2"] = bool(_ipfp(38))
            features["AVX"]    = bool(_ipfp(39))
            features["AVX2"]   = bool(_ipfp(40))
            features["AVX512"] = bool(_ipfp(41))
        except:
            pass
    else:  # Linux
        try:
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read().lower()
            features["AVX"]    = 'avx'    in content
            features["AVX2"]   = 'avx2'   in content
            features["AVX512"] = 'avx512' in content
            features["FMA"]    = 'fma'    in content
            features["F16C"]   = 'f16c'   in content
            features["SSE3"]   = 'sse3'   in content or 'pni' in content
            features["SSSE3"]  = 'ssse3'  in content
            features["SSE4_1"] = 'sse4_1' in content
            features["SSE4_2"] = 'sse4_2' in content
        except:
            pass
    _DETECTED_CPU_FEATURES = features
    return features

def is_vulkan_installed():
    """Check if Vulkan runtime is available."""
    global _DETECTED_VULKAN
    if _DETECTED_VULKAN is not None:
        return _DETECTED_VULKAN
    if PLATFORM == "windows":
        sys32 = os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32")
        vulkan_dll = os.path.join(sys32, "vulkan-1.dll")
        if os.path.exists(vulkan_dll):
            _DETECTED_VULKAN = True
            return True
        if os.environ.get("VULKAN_SDK"):
            _DETECTED_VULKAN = True
            return True
        _DETECTED_VULKAN = False
        return False
    else:  # Linux
        try:
            result = subprocess.run(["vulkaninfo", "--summary"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=5)
            _DETECTED_VULKAN = (result.returncode == 0)
        except:
            _DETECTED_VULKAN = False
        return _DETECTED_VULKAN

def _find_cmake_in_vs_installations() -> str | None:
    """
    Search for cmake.exe inside Visual Studio / Build Tools installations
    (2019 and 2022, all editions) using vswhere, then by walking known paths.
    Returns the directory containing cmake.exe, or None if not found.
    """
    prog_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    prog_files     = os.environ.get("ProgramFiles",       r"C:\Program Files")

    # --- Strategy 1: ask vswhere for every install path (all products/versions) ---
    vswhere_exe = Path(prog_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    install_roots: list[str] = []

    if vswhere_exe.exists():
        try:
            result = subprocess.run(
                [
                    str(vswhere_exe),
                    "-all",               # all installed products
                    "-prerelease",        # include pre-release
                    "-property", "installationPath",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                install_roots = [p.strip() for p in result.stdout.splitlines() if p.strip()]
        except Exception:
            pass

    # --- Strategy 2: hard-coded default roots for VS 2019 & 2022, all editions ---
    #     (catches standalone Build Tools whose vswhere entry may be missing)
    for base in (prog_files_x86, prog_files):
        for year in ("2022", "2019"):
            for edition in (
                "BuildTools",
                "Enterprise", "Professional", "Community", "Preview",
            ):
                candidate = os.path.join(base, "Microsoft Visual Studio", year, edition)
                if os.path.isdir(candidate) and candidate not in install_roots:
                    install_roots.append(candidate)

    # --- Walk each install root looking for cmake.exe under the CMake component ---
    for root in install_roots:
        cmake_base = os.path.join(root, "Common7", "IDE", "CommonExtensions",
                                  "Microsoft", "CMake", "CMake", "bin")
        cmake_exe = os.path.join(cmake_base, "cmake.exe")
        if os.path.isfile(cmake_exe):
            return cmake_base   # found – return the bin directory

    return None


def detect_build_tools_available() -> dict:
    tools = {"Git": False, "CMake": False, "MSVC": False, "MSBuild": False}

    # Git ------------------------------------------------------------------
    if shutil.which("git"):
        tools["Git"] = True

    # CMake ----------------------------------------------------------------
    # Priority 1: already on PATH
    if shutil.which("cmake"):
        tools["CMake"] = True
    elif PLATFORM == "windows":
        # Priority 2: bundled inside VS / Build Tools 2019-2022
        cmake_bin = _find_cmake_in_vs_installations()
        if cmake_bin:
            tools["CMake"] = True
            # Add to PATH for this process so subsequent cmake calls work
            os.environ["PATH"] = cmake_bin + os.pathsep + os.environ.get("PATH", "")

    # MSVC / MSBuild (Windows only) ----------------------------------------
    if PLATFORM == "windows":
        try:
            prog_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
            vswhere = (Path(prog_files_x86) / "Microsoft Visual Studio"
                       / "Installer" / "vswhere.exe")
            if vswhere.exists():
                result = subprocess.run(
                    [str(vswhere), "-latest", "-property", "installationPath"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    tools["MSVC"] = True
                    msbuild_path = (Path(result.stdout.strip())
                                   / "MSBuild" / "Current" / "Bin" / "MSBuild.exe")
                    if msbuild_path.exists():
                        tools["MSBuild"] = True
        except Exception:
            pass

    return tools

def run_detections():
    """Run all detections once and cache results."""
    global _DETECTED_CPU_FEATURES, _DETECTED_BUILD_TOOLS, _DETECTED_VULKAN, _DETECTIONS_RUN

    if _DETECTIONS_RUN:
        return

    _DETECTED_CPU_FEATURES = detect_cpu_features()
    _DETECTED_BUILD_TOOLS = detect_build_tools_available()
    _DETECTED_VULKAN = is_vulkan_installed()

    _DETECTIONS_RUN = True

# ---------------------------------------------------------------------------
# Helper Functions (critical_fail, download_with_resume, pip_retry, etc.)
# ---------------------------------------------------------------------------
def run_cmd(cmd, description, check=True, env=None):
    print(f"  -> {description}")
    result = subprocess.run(cmd, shell=True, env=env)
    if check and result.returncode != 0:
        print(f"  !! FAILED: {description}")
        return False
    return True

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

def critical_fail(message):
    print()
    print("=" * 77)
    print("  !! CRITICAL ERROR — Installation cannot continue")
    print("=" * 77)
    print(f"  {message}")
    print("=" * 77)
    print()
    print("-" * 70)
    print("  Press any key to return to Batch Menu...")
    print("-" * 70)
    if os.name == "nt":
        import msvcrt
        msvcrt.getch()
    else:
        input()
    sys.exit(1)

def download_with_resume(url, dest_path, max_retries=DOWNLOAD_MAX_RETRIES,
                         retry_delay=DOWNLOAD_RETRY_DELAY):
    """Download a file with resume support. Returns True on success."""
    for attempt in range(1, max_retries + 1):
        existing_bytes = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
        try:
            req = urllib.request.Request(url)
            if existing_bytes:
                req.add_header("Range", f"bytes={existing_bytes}-")
                print(f"  Resuming from byte {existing_bytes:,} (attempt {attempt}/{max_retries})...")
            else:
                print(f"  Starting download (attempt {attempt}/{max_retries})...")
            with urllib.request.urlopen(req, timeout=60) as response:
                content_range = response.headers.get("Content-Range", "")
                content_length = response.headers.get("Content-Length", "")
                if content_range:
                    total_bytes = int(content_range.split("/")[-1])
                elif content_length:
                    total_bytes = existing_bytes + int(content_length)
                else:
                    total_bytes = None
                status_code = response.status
                if status_code == 200 and existing_bytes:
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
                            print(f"\r  Progress: {downloaded:,} / {total_bytes:,} bytes ({pct:.1f}%)", end="", flush=True)
                print()
                final_size = os.path.getsize(dest_path)
                if total_bytes and final_size < total_bytes:
                    print(f"  Incomplete download ({final_size:,} of {total_bytes:,} bytes). Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
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
    return False

def pip_install_with_retry(pip_exe: str, package: str, extra_args: list = None,
                           max_retries: int = 10, initial_delay: float = 5.0,
                           force_reinstall: bool = False, no_deps: bool = False) -> bool:
    """Install a pip package with retry logic and exponential backoff."""
    INACTIVITY_TIMEOUT = 300
    _PROGRESS_KEYWORDS = ("downloading", "installing", "collected", "building",
                          "error", "warning", "failed", "%")
    _SUPPRESS_WARNINGS = ("pip's dependency resolver does not currently take into account",)

    if extra_args is None:
        extra_args = []

    pkg_name = package.split(">=")[0].split("==")[0].split("[")[0]
    delay = initial_delay

    install_flags = []
    if force_reinstall:
        install_flags.append("--force-reinstall")
    if no_deps:
        install_flags.append("--no-deps")

    for attempt in range(max_retries):
        cmd = [pip_exe, "install"] + install_flags + [package] + extra_args
        all_output: list[str] = []
        last_activity = [time.time()]
        reader_done  = [False]
        stall_reason: list[str] = [None]

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )

            def _read_output():
                try:
                    for raw_line in proc.stdout:
                        line = raw_line.rstrip()
                        if not line:
                            continue
                        if any(kw in line.lower() for kw in _SUPPRESS_WARNINGS):
                            continue
                        last_activity[0] = time.time()
                        all_output.append(line)
                        if any(kw in line.lower() for kw in _PROGRESS_KEYWORDS):
                            print(f"    {line}", flush=True)
                finally:
                    reader_done[0] = True

            reader = threading.Thread(target=_read_output, daemon=True)
            reader.start()

            while not reader_done[0]:
                time.sleep(2)
                idle = time.time() - last_activity[0]
                if idle >= INACTIVITY_TIMEOUT:
                    stall_reason[0] = f"No output for {idle:.0f}s — connection stalled"
                    proc.kill()
                    break

            reader.join(timeout=5)
            proc.wait()

            combined = "\n".join(all_output).lower()

            if proc.returncode == 0 or "already satisfied" in combined:
                return True

            if stall_reason[0]:
                reason = stall_reason[0]
            else:
                error_lines = [l for l in all_output if "error" in l.lower()]
                reason = (f"pip error — {error_lines[-1][:120]}" if error_lines
                         else f"pip exited with code {proc.returncode}")

            if attempt < max_retries - 1:
                print(f"    Reason: {reason}")
                print(f"    Retry {attempt + 1}/{max_retries} for {pkg_name} in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 300)

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Unexpected error: {e}")
                print(f"    Retry {attempt + 1}/{max_retries} for {pkg_name} in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 300)

    return False

def get_latest_llamacpp_python_version():
    """Fetch the latest release tag from GitHub (e.g., v0.3.26)."""
    try:
        import requests
        response = requests.get(
            "https://api.github.com/repos/abetlen/llama-cpp-python/releases",
            timeout=10
        )
        if response.status_code == 200:
            releases = response.json()
            version_pattern = re.compile(r'^v?\d+\.\d+\.\d+$')
            for release in releases:
                tag = release.get("tag_name", "")
                if release.get("prerelease"):
                    continue
                if version_pattern.match(tag):
                    print(f"[INFO] Latest llama-cpp-python release: {tag}")
                    return tag
            for release in releases:
                tag = release.get("tag_name", "")
                if version_pattern.match(tag):
                    print(f"[INFO] Using pre-release: {tag}")
                    return tag
        print(f"[WARN] Using fallback version {LLAMACPP_PYTHON_COMPILE_VERSION_FALLBACK}")
        return LLAMACPP_PYTHON_COMPILE_VERSION_FALLBACK
    except Exception as e:
        print(f"[WARN] Failed to fetch latest version: {e}")
        return LLAMACPP_PYTHON_COMPILE_VERSION_FALLBACK

def install_wheel_local(wheel_path, description):
    """Install a .whl file using pip."""
    pip_exe = os.path.join(".", "venv", "Scripts", "pip.exe")
    if not os.path.exists(pip_exe):
        critical_fail("Virtual environment not found. Please run directory creation first.")
    if not pip_install_with_retry(pip_exe, wheel_path, max_retries=3, initial_delay=5.0):
        critical_fail(f"Failed to install {description} from local wheel.")
    print(f"  -> Successfully installed {description}")

def build_llama_cpp_python_with_flags(build_flags, version_tag):
    """Build llama-cpp-python from source with given CMAKE flags using pip_install_with_retry."""
    pip_exe = os.path.join(".", "venv", "Scripts", "pip.exe")
    raw_version = version_tag.lstrip("v")
    pkg_spec = f"llama-cpp-python=={raw_version}"
    print(f"  -> Building llama-cpp-python {raw_version} from source...")
    env = os.environ.copy()
    cmake_args = [f"-D{key}={value}" for key, value in build_flags.items()]
    if cmake_args:
        env["CMAKE_ARGS"] = " ".join(cmake_args)
        print(f"  -> CMAKE_ARGS: {env['CMAKE_ARGS']}")
    env["FORCE_CMAKE"] = "1"
    old_env = os.environ.copy()
    os.environ.update(env)
    try:
        success = pip_install_with_retry(
            pip_exe, pkg_spec,
            extra_args=["--no-cache-dir", "--force-reinstall", "--verbose"],
            max_retries=BUILD_MAX_RETRIES, initial_delay=BUILD_RETRY_DELAY,
            force_reinstall=True
        )
    finally:
        os.environ.clear()
        os.environ.update(old_env)
    if not success:
        critical_fail("llama-cpp-python compilation failed after multiple attempts.")
    print("  -> llama-cpp-python compiled successfully")
    return True

def build_sd_cpp_python_with_flags(build_flags):
    """Build stable-diffusion-cpp-python from source with given CMAKE flags."""
    pip_exe = os.path.join(".", "venv", "Scripts", "pip.exe")
    print("  -> Building stable-diffusion-cpp-python from source...")
    env = os.environ.copy()
    cmake_args = [f"-D{key}={value}" for key, value in build_flags.items()]
    if cmake_args:
        env["CMAKE_ARGS"] = " ".join(cmake_args)
        print(f"  -> CMAKE_ARGS: {env['CMAKE_ARGS']}")

    # Pre-install cmake binary wheel
    print("  -> Pre-installing cmake binary wheel...")
    if not pip_install_with_retry(pip_exe, "cmake", extra_args=["--only-binary=cmake", "--upgrade"],
                                  max_retries=5, initial_delay=5.0):
        critical_fail("Failed to install cmake binary wheel. Please install CMake manually.")

    # Pre-install all dependencies as binaries BEFORE the source build.
    # This prevents pip from trying to compile Pillow and other C-extension
    # dependencies from source when --no-binary is applied.
    print("  -> Pre-installing stable-diffusion-cpp-python dependencies as binaries...")
    if not pip_install_with_retry(
        pip_exe, SD_CPP_PACKAGE,
        extra_args=["--only-binary=:all:", "--no-cache-dir"],
        max_retries=3, initial_delay=5.0
    ):
        # Non-fatal: if no binary wheel exists that's expected; we just want deps resolved
        print("  (No binary wheel for SD package itself — that is expected, continuing...)")

    old_env = os.environ.copy()
    os.environ.update(env)
    try:
        success = pip_install_with_retry(
            pip_exe, SD_CPP_PACKAGE,
            # KEY FIX: --no-binary only targets THIS package, not :all:
            # --no-deps prevents pip from re-building already-installed dependencies
            extra_args=[
                "--no-binary", "stable-diffusion-cpp-python",
                "--no-cache-dir",
                "--no-deps",          # deps already installed as binaries above
                "--force-reinstall",  # safe now because --no-deps limits scope
                "--verbose",
            ],
            max_retries=BUILD_MAX_RETRIES, initial_delay=BUILD_RETRY_DELAY,
            force_reinstall=True
        )
    finally:
        os.environ.clear()
        os.environ.update(old_env)
    if not success:
        critical_fail("stable-diffusion-cpp-python compilation failed after multiple attempts.")
    print("  -> stable-diffusion-cpp-python compiled successfully")
    return True

# ---------------------------------------------------------------------------
# Main Menu with Detections
# ---------------------------------------------------------------------------
def show_main_menu():
    """Display the main menu with system detections and 3 options."""
    run_detections()

    # CPU features
    cpu_feats = detect_cpu_features()
    cpu_list = [k for k, v in cpu_feats.items() if v]
    cpu_str = " | ".join(cpu_list) if cpu_list else "baseline"

    # Build tools summary
    build_ok = [k for k, v in _DETECTED_BUILD_TOOLS.items() if v]
    build_str = " | ".join(f"{k} OK" for k in build_ok) if build_ok else "none detected"

    # Vulkan
    vulkan_str = "YES" if is_vulkan_installed() else "NO"

    # OS and Python version
    if PLATFORM == "windows":
        try:
            import platform
            ver = platform.version()
            build_num = int(ver.split('.')[-1])
            os_str = f"Windows {'11' if build_num >= 22000 else '10'}.{build_num}"
        except:
            os_str = "Windows"
    else:
        os_str = "Linux"
    py_str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    print_header("Rpg-Gradio-Gguf Installer")
    print("System Detections:")
    print(f"  CPU Features : {cpu_str}")
    print(f"  Build Tools  : {build_str}")
    print(f"  Vulkan       : {vulkan_str}")
    print(f"  OS           : {os_str}")
    print(f"  Python       : {py_str}")
    print()
    print("-" * 79)
    print()
    print("   1) Clean Install (Remove existing venv and reinstall everything)")
    print()
    print("   2) Check/Install (Fix missing packages and libraries)")
    print()
    print("   3) Refresh Configs (Recreate persistent.json with defaults)")
    print()
    print()
    print("=" * 77)

    while True:
        choice = input("Selection; Menu Options=1-3, Abandon=A: ").strip().lower()
        if choice == "a":
            print("Installation abandoned by user.")
            sys.exit(0)
        if choice in ("1", "2", "3"):
            return int(choice)
        print("Invalid selection. Please enter 1, 2, 3 or A.")

def show_backend_menu():
    """Display backend selection menu.
       Options 1 & 2 always available (download only).
       Options 3 & 4 require CMake."""
    run_detections()
    cmake_available = _DETECTED_BUILD_TOOLS.get("CMake", False)
    print_header("Rpg-Gradio-Gguf — Llama.Cpp/StableDiffusion Backend")
    print()
    print()
    print()
    print()
    print("   1) Download CPU Binary / Default CPU Wheel (Wheel v0.3.16)")
    print()
    print("   2) Download Vulkan Binary / Default CPU Wheel (Wheel v0.3.16)")
    if cmake_available:
        print()
        print("   3) Compile CPU Binaries / Compile CPU Wheel (Wheel v0.3.26)")
        print()
        print("   4) Compile Vulkan Binaries / Compile Vulkan Wheel (Wheel v0.3.26)")
    else:
        print()
        print("   (Compile options 3 & 4 require CMake – not detected)")
        print()
        print("   Install CMake from https://cmake.org/download/ and RESTART this terminal.")
    print()
    print()
    print()
    print()
    print()
    print("=" * 77)

    while True:
        if cmake_available:
            choice = input("Selection; Menu Options=1-4, Abandon=A: ").strip().lower()
            if choice == "a":
                print("Installation abandoned by user.")
                sys.exit(0)
            if choice in ("1", "2", "3", "4"):
                return int(choice)
            print("Invalid selection. Please enter 1, 2, 3, 4 or A.")
        else:
            choice = input("Selection; Menu Options=1-2, Abandon=A: ").strip().lower()
            if choice == "a":
                print("Installation abandoned by user.")
                sys.exit(0)
            if choice in ("1", "2"):
                return int(choice)
            print("Invalid selection. Please enter 1, 2 or A.")

# ---------------------------------------------------------------------------
# Installation Steps
# ---------------------------------------------------------------------------
def step_create_directories():
    print("\n[1/6] Creating directory structure...")
    for d in ["data", "models", "models/text", "models/image",
              "scripts", "output", "logs"]:
        ensure_directory(os.path.join(".", d))
        print(f"  OK: ./{d}/")

def step_create_venv(clean=False):
    print("\n[2/6] Creating virtual environment...")
    venv_dir = os.path.join(".", "venv")
    if clean and os.path.exists(venv_dir):
        print("  Removing existing virtual environment (clean install)...")
        shutil.rmtree(venv_dir)
    if os.path.exists(os.path.join(venv_dir, "Scripts", "python.exe")):
        print("  Virtual environment already exists, skipping creation.")
        return
    if not run_cmd(f'"{sys.executable}" -m venv "{venv_dir}"', "Creating venv"):
        critical_fail("Failed to create virtual environment.")

def step_install_packages(backend_choice):
    print("\n[3/6] Installing Python packages into venv...")
    python_exe = os.path.join(".", "venv", "Scripts", "python.exe")
    pip_exe = os.path.join(".", "venv", "Scripts", "pip.exe")
    # Upgrade pip
    print("  -> Upgrading pip...")
    python_exe = os.path.join(".", "venv", "Scripts", "python.exe")
    if not run_cmd(f'"{python_exe}" -m pip install --upgrade pip', "Upgrading pip"):
        print("  WARNING: pip upgrade failed (non-critical, continuing...)")
    # Install core packages
    pkg_string = " ".join(f'"{p}"' for p in PACKAGES)
    if not run_cmd(f'"{pip_exe}" install {pkg_string}', "Installing core packages"):
        critical_fail("Core package installation failed.")

    # ------------------------------------------------------------
    # Install llama-cpp-python and stable-diffusion-cpp-python
    # ------------------------------------------------------------
    if backend_choice == 1:   # CPU pre-built (v0.3.16)
        print("\n  Installing CPU pre‑built wheels (v0.3.16)...")
        # llama-cpp-python: download eswarthammana CPU wheel
        wheel_version = LLAMACPP_PYTHON_PREBUILT_VERSION.lstrip("v")
        wheel_filename = f"llama_cpp_python-{wheel_version}-{PY_TAG}-{PY_TAG}-win_amd64.whl"
        wheel_url = f"https://github.com/eswarthammana/llama-cpp-wheels/releases/download/{LLAMACPP_PYTHON_PREBUILT_VERSION}/{wheel_filename}"
        temp_dir = tempfile.mkdtemp()
        wheel_path = os.path.join(temp_dir, wheel_filename)
        print(f"  Downloading {wheel_filename}...")
        if not download_with_resume(wheel_url, wheel_path):
            critical_fail("Failed to download llama-cpp-python CPU wheel.")
        install_wheel_local(wheel_path, "llama-cpp-python CPU wheel")
        # stable-diffusion-cpp-python: use CPU wheel from PyPI
        if not pip_install_with_retry(pip_exe, SD_CPP_PACKAGE, extra_args=["--prefer-binary"],
                                      max_retries=5, initial_delay=5.0):
            critical_fail("stable-diffusion-cpp-python CPU wheel installation failed.")
        shutil.rmtree(temp_dir, ignore_errors=True)

    elif backend_choice == 2:  # Vulkan pre-built (v0.3.16) – only llama-cpp has Vulkan wheel
        print("\n  Installing Vulkan pre‑built for llama-cpp-python, CPU wheel for stable-diffusion...")
        # llama-cpp-python Vulkan wheel from abetlen index (pre-built)
        if not pip_install_with_retry(
            pip_exe, "llama-cpp-python",
            extra_args=["--prefer-binary", "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/vulkan"],
            max_retries=5, initial_delay=5.0
        ):
            critical_fail("llama-cpp-python Vulkan wheel installation failed.")
        # stable-diffusion-cpp-python: use pre-built CPU wheel (no Vulkan wheel available)
        if not pip_install_with_retry(pip_exe, SD_CPP_PACKAGE, extra_args=["--prefer-binary"],
                                      max_retries=5, initial_delay=5.0):
            critical_fail("stable-diffusion-cpp-python CPU wheel installation failed.")

    elif backend_choice == 3:  # Compile CPU (latest)
        print("\n  Compiling both packages from source (CPU)...")
        latest_tag = get_latest_llamacpp_python_version()
        build_llama_cpp_python_with_flags({}, latest_tag)
        build_sd_cpp_python_with_flags({})   # CPU default

    elif backend_choice == 4:  # Compile Vulkan (latest)
        print("\n  Compiling both packages from source with Vulkan support...")
        latest_tag = get_latest_llamacpp_python_version()
        build_llama_cpp_python_with_flags({"LLAMA_VULKAN": "ON"}, latest_tag)
        build_sd_cpp_python_with_flags({"SD_VULKAN": "ON"})

def step_download_vulkan_binaries():
    print("\n[4/6] Downloading llama.cpp Vulkan binaries...")
    if os.path.isdir(VULKAN_BIN_DIR) and os.listdir(VULKAN_BIN_DIR):
        print("  Vulkan binaries already present, skipping download.")
        return
    ensure_directory(VULKAN_BIN_DIR)
    print(f"  Source: {VULKAN_BIN_URL}")
    if not download_with_resume(VULKAN_BIN_URL, VULKAN_ZIP_PATH):
        critical_fail("Failed to download llama.cpp Vulkan binaries.")
    print("  Extracting...")
    try:
        with zipfile.ZipFile(VULKAN_ZIP_PATH, "r") as zf:
            zf.extractall(VULKAN_BIN_DIR)
        print(f"  Extracted to {VULKAN_BIN_DIR}")
    except Exception as e:
        critical_fail(f"Extraction failed: {e}")
    finally:
        if os.path.exists(VULKAN_ZIP_PATH):
            os.remove(VULKAN_ZIP_PATH)

def step_download_vae():
    print("\n[5/6] Downloading ae.safetensors (VAE for Z-Image-Turbo)...")
    if os.path.isfile(AE_SAFETENSORS_PATH):
        size = os.path.getsize(AE_SAFETENSORS_PATH)
        if size > 100 * 1024 * 1024:
            print(f"  ae.safetensors already present ({size/(1024*1024):.0f} MB), skipping.")
            return
        else:
            print("  Existing file incomplete, re-downloading...")
            os.remove(AE_SAFETENSORS_PATH)
    ensure_directory(os.path.dirname(AE_SAFETENSORS_PATH))
    for url in AE_SAFETENSORS_URLS:
        print(f"  Trying {url}")
        if download_with_resume(url, AE_SAFETENSORS_PATH):
            final_size = os.path.getsize(AE_SAFETENSORS_PATH)
            print(f"  ae.safetensors downloaded ({final_size/(1024*1024):.0f} MB).")
            return
        if os.path.exists(AE_SAFETENSORS_PATH):
            os.remove(AE_SAFETENSORS_PATH)
    critical_fail("ae.safetensors download failed from all sources.")

def step_create_default_config():
    print("\n[6/6] Creating default configuration and assets...")
    config_path = os.path.join(".", "data", "persistent.json")
    # Backup existing config if any
    if os.path.exists(config_path):
        backup = config_path + ".backup"
        shutil.copy2(config_path, backup)
        print(f"  Existing config backed up to {backup}")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    print(f"  Created: {config_path}")
    placeholder = os.path.join(".", "data", "new_session.jpg")
    if not os.path.exists(placeholder):
        try:
            from PIL import Image
            img = Image.new("RGB", (256, 256), color=(30, 30, 40))
            img.save(placeholder)
            print(f"  Created placeholder image: {placeholder}")
        except ImportError:
            print("  Pillow not available; placeholder image skipped.")

def refresh_configs_only():
    """Only refresh persistent.json (keep existing if any, but recreate from defaults)."""
    print_header("Refreshing Configurations")
    config_path = os.path.join(".", "data", "persistent.json")
    if os.path.exists(config_path):
        backup = config_path + ".backup"
        shutil.copy2(config_path, backup)
        print(f"  Existing config backed up to {backup}")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    print(f"  Created new {config_path}")
    print("\nConfiguration refresh complete.")
    print("-" * 70)
    print("  Press any key to return to Batch Menu...")
    if os.name == "nt":
        import msvcrt
        msvcrt.getch()
    else:
        input()
    sys.exit(0)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Show Python version
    print(f"\nPython version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    if sys.version_info < (3, 10):
        critical_fail("Python 3.10+ is required.")

    # Main menu
    main_choice = show_main_menu()
    if main_choice == 3:
        refresh_configs_only()

    clean_install = (main_choice == 1)

    # Backend selection (respects CMake availability)
    backend_choice = show_backend_menu()

    # Run installation steps
    step_create_directories()
    step_create_venv(clean=clean_install)
    step_install_packages(backend_choice)
    step_download_vulkan_binaries()
    step_download_vae()
    step_create_default_config()

    # Final summary
    print(" ---")
    print("Installation Completed Successfully")
    print(" ---")
    print("")
    print("  MODELS TO DOWNLOAD:")
    print("  ---")
    print("  Text:  Qwen3-4b-Z-Image-Turbo-AbliteratedV1.Q4_K_M.gguf")
    print("         -> ./models/text/")
    print("  Image: z_image_turbo-Q4_0.gguf")
    print("         -> ./models/image/")
    print("  ---")
    print("  Then launch from the batch menu (option 1 or 2).")
    print("=" * 77)
    print("\n" + "-" * 70)
    print("  Press any key to return to Batch Menu...")
    print("-" * 70)
    if os.name == "nt":
        import msvcrt
        msvcrt.getch()
    else:
        input()

if __name__ == "__main__":
    main()