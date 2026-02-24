# launcher.py
# Entry point for Rpg-Gradio-Gguf.
# Loads persistent configuration, initializes models, and launches the UI.
#
# LAUNCH MODES  (both use the same pywebview window)
# --------------------------------------------------
# Normal  (pythonw.exe via batch option 1):
#   stdout/stderr → ./logs/launcher.log  (pythonw has no console)
#
# Debug   (python.exe via batch option 2):
#   stdout/stderr → console  (console window stays open)
#
# In both cases Gradio starts non-blocking and pywebview opens a native
# application window.  Closing the window [X] or clicking "Exit Program"
# triggers shutdown() which destroys the pywebview window, unblocks
# webview.start(), and terminates the process.

import os
import sys
import time
import threading
import atexit
import platform

# Ensure project root is on sys.path so `scripts` package resolves
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Detect console-less mode (pythonw.exe) and redirect output to a log file.
# pythonw sets sys.stdout and sys.stderr to None, which causes every
# print() call to crash with "NoneType has no attribute write".
# ---------------------------------------------------------------------------
_log_file = None  # keep a reference so it isn't garbage-collected


def _setup_logging() -> None:
    """Redirect stdout/stderr to a log file when running without a console."""
    global _log_file
    if sys.stdout is not None and sys.stderr is not None:
        # Console is available (debug mode) — nothing to do
        return

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "launcher.log")

    _log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _log_file
    sys.stderr = _log_file
    atexit.register(_close_log)
    print(f"Log started — output redirected to {log_path}")


def _close_log() -> None:
    global _log_file
    if _log_file is not None:
        try:
            _log_file.flush()
            _log_file.close()
        except Exception:
            pass
        _log_file = None


# Call immediately — before any other import that might print
_setup_logging()


# ---------------------------------------------------------------------------
# Windows: set AppUserModelID so the taskbar shows our icon, not Python's.
# Must be called BEFORE any window is created.
# ---------------------------------------------------------------------------
if platform.system() == "Windows":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "Anthropic.RpgGradioGguf.App.1"
        )
    except Exception:
        pass


def _set_window_icon_win32(icon_path: str, window_title: str = "Rpg-Gradio-Gguf") -> None:
    """
    Set the titlebar (small) and taskbar (big) icon on a Windows window
    using ctypes calls to user32.dll.

    pywebview's ``webview.start(icon=...)`` only works on GTK/QT, so on
    Windows we must call LoadImageW + SendMessageW directly.

    Parameters
    ----------
    icon_path : str
        Absolute path to the .ico file.
    window_title : str
        The exact window title so FindWindowW can locate the HWND.
    """
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32

        # Win32 constants
        WM_SETICON   = 0x0080
        ICON_SMALL   = 0
        ICON_BIG     = 1
        IMAGE_ICON   = 1
        LR_LOADFROMFILE = 0x0010

        # Find the window by its title
        hwnd = user32.FindWindowW(None, window_title)
        if not hwnd:
            print(f"  Could not find window '{window_title}' for icon assignment.")
            return

        # Load the icon at both sizes from the .ico file
        # 16×16 for the titlebar, 32×32 for the taskbar / Alt-Tab
        icon_small = user32.LoadImageW(
            None, icon_path, IMAGE_ICON, 16, 16, LR_LOADFROMFILE
        )
        icon_big = user32.LoadImageW(
            None, icon_path, IMAGE_ICON, 32, 32, LR_LOADFROMFILE
        )

        if icon_small:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, icon_small)
        if icon_big:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, icon_big)

        if icon_small or icon_big:
            print(f"  Window icon set from {icon_path}")
        else:
            print(f"  WARNING: LoadImageW returned NULL — is the .ico valid?")
    except Exception as e:
        print(f"  Could not set window icon: {e}")


# ---------------------------------------------------------------------------
# Verify pywebview is available — fail early if not installed
# ---------------------------------------------------------------------------
try:
    import webview
except ImportError:
    print("FATAL: pywebview is not installed.")
    print("Run the installer (batch option 3) to install all dependencies.")
    sys.exit(1)


# Now safe to import project modules (they print during import)
from scripts import configure as cfg
from scripts.inference import load_text_model, load_image_model, unload_text_model, unload_image_model
from scripts.displays import launch_gradio_interface


# ---------------------------------------------------------------------------
# Shutdown  (registered on cfg so displays.py can call it)
# ---------------------------------------------------------------------------
def shutdown() -> None:
    """
    Clean shutdown for the entire application.

    Called from two places:
      • "Exit Program" button in Gradio  (via cfg.shutdown_fn)
      • pywebview window [X] close       (webview.start() unblocks naturally)

    Unloads any loaded models before destroying the pywebview window so that
    VRAM / RAM is released cleanly rather than relying on process teardown.
    Destroying all webview windows causes webview.start() to return in
    main(), which then calls os._exit(0) to terminate the process and
    all background threads.
    """
    print("Shutdown requested...")

    # Unload models cleanly before the process exits
    try:
        if cfg.text_model_loaded:
            unload_text_model()
    except Exception as e:
        print(f"Error unloading text model: {e}")

    try:
        if cfg.image_model_loaded:
            unload_image_model()
    except Exception as e:
        print(f"Error unloading image model: {e}")

    try:
        for win in webview.windows:
            win.destroy()
    except Exception as e:
        print(f"Error destroying webview windows: {e}")
    # If window destruction didn't unblock main, force exit after a moment
    threading.Timer(2.0, lambda: os._exit(0)).start()


# Register on cfg so displays.py can call it without circular imports
cfg.shutdown_fn = shutdown


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------
def load_persistent_settings() -> None:
    """Load settings from persistent.json into the configure module."""
    print("Loading persistent settings...")
    cfg.load_config()
    # Seed session_history from default_history for the initial session
    cfg.session_history = cfg.default_history
    print(
        f"  Agent 1: {cfg.agent1_name} | Agent 2: {cfg.agent2_name} | "
        f"Human: {cfg.human_name} | "
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


def idle_unload_watcher() -> None:
    """
    Background daemon thread — unloads models after IDLE_UNLOAD_SECONDS of
    user inactivity.

    The idle clock is measured by cfg.user_turn_start_time:
      • None  → model is processing (not user's turn); do nothing.
      • float → timestamp of when control returned to the user.

    When the elapsed time exceeds cfg.IDLE_UNLOAD_SECONDS, any loaded model
    is unloaded and user_turn_start_time is reset to None so we don't try
    to unload again.  The next call to chat_with_model() will lazy-reload
    the models automatically.
    """
    print("Idle-unload watcher started "
          f"(timeout: {cfg.IDLE_UNLOAD_SECONDS // 60} min).")
    while True:
        try:
            t = cfg.user_turn_start_time
            if t is not None:
                elapsed = time.time() - t
                if elapsed >= cfg.IDLE_UNLOAD_SECONDS:
                    print(f"User idle for {elapsed:.0f}s — unloading models.")
                    cfg.user_turn_start_time = None  # reset before unloading
                    if cfg.text_model_loaded:
                        unload_text_model()
                    if cfg.image_model_loaded:
                        unload_image_model()
                    print("Models unloaded due to inactivity. "
                          "They will reload automatically on next message.")
        except Exception as e:
            print(f"Idle watcher error: {e}")
        time.sleep(30)   # check every 30 seconds


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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

    # Idle-unload watcher (daemon thread — unloads models after inactivity)
    idle_watcher = threading.Thread(target=idle_unload_watcher, daemon=True)
    idle_watcher.start()

    # Start Gradio server (non-blocking, returns URL)
    local_url = launch_gradio_interface()
    if not local_url:
        print("FATAL: Gradio server failed to start.")
        sys.exit(1)

    # Open the pywebview application window
    print(f"Opening application window → {local_url}")
    icon_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "rpggradiogguf_icon.ico"
    )
    if not os.path.isfile(icon_path):
        print(f"  Icon not found at {icon_path} — using default icon.")
        icon_path = None

    window = webview.create_window(
        title="Rpg-Gradio-Gguf",
        url=local_url,
        width=1280,
        height=860,
        resizable=True,
        min_size=(900, 600),
    )

    # Callback executed once the window is fully shown — sets the
    # titlebar and taskbar icon via Win32 API (ctypes).
    def _on_window_shown():
        if icon_path:
            # Small delay to ensure the window's HWND is fully initialised
            time.sleep(0.5)
            _set_window_icon_win32(icon_path, window_title="Rpg-Gradio-Gguf")

    # webview.start() blocks the main thread until ALL windows are closed.
    # This happens when the user clicks [X] or "Exit Program" calls shutdown().
    # The func parameter runs in a background thread after the first window
    # is shown — we use it to set the icon via the Win32 API.
    webview.start(func=_on_window_shown)

    # If we reach here, the window was closed — terminate everything.
    print("Application window closed. Exiting.")
    os._exit(0)


if __name__ == "__main__":
    main()