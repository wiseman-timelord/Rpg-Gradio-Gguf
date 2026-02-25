# scripts/displays.py
# Gradio UI: Conversation tab and Configuration tab.
#
# The Gradio server always starts non-blocking and returns a local URL.
# launcher.py is responsible for opening the pywebview window that points
# at this URL, and for the shutdown/exit lifecycle.
#
# UPDATED: When the image model folder is changed (browse or save),
# ae.safetensors is automatically copied into the new folder if missing.
#
# LAYOUT (Conversation tab)
# ─────────────────────────
# Left column  — "Engagement" radio: Interactions | Happenings | Personalize
#   • Interactions  : Scenario Log + Your Message + Send / Cancel Response
#   • Happenings    : Consolidated History  (no Restart button here)
#   • Personalize   : RP settings form
#
# Right column — "Visualizer" radio: No Generation | Z-Image-Turbo
#   • No Generation  : shows default/last image; skips all image-gen steps
#   • Z-Image-Turbo  : full pipeline (image prompt → generate image)
#
# Status bar row (below both columns):
#   [Status textbox ............] [Restart Session] [Exit Program]
#
# Send/Cancel toggle:
#   Clicking Send hides the Send button and shows Cancel Response.
#   Clicking Cancel sets cfg.cancel_processing = True and re-shows Send.
#   Each inference step in chat_with_model checks the flag before starting.
#
# Image generation is ONLY triggered when the Visualizer is set to
# "Z-Image-Turbo".  "Your Message" input is cleared after submit.
# Personalize panel uses default_history (saved template) separately
# from session_history (running consolidated narrative).
# Restore RP Defaults resets to hardcoded installer defaults and saves.
# Browser spellcheck context-menu is enabled on all text inputs.

import os
import time
import threading
import gradio as gr

from scripts import configure as cfg
from scripts.utilities import (
    reset_session_state,
    browse_folder,
    detect_gpus,
    detect_cpus,
    ensure_vae_in_image_folder,
    scan_for_gguf,
)
from scripts.inference import (
    prompt_response,
    generate_image,
    ensure_models_loaded,
    get_image_gen_progress,
    get_active_agents,
)


# -----------------------------------------------------------------------
# Run hardware detection once at import time
# -----------------------------------------------------------------------
cfg.DETECTED_GPUS = detect_gpus()
cfg.DETECTED_CPUS = detect_cpus()


# -----------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------
def shutdown() -> None:
    """
    Request application shutdown.

    Delegates to cfg.shutdown_fn which is set by launcher.py at startup.
    That function destroys the pywebview window, which unblocks
    webview.start() in launcher.py, leading to os._exit(0).
    """
    if cfg.shutdown_fn is not None:
        cfg.shutdown_fn()
    else:
        # Safety fallback — should never happen in normal operation
        print("WARNING: shutdown_fn not registered. Force exiting.")
        os._exit(1)


def reset_session():
    """Reload config defaults and clear session state."""
    cfg.load_config()
    reset_session_state()
    cfg.consolidated_instance = ""
    placeholder = "./data/new_session.jpg"
    if not os.path.exists(placeholder):
        placeholder = None
    # Returns: instance_display (clear), session_display, bot_response,
    #          generated_image, conv_status, user_input (clear),
    #          sequence_gallery (clear — newest first)
    return "", cfg.session_history, "", placeholder, "Session restarted.", "", []


def cancel_response():
    """
    Signal the current inference pipeline to abort.

    Sets cfg.cancel_processing = True so that each subsequent call to
    prompt_response() or generate_image() in the pipeline skips its work.
    Also restores the Send / Cancel button visibility immediately.

    Returns (status_text, send_btn_update, cancel_btn_update).
    """
    cfg.cancel_processing = True
    return (
        "Cancelling…",
        gr.update(visible=True),   # show Send Message
        gr.update(visible=False),  # hide Cancel Response
    )


def filter_model_output(raw: str, agent_name: str | None = None) -> str:
    """Prefix the responding agent name if the model didn't already include it.

    Parameters
    ----------
    agent_name : str, optional
        The name to prefix.  Falls back to cfg.agent1_name when not supplied
        (preserves backward compatibility with any external callers).
    """
    name = (agent_name or cfg.agent1_name or "Agent").strip()
    if ":" in raw[:40]:
        _, _, after = raw.partition(":")
        return f"{name}: {after.strip()}"
    return f"{name}: {raw}"


def chat_with_model(user_input: str, right_mode: str):
    """
    Main conversation loop: converse → consolidate → [image_prompt → image].

    Image generation (steps 3 & 4) is only performed when right_mode is
    "Z-Image-Turbo".  When right_mode is "No Generation" the pipeline stops
    after consolidation and keeps the existing latest_image_path on screen.

    This is a **generator** — it yields partial updates at each pipeline
    stage so the Gradio UI refreshes progressively instead of blocking
    until the entire pipeline finishes.

    Yields 9 values every iteration:
        (scenario_log, instance_display, session_history, image, status,
         user_input, sequence_gallery, send_btn_update, cancel_btn_update)

    While the pipeline is running the Send button is hidden and the Cancel
    Response button is visible.  Both are restored before the generator
    returns so the user can type another message.

    cfg.cancel_processing is reset to False at the start of each call and
    is set to True by cancel_response() when the user clicks Cancel.  Each
    inference step checks the flag before starting; if set, remaining steps
    are skipped and control is returned to the user.
    """

    # Convenience aliases for button update tuples
    _SEND_HIDDEN   = gr.update(visible=False)
    _SEND_VISIBLE  = gr.update(visible=True)
    _CANCEL_HIDDEN = gr.update(visible=False)
    _CANCEL_VISIBLE = gr.update(visible=True)

    def _restore_buttons():
        """Return (send_update, cancel_update) that restore the normal state."""
        return _SEND_VISIBLE, _CANCEL_HIDDEN

    # Reset cancel flag for this new turn
    cfg.cancel_processing = False

    # ── Empty input guard — never hide Send or alter the Scenario Log ─────
    if not user_input.strip():
        yield (
            cfg.scenario_log or "",          # preserve Scenario Log exactly
            cfg.consolidated_instance or "",
            cfg.session_history or "",
            cfg.latest_image_path,
            "Please type a message first.",
            user_input,                      # keep whatever the user typed
            list(reversed(cfg.session_image_paths)),
            _SEND_VISIBLE,                   # Send stays visible
            _CANCEL_HIDDEN,                  # Cancel stays hidden
        )
        return

    # ── Pre-flight: verify model files exist before touching any session state
    # This runs BEFORE hiding the Send button so that a misconfigured model
    # folder never leaves the UI in a broken button state.
    need_image = (right_mode == "Z-Image-Turbo")

    if not scan_for_gguf(cfg.text_model_folder):
        # Text model is always required — fail immediately, keep buttons intact.
        yield (
            cfg.scenario_log or "",
            cfg.consolidated_instance or "",
            cfg.session_history or "",
            cfg.latest_image_path,
            "No text model found. Check Configuration → Text Model Folder.",
            user_input,                      # preserve user's typed message
            list(reversed(cfg.session_image_paths)),
            _SEND_VISIBLE,
            _CANCEL_HIDDEN,
        )
        return

    if need_image and not scan_for_gguf(cfg.image_model_folder):
        # Image model required for Z-Image-Turbo — fail immediately.
        yield (
            cfg.scenario_log or "",
            cfg.consolidated_instance or "",
            cfg.session_history or "",
            cfg.latest_image_path,
            "No image model found. Check Configuration → Image Model Folder, "
            "or switch Visualizer to 'No Generation'.",
            user_input,                      # preserve user's typed message
            list(reversed(cfg.session_image_paths)),
            _SEND_VISIBLE,
            _CANCEL_HIDDEN,
        )
        return

    # Clear the idle timer — it is now the model's turn, not the user's.
    cfg.user_turn_start_time = None

    # ── Lazy-load / verify models before touching any session state ───────
    # Trigger a load check when:
    #   • the text model is not yet loaded (always required), OR
    #   • the image model is not yet loaded AND this turn needs it.
    # This also catches the case where the text model is loaded but the
    # image model was never loaded (or was idle-unloaded) while the
    # Visualizer is set to "Z-Image-Turbo".
    models_need_check = (
        not cfg.text_model_loaded
        or (need_image and not cfg.image_model_loaded)
    )
    if models_need_check:
        yield (
            cfg.scenario_log or "",
            cfg.consolidated_instance or "",
            cfg.session_history or "",
            cfg.latest_image_path,
            "Loading models… please wait.",
            "",                              # clear input — user message is sending
            list(reversed(cfg.session_image_paths)),
            _SEND_HIDDEN,
            _CANCEL_VISIBLE,
        )
        ok, load_msg = ensure_models_loaded(need_image=need_image)
        if not ok:
            cfg.user_turn_start_time = time.time()
            # Restore everything — the status bar carries the descriptive error.
            yield (
                cfg.scenario_log or "",
                cfg.consolidated_instance or "",
                cfg.session_history or "",
                cfg.latest_image_path,
                load_msg,
                "",
                list(reversed(cfg.session_image_paths)),
                *_restore_buttons(),
            )
            return

    cfg.human_input = user_input.strip()

    # Append the user's line to the running scenario log immediately
    cfg.scenario_log = (cfg.scenario_log + f"\n{cfg.human_name}: {cfg.human_input}").lstrip()

    # Show the user's line right away — clear the input box — show Cancel
    yield (
        cfg.scenario_log,
        cfg.consolidated_instance or "",
        cfg.session_history,
        cfg.latest_image_path,
        "Generating response…",
        "",                              # clear user input
        list(reversed(cfg.session_image_paths)),
        _SEND_HIDDEN,
        _CANCEL_VISIBLE,
    )

    # --- Step 1: converse — one inference call per active agent ------------------
    # get_active_agents() returns only agents with non-blank names, in slot
    # order (Agent 1 → 2 → 3), with a single fallback entry when none are set.
    active_agents = get_active_agents()
    exchange_lines: list[str] = []   # accumulates "AgentName: response" lines

    for agent_name, agent_role in active_agents:

        if cfg.cancel_processing:
            cfg.user_turn_start_time = time.time()
            yield (
                cfg.scenario_log,
                cfg.consolidated_instance or "",
                cfg.session_history,
                cfg.latest_image_path,
                "Response cancelled.",
                "",
                list(reversed(cfg.session_image_paths)),
                *_restore_buttons(),
            )
            return

        # Inform the user which agent is thinking
        agent_count = len(active_agents)
        if agent_count > 1:
            status_msg = f"Generating response… ({agent_name})"
        else:
            status_msg = "Generating response…"

        yield (
            cfg.scenario_log,
            cfg.consolidated_instance or "",
            cfg.session_history,
            cfg.latest_image_path,
            status_msg,
            "",
            list(reversed(cfg.session_image_paths)),
            _SEND_HIDDEN,
            _CANCEL_VISIBLE,
        )

        result = prompt_response("converse", responding_agent=(agent_name, agent_role))

        if cfg.cancel_processing or result.get("cancelled"):
            cfg.user_turn_start_time = time.time()
            yield (
                cfg.scenario_log,
                cfg.consolidated_instance or "",
                cfg.session_history,
                cfg.latest_image_path,
                "Response cancelled.",
                "",
                list(reversed(cfg.session_image_paths)),
                *_restore_buttons(),
            )
            return

        if "error" in result:
            cfg.user_turn_start_time = time.time()
            yield (
                cfg.scenario_log,
                cfg.consolidated_instance or "",
                cfg.session_history,
                cfg.latest_image_path,
                f"Converse error ({agent_name}): {result['error']}",
                "",
                list(reversed(cfg.session_image_paths)),
                *_restore_buttons(),
            )
            return

        # Format and append this agent's line
        formatted_line = filter_model_output(result["agent_response"], agent_name=agent_name)
        exchange_lines.append(formatted_line)
        cfg.scenario_log = cfg.scenario_log + f"\n{formatted_line}"

        # Show this agent's line in the UI immediately before moving to
        # the next agent, so dialogue streams in naturally.
        yield (
            cfg.scenario_log,
            cfg.consolidated_instance or "",
            cfg.session_history,
            cfg.latest_image_path,
            status_msg,
            "",
            list(reversed(cfg.session_image_paths)),
            _SEND_HIDDEN,
            _CANCEL_VISIBLE,
        )

    # Expose the combined exchange for the consolidate / instance steps.
    # agent_output keeps the last agent's raw line (legacy compat);
    # agent_exchange holds the full multi-agent block.
    cfg.agent_output   = exchange_lines[-1] if exchange_lines else ""
    cfg.agent_exchange = "\n".join(exchange_lines)

    # --- Step 2a: instance — visual snapshot of this rotation ----------------
    # Runs before consolidate so the image pipeline can use the result.
    if cfg.cancel_processing:
        cfg.user_turn_start_time = time.time()
        yield (
            cfg.scenario_log,
            cfg.consolidated_instance or "",
            cfg.session_history,
            cfg.latest_image_path,
            "Response cancelled.",
            "",
            list(reversed(cfg.session_image_paths)),
            *_restore_buttons(),
        )
        return

    yield (
        cfg.scenario_log,
        cfg.consolidated_instance or "",
        cfg.session_history,
        cfg.latest_image_path,
        "Generating instance summary…",
        "",
        list(reversed(cfg.session_image_paths)),
        _SEND_HIDDEN,
        _CANCEL_VISIBLE,
    )

    instance_result = prompt_response("instance")
    if cfg.cancel_processing or instance_result.get("cancelled"):
        cfg.user_turn_start_time = time.time()
        yield (
            cfg.scenario_log,
            cfg.consolidated_instance or "",
            cfg.session_history,
            cfg.latest_image_path,
            "Response cancelled.",
            "",
            list(reversed(cfg.session_image_paths)),
            *_restore_buttons(),
        )
        return

    if "agent_response" in instance_result:
        cfg.consolidated_instance = instance_result["agent_response"]

    # --- Step 2b: consolidate — update running narrative history -------------
    yield (
        cfg.scenario_log,
        cfg.consolidated_instance or "",
        cfg.session_history,
        cfg.latest_image_path,
        "Generating history…",
        "",
        list(reversed(cfg.session_image_paths)),
        _SEND_HIDDEN,
        _CANCEL_VISIBLE,
    )

    consolidate = prompt_response("consolidate")
    if not cfg.cancel_processing and not consolidate.get("cancelled") and "error" not in consolidate:
        cfg.session_history = consolidate["agent_response"]

    # ── Image generation — only when Visualizer mode is "Z-Image-Turbo" ──────────
    if cfg.cancel_processing or right_mode != "Z-Image-Turbo":
        cfg.user_turn_start_time = time.time()
        status = "Response cancelled." if cfg.cancel_processing else "Response generated."
        yield (
            cfg.scenario_log,
            cfg.consolidated_instance or "",
            cfg.session_history,
            cfg.latest_image_path,
            status,
            "",
            list(reversed(cfg.session_image_paths)),
            *_restore_buttons(),
        )
        return

    # --- Step 3: generate the visual prompt --------------------------------------
    yield (
        cfg.scenario_log,
        cfg.consolidated_instance or "",
        cfg.session_history,
        cfg.latest_image_path,
        "Generating prompt…",
        "",
        list(reversed(cfg.session_image_paths)),
        _SEND_HIDDEN,
        _CANCEL_VISIBLE,
    )

    visual_prompt = None
    if cfg.text_model_loaded:
        img_prompt_result = prompt_response("image_prompt")
        if not cfg.cancel_processing and not img_prompt_result.get("cancelled"):
            if "agent_response" in img_prompt_result:
                visual_prompt = img_prompt_result["agent_response"]

    if cfg.cancel_processing:
        cfg.user_turn_start_time = time.time()
        yield (
            cfg.scenario_log,
            cfg.consolidated_instance or "",
            cfg.session_history,
            cfg.latest_image_path,
            "Response cancelled.",
            "",
            list(reversed(cfg.session_image_paths)),
            *_restore_buttons(),
        )
        return

    if not visual_prompt:
        visual_prompt = f"A scene at {cfg.scene_location}."

    # --- Step 4: generate the image (threaded with progress polling) --------------
    yield (
        cfg.scenario_log,
        cfg.consolidated_instance or "",
        cfg.session_history,
        cfg.latest_image_path,
        "Generating image…",
        "",
        list(reversed(cfg.session_image_paths)),
        _SEND_HIDDEN,
        _CANCEL_VISIBLE,
    )

    # Run image generation in a background thread so we can poll progress
    _img_result: dict = {"path": None, "done": False}

    def _worker():
        _img_result["path"] = generate_image(scene_prompt=visual_prompt)
        _img_result["done"] = True

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    # Poll progress and yield status updates every ~1 second.
    # If cancel is requested, stop polling (the worker thread will check the
    # flag itself at its next opportunity and return early).
    while not _img_result["done"]:
        if cfg.cancel_processing:
            # Wait for the thread to finish cleanly then bail out
            thread.join()
            cfg.user_turn_start_time = time.time()
            yield (
                cfg.scenario_log,
                cfg.consolidated_instance or "",
                cfg.session_history,
                cfg.latest_image_path,
                "Response cancelled.",
                "",
                list(reversed(cfg.session_image_paths)),
                *_restore_buttons(),
            )
            return
        step, total = get_image_gen_progress()
        if total > 0 and step > 0:
            status = f"Generating image, step {step}/{total}…"
        else:
            status = "Generating image…"
        yield (
            cfg.scenario_log,
            cfg.consolidated_instance or "",
            cfg.session_history,
            cfg.latest_image_path,
            status,
            "",
            list(reversed(cfg.session_image_paths)),
            _SEND_HIDDEN,
            _CANCEL_VISIBLE,
        )
        time.sleep(1.0)

    thread.join()
    image_path = _img_result["path"]
    cfg.latest_image_path = image_path

    # Control returns to user — start the idle timer.
    cfg.user_turn_start_time = time.time()

    yield (
        cfg.scenario_log,
        cfg.consolidated_instance or "",
        cfg.session_history,
        image_path,
        "Response generated with image." if image_path else "Response generated (image failed).",
        "",
        list(reversed(cfg.session_image_paths)),
        *_restore_buttons(),
    )


# -----------------------------------------------------------------------
# Panel switching callbacks
# -----------------------------------------------------------------------
def switch_left_panel(mode: str):
    """Toggle visibility of Interactions / Happenings / Personalize panels."""
    return (
        gr.update(visible=(mode == "Interactions")),
        gr.update(visible=(mode == "Happenings")),
        gr.update(visible=(mode == "Personalize")),
    )


def switch_right_panel_and_state(mode: str):
    """
    Update the Visualizer mode state.

    The right column is always the Visualizer (image display); only the
    generation behaviour changes based on the selected mode.  No panel
    visibility toggling is required — we just store the state so that
    chat_with_model knows whether to run image generation.
    """
    return mode


def on_gallery_select(evt: gr.SelectData):
    """When the user clicks a thumbnail in the Sequence strip, display it
    in the Generated Image box.  The gallery value list is newest-first."""
    if evt.value and isinstance(evt.value, dict):
        # Gradio 4.x: evt.value is a dict with 'image'/'path' keys
        path = evt.value.get("image", {}).get("path") or evt.value.get("path")
        if path:
            return path
    elif evt.value and isinstance(evt.value, str):
        return evt.value
    return gr.update()  # no change


# -----------------------------------------------------------------------
# Roleplay settings callbacks (Conversation tab — Personalize panel)
# -----------------------------------------------------------------------
def save_roleplay_settings(
    a1_name, a1_role,
    a2_name, a2_role,
    a3_name, a3_role,
    human, human_age, human_gender,
    location, event_time, history,
):
    """Save the roleplay parameters from the Personalize panel."""
    cfg.agent1_name = a1_name
    cfg.agent1_role = a1_role

    cfg.agent2_name = a2_name
    cfg.agent2_role = a2_role

    cfg.agent3_name = a3_name
    cfg.agent3_role = a3_role

    cfg.human_name   = human
    cfg.human_age    = str(human_age) if human_age is not None else ""
    cfg.human_gender = human_gender

    cfg.scene_location  = location
    cfg.event_time      = event_time
    cfg.default_history = history
    cfg.save_config()
    return "RP settings saved."


def restore_roleplay_settings():
    """Restore RP fields to the hardcoded installer defaults.

    Writes the defaults to both the runtime globals AND persistent.json so
    that subsequent startups also use the defaults.
    """
    d = cfg.DEFAULT_RP_SETTINGS
    cfg.agent1_name = d["agent1_name"]
    cfg.agent1_role = d["agent1_role"]

    cfg.agent2_name = d["agent2_name"]
    cfg.agent2_role = d["agent2_role"]

    cfg.agent3_name = d["agent3_name"]
    cfg.agent3_role = d["agent3_role"]

    cfg.human_name    = d["human_name"]
    cfg.human_age     = d["human_age"]
    cfg.human_gender  = d["human_gender"]

    cfg.scene_location = d["scene_location"]
    cfg.event_time     = d["event_time"]
    cfg.default_history = d["default_history"]
    cfg.save_config()
    return (
        cfg.agent1_name,   cfg.agent1_role,
        cfg.agent2_name,   cfg.agent2_role,
        cfg.agent3_name,   cfg.agent3_role,
        cfg.human_name,    cfg.human_age,     cfg.human_gender,
        cfg.scene_location, cfg.event_time,
        cfg.default_history,
        "RP defaults restored and saved.",
    )


# -----------------------------------------------------------------------
# Configuration callbacks
# -----------------------------------------------------------------------
def save_config_tab_settings(
    gpu_choice, vram, cpu_choice, threads, auto_unload, max_mem,
    img_sz, steps, samp, cfgs, neg, tf, imf,
):
    """Save all Configuration-tab fields.  Roleplay values stay in sync
    via the Personalize panel on the Conversation tab.

    When the image model folder changes, ae.safetensors is automatically
    copied into the new folder if it is not already present.
    """

    # Parse GPU index from dropdown string  "GPU0: Name (VRAM MB)"
    try:
        cfg.selected_gpu = int(gpu_choice.split(":")[0].replace("GPU", ""))
    except (ValueError, AttributeError):
        cfg.selected_gpu = 0

    cfg.vram_assigned = int(vram)

    # Parse CPU index
    try:
        cfg.selected_cpu = int(cpu_choice.split(":")[0].replace("CPU", ""))
    except (ValueError, AttributeError):
        cfg.selected_cpu = 0

    cfg.cpu_threads = int(threads)
    cfg.optimal_threads = max(1, cfg.cpu_threads)
    cfg.auto_unload = auto_unload
    cfg.max_memory_percent = int(max_mem)

    cfg.IMAGE_SIZE_OPTIONS["selected_size"] = img_sz
    cfg.selected_steps = int(steps)
    cfg.selected_sample_method = samp
    cfg.selected_cfg_scale = float(cfgs)
    cfg.default_negative_prompt = neg
    cfg.text_model_folder = tf

    # --- Image model folder: auto-copy ae.safetensors if needed ---
    old_image_folder = cfg.image_model_folder
    cfg.image_model_folder = imf

    vae_status = ""
    if imf and os.path.isdir(imf):
        vae_result = ensure_vae_in_image_folder(imf)
        if vae_result:
            vae_status = " ae.safetensors confirmed."
        else:
            vae_status = " WARNING: ae.safetensors missing — re-run installer."

    cfg.save_config()
    return f"Configuration saved.{vae_status}"


def browse_text_folder(current: str) -> str:
    return browse_folder(current)


def browse_image_folder(current: str) -> str:
    """
    Open folder picker for the image model folder.
    After selection, automatically ensure ae.safetensors is present.
    """
    chosen = browse_folder(current)
    if chosen and os.path.isdir(chosen):
        vae = ensure_vae_in_image_folder(chosen)
        if vae:
            print(f"ae.safetensors ready in {chosen}")
        else:
            print(f"WARNING: ae.safetensors could not be placed in {chosen}. "
                  "Re-run the installer to download it.")
    return chosen


# -----------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------
def launch_gradio_interface() -> str | None:
    """
    Build and launch the Gradio Blocks interface.

    The server always starts non-blocking (prevent_thread_lock=True) so
    that launcher.py can open a pywebview window pointing at the URL.

    Returns
    -------
    str | None
        The local URL string, or None on failure.
    """

    # Pre-compute hardware display strings
    gpu_names = [
        f"GPU{gpu['index']}: {gpu['name']} ({gpu['vram_mb']} MB VRAM)"
        for gpu in cfg.DETECTED_GPUS
    ] if cfg.DETECTED_GPUS else ["No GPUs detected"]

    valid_gpu_index = min(cfg.selected_gpu, len(gpu_names) - 1) if gpu_names else 0

    cpu_names = [
        f"CPU{cpu['index']}: {cpu['name']} ({cpu['threads']} threads)"
        for cpu in cfg.DETECTED_CPUS
    ] if cfg.DETECTED_CPUS else ["No CPUs detected"]

    valid_cpu_index = min(cfg.selected_cpu, len(cpu_names) - 1) if cpu_names else 0

    max_threads = (
        cfg.DETECTED_CPUS[valid_cpu_index]["threads"]
        if cfg.DETECTED_CPUS and valid_cpu_index < len(cfg.DETECTED_CPUS)
        else cfg.CPU_THREADS
    )
    valid_threads = min(cfg.cpu_threads, max_threads)

    # Custom JS to enable native browser spellcheck context-menu on all
    # textareas and text inputs — allows right-click spelling suggestions.
    _SPELLCHECK_JS = """
    () => {
        // Enable spellcheck on all textareas and text inputs
        function enableSpellcheck() {
            document.querySelectorAll('textarea, input[type="text"]').forEach(el => {
                el.setAttribute('spellcheck', 'true');
                el.setAttribute('lang', 'en');
            });
        }
        enableSpellcheck();
        // Re-apply when Gradio dynamically adds new elements
        const observer = new MutationObserver(enableSpellcheck);
        observer.observe(document.body, { childList: true, subtree: true });
    }
    """

    # Custom CSS for proper image scaling and gallery thumbnail sizing
    _CUSTOM_CSS = """
    /* ── Generated Image: scale to fit, not stretch, smooth ── */
    #generated_image img {
        object-fit: contain !important;
        width: 100% !important;
        height: 100% !important;
    }

    /* ── Sequence gallery: single-row thumbnail strip ── */
    #sequence_gallery .grid-container,
    #sequence_gallery .grid-wrap {
        /* Force thumbnails to fit within the gallery height */
        gap: 4px !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        flex-wrap: nowrap !important;
    }
    #sequence_gallery .thumbnail-item,
    #sequence_gallery .thumbnail-lg {
        /* Fixed thumbnail cells — square, fitting within 188px strip */
        min-width: 156px !important;
        max-width: 156px !important;
        min-height: 156px !important;
        max-height: 156px !important;
        flex-shrink: 0 !important;
    }
    #sequence_gallery .thumbnail-item img,
    #sequence_gallery .thumbnail-lg img {
        object-fit: contain !important;
        width: 100% !important;
        height: 100% !important;
    }
    """

    with gr.Blocks(
        title="Rpg-Gradio-Gguf",
        js=_SPELLCHECK_JS,
        css=_CUSTOM_CSS,
    ) as interface:
        gr.Markdown("# ⚔️ Rpg-Gradio-Gguf")

        with gr.Tabs():
            # ==============================================================
            # TAB 1 — Conversation
            # ==============================================================
            with gr.Tab("Conversation"):

                # State to track Visualizer mode for image-generation logic
                right_panel_state = gr.State("Z-Image-Turbo")

                with gr.Row():
                    # ── LEFT COLUMN ──────────────────────────────────────
                    with gr.Column(scale=1):
                        left_mode = gr.Radio(
                            choices=["Interactions", "Happenings", "Personalize"],
                            value="Interactions",
                            label="Engagement",
                            interactive=True,
                        )

                        # -- Interactions panel --
                        with gr.Column(visible=True) as interaction_panel:
                            bot_response = gr.Textbox(
                                label="Scenario Log",
                                lines=16,
                                interactive=False,
                            )
                            user_input = gr.Textbox(
                                label="Your Message",
                                lines=6,
                                placeholder="Type your message here...",
                                interactive=True,
                            )
                            send_btn = gr.Button("Send Message", visible=True)
                            cancel_btn = gr.Button(
                                "Cancel Response", variant="stop", visible=False
                            )

                        # -- Happenings panel (Consolidated Instance + Consolidated History) --
                        with gr.Column(visible=False) as happenings_panel:
                            instance_display = gr.Textbox(
                                label="Consolidated Instance",
                                lines=6,
                                value=cfg.consolidated_instance,
                                interactive=False,
                                info="Visual snapshot of the most recent exchange — used for image generation.",
                            )
                            session_display = gr.Textbox(
                                label="Consolidated History",
                                lines=18,
                                value=cfg.session_history,
                                interactive=False,
                            )

                        # -- Personalize panel --
                        with gr.Column(visible=False) as personalize_panel:

                            # Row 1: Agent Names
                            with gr.Row():
                                rp_agent1_name = gr.Textbox(
                                    label="Agent 1 Name",
                                    value=cfg.agent1_name,
                                )
                                rp_agent2_name = gr.Textbox(
                                    label="Agent 2 Name",
                                    value=cfg.agent2_name,
                                )
                                rp_agent3_name = gr.Textbox(
                                    label="Agent 3 Name",
                                    value=cfg.agent3_name,
                                )

                            # Row 2: Agent Roles
                            with gr.Row():
                                rp_agent1_role = gr.Textbox(
                                    label="Agent 1 Role",
                                    value=cfg.agent1_role,
                                )
                                rp_agent2_role = gr.Textbox(
                                    label="Agent 2 Role",
                                    value=cfg.agent2_role,
                                )
                                rp_agent3_role = gr.Textbox(
                                    label="Agent 3 Role",
                                    value=cfg.agent3_role,
                                )

                            # Row 3: Human Name / Age / Gender
                            with gr.Row():
                                rp_human_name = gr.Textbox(
                                    label="Human Name",
                                    value=cfg.human_name,
                                )
                                rp_human_age = gr.Number(
                                    label="Human Age",
                                    value=int(cfg.human_age) if cfg.human_age and cfg.human_age.strip().isdigit() else None,
                                    precision=0,
                                    minimum=0,
                                    maximum=999,
                                )
                                rp_human_gender = gr.Dropdown(
                                    label="Human Gender",
                                    choices=cfg.GENDER_OPTIONS,
                                    value=cfg.human_gender,
                                )

                            # Row 5: Scene Location / Event Time
                            with gr.Row():
                                rp_scene_location = gr.Textbox(
                                    label="Scene Location",
                                    value=cfg.scene_location,
                                )
                                rp_event_time = gr.Textbox(
                                    label="Event Time",
                                    value=cfg.event_time,
                                    placeholder="e.g. 16:20, 4:20pm, dawn — leave blank to omit",
                                )

                            rp_default_history = gr.Textbox(
                                label="Starting Narrative",
                                value=cfg.default_history,
                                lines=3,
                            )
                            with gr.Row():
                                rp_save_btn = gr.Button(
                                    "Save RP Settings", variant="primary"
                                )
                                rp_restore_btn = gr.Button(
                                    "Restore RP Defaults", variant="stop"
                                )

                    # ── RIGHT COLUMN — Visualizer ─────────────────────────
                    with gr.Column(scale=1):
                        right_mode = gr.Radio(
                            choices=["No Generation", "Z-Image-Turbo"],
                            value="Z-Image-Turbo",
                            label="Visualizer",
                            interactive=True,
                        )

                        # The image display is always visible; only generation
                        # behaviour changes with the Visualizer mode.
                        default_img = (
                            "./data/new_session.jpg"
                            if os.path.exists("./data/new_session.jpg")
                            else None
                        )
                        generated_image = gr.Image(
                            label="Generated Image",
                            type="filepath",
                            interactive=False,
                            height=420,
                            value=default_img,
                            elem_id="generated_image",
                        )
                        # Single-row thumbnail strip — newest image first.
                        # Height accommodates ~100px thumbnails + label/padding.
                        # object_fit="contain" scales each thumbnail to fit
                        # within its cell without cropping.
                        sequence_gallery = gr.Gallery(
                            label="Sequence",
                            columns=8,
                            rows=1,
                            height=188,
                            object_fit="contain",
                            allow_preview=False,
                            value=list(reversed(cfg.session_image_paths)),
                            elem_id="sequence_gallery",
                        )

                # ── Status bar + Restart + Exit ──────────────────────────
                with gr.Row():
                    conv_status = gr.Textbox(
                        value="Ready.",
                        label="Status",
                        interactive=False,
                        max_lines=1,
                        scale=20,
                    )
                    with gr.Column(scale=3, min_width=240):
                        restart_btn = gr.Button(
                            "Restart Session",
                            variant="primary",
                        )
                        exit_conv = gr.Button(
                            "Exit Program",
                            variant="stop",
                        )

                # ── Left panel switching (3 panels) ──────────────────────
                left_mode.change(
                    fn=switch_left_panel,
                    inputs=left_mode,
                    outputs=[interaction_panel, happenings_panel, personalize_panel],
                )

                # ── Right Visualizer mode → update state ─────────────────
                right_mode.change(
                    fn=switch_right_panel_and_state,
                    inputs=right_mode,
                    outputs=right_panel_state,
                )

                # ── Gallery thumbnail click → show in Generated Image ────
                sequence_gallery.select(
                    fn=on_gallery_select,
                    inputs=None,
                    outputs=generated_image,
                )

                # ── Send message ─────────────────────────────────────────
                # Outputs include user_input (to clear it),
                # sequence_gallery (to update the image list), and
                # send_btn / cancel_btn (to toggle visibility).
                #
                # The .then() callback fires after the generator stream ends
                # and explicitly restores the Send / Cancel button state.
                # This is necessary because Gradio streaming generators do not
                # reliably apply UI updates from the final yield; the .then()
                # ensures Cancel is always hidden once the pipeline is done.
                send_btn.click(
                    fn=chat_with_model,
                    inputs=[user_input, right_panel_state],
                    outputs=[
                        bot_response,
                        instance_display,
                        session_display,
                        generated_image,
                        conv_status,
                        user_input,
                        sequence_gallery,
                        send_btn,
                        cancel_btn,
                    ],
                ).then(
                    fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                    inputs=[],
                    outputs=[send_btn, cancel_btn],
                )

                # ── Cancel response ───────────────────────────────────────
                cancel_btn.click(
                    fn=cancel_response,
                    inputs=[],
                    outputs=[conv_status, send_btn, cancel_btn],
                )

                # ── Restart Session (status bar button) ──────────────────
                _reset_outputs = [
                    instance_display,
                    session_display,
                    bot_response,
                    generated_image,
                    conv_status,
                    user_input,
                    sequence_gallery,
                ]
                restart_btn.click(
                    fn=reset_session, inputs=[], outputs=_reset_outputs
                )

                # ── Save RP settings ─────────────────────────────────────
                rp_save_btn.click(
                    fn=save_roleplay_settings,
                    inputs=[
                        rp_agent1_name, rp_agent1_role,
                        rp_agent2_name, rp_agent2_role,
                        rp_agent3_name, rp_agent3_role,
                        rp_human_name, rp_human_age, rp_human_gender,
                        rp_scene_location, rp_event_time,
                        rp_default_history,
                    ],
                    outputs=conv_status,
                )

                # ── Restore RP settings ──────────────────────────────────
                rp_restore_btn.click(
                    fn=restore_roleplay_settings,
                    inputs=[],
                    outputs=[
                        rp_agent1_name,   rp_agent1_role,
                        rp_agent2_name,   rp_agent2_role,
                        rp_agent3_name,   rp_agent3_role,
                        rp_human_name,    rp_human_age,     rp_human_gender,
                        rp_scene_location, rp_event_time,
                        rp_default_history,
                        conv_status,
                    ],
                )

                # ── Exit ─────────────────────────────────────────────────
                exit_conv.click(fn=shutdown, inputs=[], outputs=[])

            # ==============================================================
            # TAB 2 — Configuration
            # ==============================================================
            with gr.Tab("Configuration"):

                # ── HARDWARE & BACKEND ───────────────────────────────────
                with gr.Group():
                    gr.Markdown("### Hardware & Backend")
                    with gr.Row():
                        selected_gpu_dropdown = gr.Dropdown(
                            choices=gpu_names,
                            value=gpu_names[valid_gpu_index],
                            label="Selected GPU",
                            info="Select which GPU to use for inference. User sets VRAM allocation.",
                            interactive=(len(gpu_names) > 1),
                        )
                        vram_options_str = [str(v) for v in cfg.VRAM_OPTIONS]
                        valid_vram = (
                            cfg.vram_assigned
                            if cfg.vram_assigned in cfg.VRAM_OPTIONS
                            else 8192
                        )
                        selected_vram_dropdown = gr.Dropdown(
                            choices=vram_options_str,
                            value=str(valid_vram),
                            label="VRAM Allocation (MB)",
                            info="VRAM budget for both text + image models. Layers auto-calculated.",
                            interactive=True,
                        )
                    with gr.Row():
                        selected_cpu_dropdown = gr.Dropdown(
                            choices=cpu_names,
                            value=cpu_names[valid_cpu_index],
                            label="Selected CPU",
                            info="Select the CPU to use for inference. If only one is available, this is fixed.",
                            interactive=(len(cpu_names) > 1),
                        )
                        threads_assigned_slider = gr.Slider(
                            minimum=1,
                            maximum=max_threads,
                            step=1,
                            value=valid_threads,
                            label="Threads Assigned",
                            info=f"Number of threads for inference. System has {cfg.CPU_THREADS} total threads.",
                        )
                    with gr.Row():
                        auto_unload_check = gr.Checkbox(
                            label="Auto-unload model on high SYSTEM RAM",
                            value=cfg.auto_unload,
                        )
                        max_memory_slider = gr.Slider(
                            minimum=50,
                            maximum=95,
                            step=5,
                            value=cfg.max_memory_percent,
                            label="Max System RAM % (unload threshold)",
                            info="Triggers model unload when system RAM exceeds this percentage (NOT VRAM)",
                        )

                # ── MODEL FOLDERS ────────────────────────────────────────
                gr.Markdown("### Model Folders") # Text model also is image encoder. ae.safetensors auto-copied to image folder when set.
                with gr.Row():
                    text_folder_in = gr.Textbox(
                        label="Text Model Folder",
                        value=cfg.text_model_folder,
                        scale=4,
                    )
                    text_browse_btn = gr.Button("Browse...", scale=1)
                    image_folder_in = gr.Textbox(
                        label="Image Model Folder",
                        value=cfg.image_model_folder,
                        scale=4,
                    )
                    image_browse_btn = gr.Button("Browse...", scale=1)

                text_browse_btn.click(
                    fn=browse_text_folder,
                    inputs=text_folder_in,
                    outputs=text_folder_in,
                )
                image_browse_btn.click(
                    fn=browse_image_folder,
                    inputs=image_folder_in,
                    outputs=image_folder_in,
                )

                # ── IMAGE GENERATION ─────────────────────────────────────
                with gr.Group():
                    gr.Markdown("### Image Generation (Z-Image-Turbo)")
                    with gr.Row():
                        image_size_dd = gr.Dropdown(
                            label="Image Size",
                            choices=cfg.IMAGE_SIZE_OPTIONS["available_sizes"],
                            value=cfg.IMAGE_SIZE_OPTIONS["selected_size"],
                            type="value",
                        )
                        steps_dd = gr.Dropdown(
                            label="Image Steps",
                            choices=[str(s) for s in cfg.STEPS_OPTIONS],
                            value=str(cfg.selected_steps),
                            type="value",
                            info="Z-Image-Turbo is optimised for 8 steps.",
                        )
                        sample_dd = gr.Dropdown(
                            label="Sample Method",
                            choices=cfg.SAMPLE_METHOD_OPTIONS,
                            value=cfg.selected_sample_method,
                            type="value",
                        )
                        cfg_scale_dd = gr.Dropdown(
                            label="CFG Scale",
                            choices=[str(c) for c in cfg.CFG_SCALE_OPTIONS],
                            value=str(cfg.selected_cfg_scale),
                            type="value",
                            info="Use 1.0 for Z-Image-Turbo (distilled). Higher = stronger prompt adherence.",
                        )
                    negative_prompt_in = gr.Textbox(
                        label="Negative Prompt (Z-Image-Turbo mostly ignores this)",
                        value=cfg.default_negative_prompt,
                        lines=1,
                        placeholder="Leave empty for best results with Z-Image-Turbo",
                    )

                # ── Status bar + Save + Exit ─────────────────────────────
                with gr.Row():
                    cfg_status = gr.Textbox(
                        value="Configuration tab loaded.",
                        label="Status",
                        interactive=False,
                        max_lines=1,
                        scale=20,
                    )
                    with gr.Column(scale=3, min_width=240):
                        save_btn = gr.Button(
                            "Save Configuration", variant="primary"
                        )
                        exit_cfg = gr.Button(
                            "Exit Program",
                            variant="stop",
                        )

                save_btn.click(
                    fn=save_config_tab_settings,
                    inputs=[
                        selected_gpu_dropdown,
                        selected_vram_dropdown,
                        selected_cpu_dropdown,
                        threads_assigned_slider,
                        auto_unload_check,
                        max_memory_slider,
                        image_size_dd,
                        steps_dd,
                        sample_dd,
                        cfg_scale_dd,
                        negative_prompt_in,
                        text_folder_in,
                        image_folder_in,
                    ],
                    outputs=cfg_status,
                )

                exit_cfg.click(fn=shutdown, inputs=[], outputs=[])

    # ------------------------------------------------------------------
    # Start the Gradio server (non-blocking, no browser)
    # ------------------------------------------------------------------
    # queue() is required for generator functions (like chat_with_model)
    # to stream partial results back to the UI progressively.
    interface.queue()
    _app, local_url, _share_url = interface.launch(
        inbrowser=False,
        prevent_thread_lock=True,
        server_name="127.0.0.1",
    )
    print(f"Gradio server started at {local_url}")
    return local_url