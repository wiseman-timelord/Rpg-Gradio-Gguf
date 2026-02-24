# scripts/displays.py
# Gradio UI: Conversation tab and Configuration tab.
#
# The Gradio server always starts non-blocking and returns a local URL.
# launcher.py is responsible for opening the pywebview window that points
# at this URL, and for the shutdown/exit lifecycle.
#
# UPDATED: When the image model folder is changed (browse or save),
# ae.safetensors is automatically copied into the new folder if missing.

import os
import time
import gradio as gr

from scripts import configure as cfg
from scripts.utilities import (
    reset_session_state,
    browse_folder,
    detect_gpus,
    detect_cpus,
    ensure_vae_in_image_folder,
)
from scripts.inference import prompt_response, generate_image, ensure_models_loaded


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
    placeholder = "./data/new_session.jpg"
    if not os.path.exists(placeholder):
        placeholder = None
    return cfg.session_history, "", placeholder, "Session restarted."


def filter_model_output(raw: str) -> str:
    """Prefix the agent name if the model didn't already include it."""
    if ":" in raw[:40]:
        _, _, after = raw.partition(":")
        return f"{cfg.agent_name}: {after.strip()}"
    return f"{cfg.agent_name}: {raw}"


def chat_with_model(user_input: str, right_mode: str):
    """
    Main conversation loop: converse → consolidate → image.
    If the right panel is on 'Happenings', skip image generation.

    Models are loaded on-demand (lazy) if they are not already in memory.
    After a response is delivered, the idle timer is started so the watcher
    thread can unload models after IDLE_UNLOAD_SECONDS of inactivity.
    """
    if not user_input.strip():
        return (
            "Please type a message.",
            cfg.session_history,
            None,
            "Waiting for input...",
        )

    # Clear the idle timer — it is now the model's turn, not the user's.
    cfg.user_turn_start_time = None

    # ── Lazy-load models if needed ────────────────────────────────────────
    if not cfg.text_model_loaded:
        ok, load_msg = ensure_models_loaded()
        if not ok:
            # Restore idle timer so the watcher doesn't spin forever
            cfg.user_turn_start_time = time.time()
            return (
                "Models could not be loaded. Check the Configuration tab.",
                cfg.session_history,
                None,
                load_msg,
            )

    cfg.human_input = user_input.strip()   # raw text — templates add the name prefix

    # Append the user's line to the running scenario log immediately
    cfg.scenario_log = (cfg.scenario_log + f"\n{cfg.human_name}: {cfg.human_input}").lstrip()

    # --- Step 1: converse --------------------------------------------------------
    result = prompt_response("converse")
    if "error" in result:
        cfg.user_turn_start_time = time.time()
        return (
            cfg.scenario_log,
            cfg.session_history,
            None,
            f"Converse error: {result['error']}",
        )

    filtered = filter_model_output(result["agent_response"])
    cfg.agent_output = filtered

    # Append the agent's response to the scenario log
    cfg.scenario_log = cfg.scenario_log + f"\n{cfg.agent_output}"

    # --- Step 2: consolidate -----------------------------------------------------
    consolidate = prompt_response("consolidate")
    if "error" not in consolidate:
        cfg.session_history = consolidate["agent_response"]

    # --- Step 3: image (only if Visualizer is active) ----------------------------
    image_path = None
    if right_mode == "Visualizer":
        image_path = generate_image()
        cfg.latest_image_path = image_path
        status_msg = "Response generated with image."
    else:
        status_msg = "Response generated. Image skipped (Visualizer not active)."

    # Control returns to user — start the idle timer.
    cfg.user_turn_start_time = time.time()

    return cfg.scenario_log, cfg.session_history, image_path, status_msg


# -----------------------------------------------------------------------
# Panel switching callbacks
# -----------------------------------------------------------------------
def switch_left_panel(mode: str):
    """Toggle visibility of Interactions vs Personalize panels."""
    if mode == "Interactions":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)


def switch_right_panel_and_state(mode: str):
    """Toggle visibility of Happenings vs Visualizer and update state."""
    if mode == "Happenings":
        return gr.update(visible=True), gr.update(visible=False), mode
    else:
        return gr.update(visible=False), gr.update(visible=True), mode


# -----------------------------------------------------------------------
# Roleplay settings callbacks (Conversation tab — Personalize panel)
# -----------------------------------------------------------------------
def save_roleplay_settings(name, role, human, location, history):
    """Save only the roleplay parameters."""
    cfg.agent_name = name
    cfg.agent_role = role
    cfg.human_name = human
    cfg.scene_location = location
    cfg.session_history = history
    cfg.save_config()
    return "RP settings saved."


def restore_roleplay_settings():
    """Reload roleplay parameters from persistent.json (discard unsaved edits)."""
    cfg.load_config()
    return (
        cfg.agent_name,
        cfg.agent_role,
        cfg.human_name,
        cfg.scene_location,
        cfg.session_history,
        "RP settings restored from file.",
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

    with gr.Blocks(title="Rpg-Gradio-Gguf") as interface:
        gr.Markdown("# ⚔️ Rpg-Gradio-Gguf")

        with gr.Tabs():
            # ==============================================================
            # TAB 1 — Conversation
            # ==============================================================
            with gr.Tab("Conversation"):

                # State to track right panel mode for image-skip logic
                right_panel_state = gr.State("Visualizer")

                with gr.Row():
                    # ── LEFT COLUMN ──────────────────────────────────────
                    with gr.Column(scale=1):
                        left_mode = gr.Radio(
                            choices=["Interactions", "Personalize"],
                            value="Interactions",
                            label="Engagement",
                            interactive=True,
                        )

                        # -- Interaction panel --
                        with gr.Column(visible=True) as interaction_panel:
                            bot_response = gr.Textbox(
                                label="Scenario Log",
                                lines=12,
                                interactive=False,
                            )
                            user_input = gr.Textbox(
                                label="Your Message",
                                lines=6,
                                placeholder="Type your message here...",
                                interactive=True,
                            )

                        # -- Personalize panel (no header) --
                        with gr.Column(visible=False) as personalize_panel:
                            rp_agent_name = gr.Textbox(
                                label="Agent Name",
                                value=cfg.agent_name,
                            )
                            rp_agent_role = gr.Textbox(
                                label="Agent Role",
                                value=cfg.agent_role,
                            )
                            rp_human_name = gr.Textbox(
                                label="Human Name",
                                value=cfg.human_name,
                            )
                            rp_scene_location = gr.Textbox(
                                label="Scene Location",
                                value=cfg.scene_location,
                            )
                            rp_session_history = gr.Textbox(
                                label="Default History",
                                value=cfg.session_history,
                                lines=3,
                            )
                            with gr.Row():
                                rp_save_btn = gr.Button(
                                    "Save RP Settings", variant="primary"
                                )
                                rp_restore_btn = gr.Button(
                                    "Restore RP Defaults"
                                )

                    # ── RIGHT COLUMN ─────────────────────────────────────
                    with gr.Column(scale=1):
                        right_mode = gr.Radio(
                            choices=["Happenings", "Visualizer"],
                            value="Visualizer",
                            label="Chronicler",
                            interactive=True,
                        )

                        # -- Happenings panel --
                        with gr.Column(visible=False) as happenings_panel:
                            session_display = gr.Textbox(
                                label="Consolidated History",
                                lines=20,
                                value=cfg.session_history,
                                interactive=False,
                            )

                        # -- Visualizer panel --
                        with gr.Column(visible=True) as visualizer_panel:
                            default_img = (
                                "./data/new_session.jpg"
                                if os.path.exists("./data/new_session.jpg")
                                else None
                            )
                            generated_image = gr.Image(
                                label="Generated Image",
                                type="filepath",
                                interactive=False,
                                height=490,
                                value=default_img,
                            )

                # ── Action buttons ───────────────────────────────────────
                with gr.Row():
                    send_btn = gr.Button("Send Message", variant="primary")
                    reset_btn = gr.Button("Restart Session")

                # ── Status bar + Exit ────────────────────────────────────
                with gr.Row():
                    conv_status = gr.Textbox(
                        value="Ready.",
                        label="Status",
                        interactive=False,
                        max_lines=1,
                        scale=20,
                    )
                    exit_conv = gr.Button(
                        "Exit Program",
                        variant="stop",
                        scale=1,
                    )

                # ── Left panel switching ─────────────────────────────────
                left_mode.change(
                    fn=switch_left_panel,
                    inputs=left_mode,
                    outputs=[interaction_panel, personalize_panel],
                )

                # ── Right panel switching ────────────────────────────────
                right_mode.change(
                    fn=switch_right_panel_and_state,
                    inputs=right_mode,
                    outputs=[happenings_panel, visualizer_panel, right_panel_state],
                )

                # ── Send message ─────────────────────────────────────────
                send_btn.click(
                    fn=chat_with_model,
                    inputs=[user_input, right_panel_state],
                    outputs=[
                        bot_response,
                        session_display,
                        generated_image,
                        conv_status,
                    ],
                )

                # ── Reset session ────────────────────────────────────────
                reset_btn.click(
                    fn=reset_session,
                    inputs=[],
                    outputs=[
                        session_display,
                        bot_response,
                        generated_image,
                        conv_status,
                    ],
                )

                # ── Save RP settings ─────────────────────────────────────
                rp_save_btn.click(
                    fn=save_roleplay_settings,
                    inputs=[
                        rp_agent_name,
                        rp_agent_role,
                        rp_human_name,
                        rp_scene_location,
                        rp_session_history,
                    ],
                    outputs=conv_status,
                )

                # ── Restore RP settings ──────────────────────────────────
                rp_restore_btn.click(
                    fn=restore_roleplay_settings,
                    inputs=[],
                    outputs=[
                        rp_agent_name,
                        rp_agent_role,
                        rp_human_name,
                        rp_scene_location,
                        rp_session_history,
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
                gr.Markdown("### Model Folders")
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

                gr.Markdown(
                    "*Text model (Qwen3-4b) also serves as the image encoder. "
                    "ae.safetensors is auto-copied to the image folder when set.*",
                )

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
                            info="Use 0.0 for Turbo models. Higher = stronger prompt adherence.",
                        )
                    negative_prompt_in = gr.Textbox(
                        label="Negative Prompt (Z-Image-Turbo mostly ignores this)",
                        value=cfg.default_negative_prompt,
                        lines=2,
                        placeholder="Leave empty for best results with Z-Image-Turbo",
                    )

                # ── Save Settings ─────────────────────────────────────
                with gr.Row():
                    save_btn = gr.Button(
                        "Save Configuration", variant="primary"
                    )

                # ── Status bar + Exit ────────────────────────────────────
                with gr.Row():
                    cfg_status = gr.Textbox(
                        value="Configuration tab loaded.",
                        label="Status",
                        interactive=False,
                        max_lines=1,
                        scale=20,
                    )
                    exit_cfg = gr.Button(
                        "Exit Program",
                        variant="stop",
                        scale=1,
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
    _app, local_url, _share_url = interface.launch(
        inbrowser=False,
        prevent_thread_lock=True,
        server_name="127.0.0.1",
    )
    print(f"Gradio server started at {local_url}")
    return local_url