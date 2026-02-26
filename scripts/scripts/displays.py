# Script: scripts/displays.py
# Gradio UI: Conversation tab and Configuration tab.
# UPDATED: Visual workflow progress in status bar with ✓ ★ x indicators.
# - Workflow progress shows all stages with completion indicators.
# - ✓ = completed, ★ = current, x = not yet done.
# - Your Message always visible.
# - Send/Cancel row: Send always visible (scale=3), Cancel always visible (scale=1).
# - NO dynamic visibility or interactive state changes for buttons.

# Imports...
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
# Workflow Progress Helper (status bar string)
# -----------------------------------------------------------------------
def build_workflow_status(current_stage: int, completed_stages: list[int]) -> str:
    """
    Return a visual workflow progress string showing all stages.
    ✓ = completed, ★ = current, x = not yet done.
    Example: "Initial Preparation ✓  Character Responses ★  Instance Summary x  ..."
    """
    stages = cfg.WORKFLOW_STAGES
    if current_stage < 0 or current_stage >= len(stages):
        return "Ready."
    status_parts = []
    for idx, stage_name in enumerate(stages):
        if idx in completed_stages:
            status_parts.append(f"{stage_name} ✓")
        elif idx == current_stage:
            status_parts.append(f"{stage_name} ★")
        else:
            status_parts.append(f"{stage_name} x")

    return "  ".join(status_parts)
# -----------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------
def shutdown() -> None:
    """Request application shutdown."""
    if cfg.shutdown_fn is not None:
        cfg.shutdown_fn()
    else:
        print("WARNING: shutdown_fn not registered. Force exiting.")
        os._exit(1)

def reset_session():
    """Reload config defaults and clear session state."""
    cfg.load_config()
    reset_session_state()
    cfg.consolidated_instance = ""
    cfg.workflow_stage_index = -1
    cfg.workflow_completed_stages = []
    placeholder = "./data/new_session.jpg"
    if not os.path.exists(placeholder):
        placeholder = None
    # Return: instance_display, session_display, bot_response, generated_image,
    #         conv_status, user_input (cleared, enabled), sequence_gallery
    # Note: send_btn and cancel_btn are NOT updated - they stay static
    return (
        "",
        cfg.session_history,
        "",
        placeholder,
        "Session restarted.",
        gr.update(value="", interactive=True),  # user_input enabled
        [],
    )

def cancel_response():
    """
    Signal the current inference pipeline to abort.
    """
    cfg.cancel_processing = True
    return "Cancelling…"

def filter_model_output(raw: str, agent_name: str | None = None) -> str:
    """Prefix the responding agent name if the model didn't already include it."""
    name = (agent_name or cfg.agent1_name or "Agent").strip()
    if ": " in raw[:40]:
        _, _, after = raw.partition(": ")
        return f"{name}: {after.strip()}"
    return f"{name}: {raw}"

def chat_with_model(user_input: str, right_mode: str):
    """
    Main conversation loop: converse → consolidate → [image_prompt → image].
    Workflow progress is shown in the status bar (conv_status) with visual indicators.
    UI elements: user_input always visible, send/cancel buttons always visible.
    Buttons do NOT change state - they are always clickable (handled by cancel flag).
    Yields 7 values every iteration:
    (scenario_log, instance_display, session_history, image, status,
    user_input_update, sequence_gallery)
    """
    # Reset cancel flag and workflow tracking
    cfg.cancel_processing = False
    cfg.workflow_stage_index = -1
    cfg.workflow_completed_stages = []

    # ── Empty input guard ────────────────────────────────────────
    if not user_input.strip():
        yield (
            cfg.scenario_log or "   ",
            cfg.consolidated_instance or "   ",
            cfg.session_history or "   ",
            cfg.latest_image_path,
            "Please type a message first.",
            gr.update(value=user_input, interactive=True),  # keep input
            list(reversed(cfg.session_image_paths)),
        )
        return

    # ── Pre-flight: verify model files exist ────────────────────────
    need_image = (right_mode == "Z-Image-Turbo")

    if not scan_for_gguf(cfg.text_model_folder):
        yield (
            cfg.scenario_log or "   ",
            cfg.consolidated_instance or "   ",
            cfg.session_history or "   ",
            cfg.latest_image_path,
            "No text model found. Check Configuration → Text Model Folder.",
            gr.update(value=user_input, interactive=True),
            list(reversed(cfg.session_image_paths)),
        )
        return

    if need_image and not scan_for_gguf(cfg.image_model_folder):
        yield (
            cfg.scenario_log or "   ",
            cfg.consolidated_instance or "   ",
            cfg.session_history or "   ",
            cfg.latest_image_path,
            "No image model found. Check Configuration → Image Model Folder, "
            "or switch Visualizer to 'No Generation'.",
            gr.update(value=user_input, interactive=True),
            list(reversed(cfg.session_image_paths)),
        )
        return

    # Clear idle timer – model's turn starts now
    cfg.user_turn_start_time = None

    # ── Lazy-load models if needed ───────────────────────────────────
    models_need_check = (
        not cfg.text_model_loaded
        or (need_image and not cfg.image_model_loaded)
    )
    if models_need_check:
        yield (
            cfg.scenario_log or "   ",
            cfg.consolidated_instance or "   ",
            cfg.session_history or "   ",
            cfg.latest_image_path,
            "Loading models… please wait.",
            gr.update(value="", interactive=False),  # user_input disabled
            list(reversed(cfg.session_image_paths)),
        )
        ok, load_msg = ensure_models_loaded(need_image=need_image)
        if not ok:
            cfg.user_turn_start_time = time.time()
            yield (
                cfg.scenario_log or "   ",
                cfg.consolidated_instance or "   ",
                cfg.session_history or "   ",
                cfg.latest_image_path,
                load_msg,
                gr.update(value="", interactive=True),  # user_input enabled
                list(reversed(cfg.session_image_paths)),
            )
            return

    # Mark Initial Preparation (stage 0) as complete after models loaded
    cfg.workflow_stage_index = 1  # Now on Character Responses
    cfg.workflow_completed_stages = [0]  # Initial Preparation done

    cfg.human_input = user_input.strip()

    # Append user line to scenario log
    cfg.scenario_log = (cfg.scenario_log + f"\n{cfg.human_name}: {cfg.human_input}").lstrip()

    # Show user input immediately, then clear and disable for processing
    yield (
        cfg.scenario_log,
        cfg.consolidated_instance or "   ",
        cfg.session_history,
        cfg.latest_image_path,
        "Rotation Progress: " + build_workflow_status(cfg.workflow_stage_index, cfg.workflow_completed_stages),
        gr.update(value="", interactive=False),  # clear and disable
        list(reversed(cfg.session_image_paths)),
    )

    # --- Step 1: converse (per active agent) -------------------------
    active_agents = get_active_agents()
    exchange_lines: list[str] = []

    for idx, (agent_name, agent_role) in enumerate(active_agents):
        if cfg.cancel_processing:
            cfg.user_turn_start_time = time.time()
            yield (
                cfg.scenario_log,
                cfg.consolidated_instance or "   ",
                cfg.session_history,
                cfg.latest_image_path,
                "Response cancelled.",
                gr.update(value="", interactive=True),  # user_input enabled
                list(reversed(cfg.session_image_paths)),
            )
            return

        agent_count = len(active_agents)
        status_msg = f"Generating response… ({agent_name})" if agent_count > 1 else "Generating response…"
        # Update status with stage progress (stage 1 = Character Responses)
        stage_status = build_workflow_status(1, cfg.workflow_completed_stages)
        yield (
            cfg.scenario_log,
            cfg.consolidated_instance or "   ",
            cfg.session_history,
            cfg.latest_image_path,
            "Rotation Progress: " + stage_status,
            gr.update(value="", interactive=False),  # user_input disabled
            list(reversed(cfg.session_image_paths)),
        )

        result = prompt_response("converse", responding_agent=(agent_name, agent_role))

        if cfg.cancel_processing or result.get("cancelled"):
            cfg.user_turn_start_time = time.time()
            yield (
                cfg.scenario_log,
                cfg.consolidated_instance or "   ",
                cfg.session_history,
                cfg.latest_image_path,
                "Response cancelled.",
                gr.update(value="", interactive=True),  # user_input enabled
                list(reversed(cfg.session_image_paths)),
            )
            return

        if "error" in result:
            cfg.user_turn_start_time = time.time()
            yield (
                cfg.scenario_log,
                cfg.consolidated_instance or "   ",
                cfg.session_history,
                cfg.latest_image_path,
                f"Converse error ({agent_name}): {result['error']}",
                gr.update(value="", interactive=True),  # user_input enabled
                list(reversed(cfg.session_image_paths)),
            )
            return

        formatted_line = filter_model_output(result["agent_response"], agent_name=agent_name)
        exchange_lines.append(formatted_line)
        cfg.scenario_log = cfg.scenario_log + f"\n{formatted_line}"

        yield (
            cfg.scenario_log,
            cfg.consolidated_instance or "   ",
            cfg.session_history,
            cfg.latest_image_path,
            "Rotation Progress: " + stage_status,
            gr.update(value="", interactive=False),  # user_input disabled
            list(reversed(cfg.session_image_paths)),
        )

    cfg.agent_output = exchange_lines[-1] if exchange_lines else ""
    cfg.agent_exchange = "\n".join(exchange_lines)

    # Mark Character Responses (stage 1) as complete
    cfg.workflow_completed_stages.append(1)

    # --- Step 2a: instance summary ------------------------------------
    if cfg.cancel_processing:
        cfg.user_turn_start_time = time.time()
        yield (
            cfg.scenario_log,
            cfg.consolidated_instance or "   ",
            cfg.session_history,
            cfg.latest_image_path,
            "Response cancelled.",
            gr.update(value="", interactive=True),  # user_input enabled
            list(reversed(cfg.session_image_paths)),
        )
        return

    cfg.workflow_stage_index = 2  # Instance Summary
    stage_status = build_workflow_status(2, cfg.workflow_completed_stages)
    yield (
        cfg.scenario_log,
        cfg.consolidated_instance or "   ",
        cfg.session_history,
        cfg.latest_image_path,
        "Rotation Progress: " + stage_status,
        gr.update(value="", interactive=False),  # user_input disabled
        list(reversed(cfg.session_image_paths)),
    )

    instance_result = prompt_response("instance")
    if cfg.cancel_processing or instance_result.get("cancelled"):
        cfg.user_turn_start_time = time.time()
        yield (
            cfg.scenario_log,
            cfg.consolidated_instance or "   ",
            cfg.session_history,
            cfg.latest_image_path,
            "Response cancelled.",
            gr.update(value="", interactive=True),  # user_input enabled
            list(reversed(cfg.session_image_paths)),
        )
        return

    if "agent_response" in instance_result:
        cfg.consolidated_instance = instance_result["agent_response"]

    # Mark Instance Summary (stage 2) as complete
    cfg.workflow_completed_stages.append(2)

    # --- Step 2b: consolidate -----------------------------------------
    cfg.workflow_stage_index = 3  # History Consolidation
    stage_status = build_workflow_status(3, cfg.workflow_completed_stages)
    yield (
        cfg.scenario_log,
        cfg.consolidated_instance or "   ",
        cfg.session_history,
        cfg.latest_image_path,
        "Rotation Progress: " + stage_status,
        gr.update(value="", interactive=False),  # user_input disabled
        list(reversed(cfg.session_image_paths)),
    )

    consolidate = prompt_response("consolidate")
    if not cfg.cancel_processing and not consolidate.get("cancelled") and "error" not in consolidate:
        cfg.session_history = consolidate["agent_response"]

    # Mark History Consolidation (stage 3) as complete
    cfg.workflow_completed_stages.append(3)

    # ── Image generation only when Visualizer is Z-Image-Turbo ────────
    if cfg.cancel_processing or right_mode != "Z-Image-Turbo":
        cfg.user_turn_start_time = time.time()
        status = "Response cancelled." if cfg.cancel_processing else "Response generated."
        # Show completion for a moment before enabling input
        yield (
            cfg.scenario_log,
            cfg.consolidated_instance or "   ",
            cfg.session_history,
            cfg.latest_image_path,
            status + " Rotation complete, produce new input..",
            gr.update(value="", interactive=False),  # user_input disabled
            list(reversed(cfg.session_image_paths)),
        )
        time.sleep(1.0)
        yield (
            cfg.scenario_log,
            cfg.consolidated_instance or "   ",
            cfg.session_history,
            cfg.latest_image_path,
            "Ready.",
            gr.update(value="", interactive=True),  # user_input enabled
            list(reversed(cfg.session_image_paths)),
        )
        return

    # --- Step 3: generate visual prompt -------------------------------
    cfg.workflow_stage_index = 4  # Encoding Prompt
    stage_status = build_workflow_status(4, cfg.workflow_completed_stages)
    yield (
        cfg.scenario_log,
        cfg.consolidated_instance or "   ",
        cfg.session_history,
        cfg.latest_image_path,
        "Rotation Progress: " + stage_status,
        gr.update(value="", interactive=False),  # user_input disabled
        list(reversed(cfg.session_image_paths)),
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
            cfg.consolidated_instance or "   ",
            cfg.session_history,
            cfg.latest_image_path,
            "Response cancelled.",
            gr.update(value="", interactive=True),  # user_input enabled
            list(reversed(cfg.session_image_paths)),
        )
        return

    if not visual_prompt:
        visual_prompt = f"A scene at {cfg.scene_location}."

    # Mark Encoding Prompt (stage 4) as complete
    cfg.workflow_completed_stages.append(4)

    # --- Step 4: generate image (threaded with progress) -------------
    cfg.workflow_stage_index = 5  # Image Generation
    stage_status = build_workflow_status(5, cfg.workflow_completed_stages)
    yield (
        cfg.scenario_log,
        cfg.consolidated_instance or "   ",
        cfg.session_history,
        cfg.latest_image_path,
        "Rotation Progress: " + stage_status,
        gr.update(value="", interactive=False),  # user_input disabled
        list(reversed(cfg.session_image_paths)),
    )

    _img_result: dict = {"path": None, "done": False}

    def _worker():
        _img_result["path"] = generate_image(scene_prompt=visual_prompt)
        _img_result["done"] = True

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    while not _img_result["done"]:
        if cfg.cancel_processing:
            thread.join()
            cfg.user_turn_start_time = time.time()
            yield (
                cfg.scenario_log,
                cfg.consolidated_instance or "   ",
                cfg.session_history,
                cfg.latest_image_path,
                "Response cancelled.",
                gr.update(value="", interactive=True),  # user_input enabled
                list(reversed(cfg.session_image_paths)),
            )
            return
        step, total = get_image_gen_progress()
        if total > 0 and step > 0:
            status = f"Generating image, step {step}/{total}…"
        else:
            status = "Generating image…"
        yield (
            cfg.scenario_log,
            cfg.consolidated_instance or "   ",
            cfg.session_history,
            cfg.latest_image_path,
            "Rotation Progress: " + stage_status,
            gr.update(value="", interactive=False),  # user_input disabled
            list(reversed(cfg.session_image_paths)),
        )
        time.sleep(1.0)

    thread.join()
    image_path = _img_result["path"]
    cfg.latest_image_path = image_path

    # Mark Image Generation (stage 5) as complete
    cfg.workflow_completed_stages.append(5)

    # All stages complete – return to user
    cfg.user_turn_start_time = time.time()
    yield (
        cfg.scenario_log,
        cfg.consolidated_instance or "   ",
        cfg.session_history,
        image_path,
        "Rotation complete, produce new input..",
        gr.update(value="", interactive=False),  # user_input disabled
        list(reversed(cfg.session_image_paths)),
    )
    time.sleep(1.0)
    yield (
        cfg.scenario_log,
        cfg.consolidated_instance or "   ",
        cfg.session_history,
        image_path,
        "Ready.",
        gr.update(value="", interactive=True),  # user_input enabled
        list(reversed(cfg.session_image_paths)),
    )
    return
# -----------------------------------------------------------------------
# Panel switching callback - SIMPLIFIED
# -----------------------------------------------------------------------
def switch_left_panel(mode: str):
    """Toggle visibility of Interactions / Happenings / Personalize panels.
    Buttons are NOT updated - they remain static.
    """
    return (
        gr.update(visible=(mode == "Interactions")),   # interaction_panel
        gr.update(visible=(mode == "Happenings")),     # happenings_panel
        gr.update(visible=(mode == "Personalize")),    # personalize_panel
        # NO button updates - they stay exactly as defined in layout
    )

def switch_right_panel_and_state(mode: str):
    """Update the Visualizer mode state."""
    return mode

def on_gallery_select(evt: gr.SelectData):
    """When the user clicks a thumbnail, display it in the main image."""
    if evt.value and isinstance(evt.value, dict):
        path = evt.value.get("image", {}).get("path") or evt.value.get("path")
        if path:
            return path
    elif evt.value and isinstance(evt.value, str):
        return evt.value
    return gr.update()
# -----------------------------------------------------------------------
# Roleplay settings callbacks (unchanged)
# -----------------------------------------------------------------------
def save_roleplay_settings(
    a1_name, a1_role, a2_name, a2_role, a3_name, a3_role,
    human, human_age, human_gender, location, event_time, history,
):
    cfg.agent1_name = a1_name
    cfg.agent1_role = a1_role
    cfg.agent2_name = a2_name
    cfg.agent2_role = a2_role
    cfg.agent3_name = a3_name
    cfg.agent3_role = a3_role
    cfg.human_name = human
    cfg.human_age = str(human_age) if human_age is not None else ""
    cfg.human_gender = human_gender
    cfg.scene_location = location
    cfg.event_time = event_time
    cfg.default_history = history
    cfg.save_config()
    return "RP settings saved."

def restore_roleplay_settings():
    d = cfg.DEFAULT_RP_SETTINGS
    cfg.agent1_name = d["agent1_name"]
    cfg.agent1_role = d["agent1_role"]
    cfg.agent2_name = d["agent2_name"]
    cfg.agent2_role = d["agent2_role"]
    cfg.agent3_name = d["agent3_name"]
    cfg.agent3_role = d["agent3_role"]
    cfg.human_name = d["human_name"]
    cfg.human_age = d["human_age"]
    cfg.human_gender = d["human_gender"]
    cfg.scene_location = d["scene_location"]
    cfg.event_time = d["event_time"]
    cfg.default_history = d["default_history"]
    cfg.save_config()
    return (
        cfg.agent1_name, cfg.agent1_role,
        cfg.agent2_name, cfg.agent2_role,
        cfg.agent3_name, cfg.agent3_role,
        cfg.human_name, cfg.human_age, cfg.human_gender, 
        cfg.scene_location, cfg.event_time,
        cfg.default_history,
        "RP defaults restored and saved.",
    )
# -----------------------------------------------------------------------
# Configuration callbacks (unchanged except browse_folder helpers)
# -----------------------------------------------------------------------
def save_config_tab_settings(
    gpu_choice, vram, cpu_choice, threads, auto_unload, max_mem,
    img_sz, steps, samp, cfgs, neg, tf, imf,
):
    try:
        cfg.selected_gpu = int(gpu_choice.split(": ")[0].replace("GPU", ""))
    except (ValueError, AttributeError):
        cfg.selected_gpu = 0
    cfg.vram_assigned = int(vram)
    try:
        cfg.selected_cpu = int(cpu_choice.split(": ")[0].replace("CPU", ""))
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

    old_image_folder = cfg.image_model_folder
    cfg.image_model_folder = imf

    vae_status = ""
    if imf and os.path.isdir(imf):
        vae_result = ensure_vae_in_image_folder(imf)
        if vae_result:
            vae_status = "ae.safetensors confirmed."
        else:
            vae_status = "WARNING: ae.safetensors missing — re-run installer."

    cfg.save_config()
    return f"Configuration saved.{vae_status}"

def browse_text_folder(current: str) -> str:
    return browse_folder(current)

def browse_image_folder(current: str) -> str:
    chosen = browse_folder(current)
    if chosen and os.path.isdir(chosen):
        ensure_vae_in_image_folder(chosen)
    return chosen
# -----------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------
def launch_gradio_interface() -> str | None:
    """Build and launch the Gradio Blocks interface."""
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

    _SPELLCHECK_JS = """
(function() {
    function enableSpellcheck() {
        document.querySelectorAll('textarea, input[type="text"]').forEach(el => {
            el.setAttribute('spellcheck', 'true');
            el.setAttribute('lang', 'en');
        });
    }
    enableSpellcheck();
    const observer = new MutationObserver(enableSpellcheck);
    observer.observe(document.body, { childList: true, subtree: true });
})();
"""

    _ZOOM_HEAD_JS = """
(function() {
    function enableZoom() {
        let zoomLevel = 1.0;
        const minZoom = 0.5;
        const maxZoom = 2.0;
        const step = 0.1;

        window.addEventListener('wheel', function(e) {
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                e.stopPropagation();
                if (e.deltaY < 0) {
                    zoomLevel = Math.min(maxZoom, zoomLevel + step);
                } else {
                    zoomLevel = Math.max(minZoom, zoomLevel - step);
                }
                document.body.style.transform = 'scale(' + zoomLevel + ')';
                document.body.style.transformOrigin = 'top left';
                document.body.style.width = (100 / zoomLevel) + '%';
                console.log('Zoom level:', zoomLevel);
            }
        }, { capture: true, passive: false });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', enableZoom);
    } else {
        enableZoom();
    }
})();
"""

    _CONTEXT_MENU_JS = """
(function() {
    let currentTarget = null;

    function createContextMenu() {
        if (document.getElementById('gradio-context-menu')) return;
        
        const menu = document.createElement('div');
        menu.id = 'gradio-context-menu';
        menu.innerHTML = `
            <div class="menu-item" data-action="select-all">Select All</div>
            <div class="menu-item" data-action="select-none">Select None</div>
            <div class="separator"></div>
            <div class="menu-item" data-action="copy">Copy</div>
            <div class="menu-item" data-action="cut">Cut</div>
            <div class="menu-item" data-action="paste">Paste</div>
        `;
        document.body.appendChild(menu);
        
        menu.addEventListener('click', (e) => {
            const action = e.target.dataset.action;
            if (!action || !currentTarget) return;
            e.stopPropagation();
            handleMenuAction(action, currentTarget);
            hideContextMenu();
        });
        
        document.addEventListener('click', (e) => {
            if (!menu.contains(e.target)) hideContextMenu();
        });
        
        window.addEventListener('scroll', hideContextMenu, { passive: true });
        window.addEventListener('resize', hideContextMenu);
    }
    
    function showContextMenu(x, y, target) {
        createContextMenu();
        currentTarget = target;
        
        const menu = document.getElementById('gradio-context-menu');
        menu.style.left = x + 'px';
        menu.style.top = y + 'px';
        menu.classList.add('visible');
        
        const rect = menu.getBoundingClientRect();
        if (rect.right > window.innerWidth) {
            menu.style.left = (window.innerWidth - rect.width - 8) + 'px';
        }
        if (rect.bottom > window.innerHeight) {
            menu.style.top = (window.innerHeight - rect.height - 8) + 'px';
        }
    }
    
    function hideContextMenu() {
        const menu = document.getElementById('gradio-context-menu');
        if (menu) menu.classList.remove('visible');
        currentTarget = null;
    }
    
    function handleMenuAction(action, target) {
        if (!target) return;
        target.focus();
        
        switch(action) {
            case 'select-all':
                target.select();
                break;
            case 'select-none':
                if (window.getSelection) {
                    window.getSelection().removeAllRanges();
                } else if (document.selection) {
                    document.selection.empty();
                }
                break;
            case 'copy':
                target.select();
                try { document.execCommand('copy'); } catch (err) { console.warn('Copy failed:', err); }
                break;
            case 'cut':
                target.select();
                try { document.execCommand('cut'); } catch (err) { console.warn('Cut failed:', err); }
                break;
            case 'paste':
                if (navigator.clipboard && navigator.clipboard.readText) {
                    navigator.clipboard.readText().then(text => {
                        insertAtCursor(target, text);
                    }).catch(() => {
                        try { document.execCommand('paste'); } catch (e) { console.warn('Paste failed:', e); }
                    });
                } else {
                    try { document.execCommand('paste'); } catch (e) { console.warn('Paste failed:', e); }
                }
                break;
        }
    }
    
    function insertAtCursor(element, text) {
        const start = element.selectionStart;
        const end = element.selectionEnd;
        const value = element.value;
        element.value = value.substring(0, start) + text + value.substring(end);
        element.selectionStart = element.selectionEnd = start + text.length;
        element.dispatchEvent(new Event('input', { bubbles: true }));
    }
    
    function attachListeners() {
        document.querySelectorAll('textarea, input[type="text"]').forEach(el => {
            el.addEventListener('contextmenu', (e) => {
                e.preventDefault();
                showContextMenu(e.clientX, e.clientY, e.target);
            });
            el.addEventListener('keydown', (e) => {
                if (e.key === 'ContextMenu' || (e.shiftKey && e.key === 'F10')) {
                    e.preventDefault();
                    const rect = el.getBoundingClientRect();
                    showContextMenu(rect.right - 5, rect.top + 5, el);
                }
            });
        });
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            attachListeners();
            const observer = new MutationObserver(attachListeners);
            observer.observe(document.body, { childList: true, subtree: true });
        });
    } else {
        attachListeners();
        const observer = new MutationObserver(attachListeners);
        observer.observe(document.body, { childList: true, subtree: true });
    }
})();
"""

    _CUSTOM_CSS = """
#generated_image img {
    object-fit: contain !important;
    width: 100% !important;
    height: 100% !important;
}
#sequence_gallery .grid-container,
#sequence_gallery .grid-wrap {
    gap: 4px !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    flex-wrap: nowrap !important;
}
#sequence_gallery .thumbnail-item,
#sequence_gallery .thumbnail-lg {
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
#conv_status textarea {
    white-space: nowrap !important;
    overflow-x: auto !important;
    font-size: 12px !important;
}
#gradio-context-menu {
    position: fixed !important;
    z-index: 9999 !important;
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 6px !important;
    padding: 4px 0 !important;
    min-width: 140px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important;
    font-family: system-ui, sans-serif !important;
    font-size: 13px !important;
    display: none !important;
}
#gradio-context-menu.visible {
    display: block !important;
}
#gradio-context-menu .menu-item {
    padding: 6px 12px !important;
    cursor: pointer !important;
    color: #e5e7eb !important;
    user-select: none !important;
}
#gradio-context-menu .menu-item:hover {
    background: #374151 !important;
}
#gradio-context-menu .menu-item:active {
    background: #4b5563 !important;
}
#gradio-context-menu .separator {
    height: 1px !important;
    background: #374151 !important;
    margin: 4px 0 !important;
}
"""

    _combined_head_js = "\n".join([_SPELLCHECK_JS, _ZOOM_HEAD_JS, _CONTEXT_MENU_JS])

    with gr.Blocks(title="Rpg-Gradio-Gguf") as interface:
        gr.Markdown("# ⚔️ Rpg-Gradio-Gguf")

        with gr.Tabs():
            with gr.Tab("Conversation"):
                right_panel_state = gr.State("Z-Image-Turbo")

                with gr.Row():
                    with gr.Column(scale=1):
                        left_mode = gr.Radio(
                            choices=["Interactions", "Happenings", "Personalize"],
                            value="Interactions",
                            label="Engagement",
                            interactive=True,
                        )

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
                            with gr.Row():
                                send_btn = gr.Button(
                                    "Send Message",
                                    variant="primary",
                                    scale=3,
                                    interactive=True,
                                )
                                cancel_btn = gr.Button(
                                    "Cancel Response",
                                    variant="stop",
                                    scale=1,
                                    interactive=True,
                                )

                        with gr.Column(visible=False) as happenings_panel:
                            instance_display = gr.Textbox(
                                label="Instance Summary",
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

                        with gr.Column(visible=False) as personalize_panel:
                            with gr.Row():
                                rp_agent1_name = gr.Textbox(label="Agent 1 Name", value=cfg.agent1_name)
                                rp_agent2_name = gr.Textbox(label="Agent 2 Name", value=cfg.agent2_name)
                                rp_agent3_name = gr.Textbox(label="Agent 3 Name", value=cfg.agent3_name)
                            with gr.Row():
                                rp_agent1_role = gr.Textbox(label="Agent 1 Role", value=cfg.agent1_role)
                                rp_agent2_role = gr.Textbox(label="Agent 2 Role", value=cfg.agent2_role)
                                rp_agent3_role = gr.Textbox(label="Agent 3 Role", value=cfg.agent3_role)
                            with gr.Row():
                                rp_human_name = gr.Textbox(label="Human Name", value=cfg.human_name)
                                rp_human_age = gr.Number(
                                    label="Human Age",
                                    value=int(cfg.human_age) if cfg.human_age and cfg.human_age.strip().isdigit() else None,
                                    precision=0, minimum=0, maximum=999,
                                )
                                rp_human_gender = gr.Dropdown(
                                    label="Human Gender",
                                    choices=cfg.GENDER_OPTIONS,
                                    value=cfg.human_gender,
                                )
                            with gr.Row():
                                rp_scene_location = gr.Textbox(label="Scene Location", value=cfg.scene_location)
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
                                rp_save_btn = gr.Button("Save RP Settings", variant="primary")
                                rp_restore_btn = gr.Button("Restore RP Defaults", variant="stop")

                    with gr.Column(scale=1):
                        right_mode = gr.Radio(
                            choices=["No Generation", "Z-Image-Turbo"],
                            value="Z-Image-Turbo",
                            label="Visualizer",
                            interactive=True,
                        )
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

                with gr.Row():
                    conv_status = gr.Textbox(
                        value="Ready.",
                        label="Status",
                        interactive=False,
                        max_lines=2,
                        scale=20,
                        elem_id="conv_status",
                    )
                    with gr.Column(scale=3, min_width=240):
                        restart_btn = gr.Button("Restart Session", variant="primary")
                        exit_conv = gr.Button("Exit Program", variant="stop")

                left_mode.change(
                    fn=switch_left_panel,
                    inputs=left_mode,
                    outputs=[
                        interaction_panel,
                        happenings_panel,
                        personalize_panel,
                    ],
                )

                right_mode.change(
                    fn=switch_right_panel_and_state,
                    inputs=right_mode,
                    outputs=right_panel_state,
                )

                sequence_gallery.select(
                    fn=on_gallery_select,
                    inputs=None,
                    outputs=generated_image,
                )

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
                    ],
                )

                cancel_btn.click(
                    fn=cancel_response, 
                    inputs=[],
                    outputs=[conv_status],
                )

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

                rp_restore_btn.click(
                    fn=restore_roleplay_settings,
                    inputs=[],
                    outputs=[
                        rp_agent1_name, rp_agent1_role,
                        rp_agent2_name, rp_agent2_role,
                        rp_agent3_name, rp_agent3_role,
                        rp_human_name, rp_human_age, rp_human_gender,
                        rp_scene_location, rp_event_time,
                        rp_default_history,
                        conv_status,
                    ],
                )

                exit_conv.click(fn=shutdown, inputs=[], outputs=[])

            with gr.Tab("Configuration"):
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
                        valid_vram = cfg.vram_assigned if cfg.vram_assigned in cfg.VRAM_OPTIONS else 8192
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
                            minimum=1, maximum=max_threads, step=1, value=valid_threads,
                            label="Threads Assigned",
                            info=f"Number of threads for inference. System has {cfg.CPU_THREADS} total threads.",
                        )
                    with gr.Row():
                        auto_unload_check = gr.Checkbox(
                            label="Auto-unload model on high SYSTEM RAM",
                            value=cfg.auto_unload,
                        )
                        max_memory_slider = gr.Slider(
                            minimum=50, maximum=95, step=5, value=cfg.max_memory_percent,
                            label="Max System RAM % (unload threshold)",
                            info="Triggers model unload when system RAM exceeds this percentage (NOT VRAM)",
                        )

                gr.Markdown("### Model Folders")
                with gr.Row():
                    text_folder_in = gr.Textbox(label="Text Model Folder", value=cfg.text_model_folder, scale=4)
                    text_browse_btn = gr.Button("Browse...", scale=1)
                    image_folder_in = gr.Textbox(label="Image Model Folder", value=cfg.image_model_folder, scale=4)
                    image_browse_btn = gr.Button("Browse...", scale=1)

                text_browse_btn.click(fn=browse_text_folder, inputs=text_folder_in, outputs=text_folder_in)
                image_browse_btn.click(fn=browse_image_folder, inputs=image_folder_in, outputs=image_folder_in)

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

                with gr.Row():
                    cfg_status = gr.Textbox(
                        value="Configuration tab loaded.",
                        label="Status",
                        interactive=False,
                        max_lines=1,
                        scale=20,
                    )
                    with gr.Column(scale=3, min_width=240):
                        save_btn = gr.Button("Save Configuration", variant="primary")
                        exit_cfg = gr.Button("Exit Program", variant="stop")

                save_btn.click(
                    fn=save_config_tab_settings,
                    inputs=[
                        selected_gpu_dropdown, selected_vram_dropdown,
                        selected_cpu_dropdown, threads_assigned_slider,
                        auto_unload_check, max_memory_slider,
                        image_size_dd, steps_dd, sample_dd, cfg_scale_dd,
                        negative_prompt_in, text_folder_in, image_folder_in,
                    ],
                    outputs=cfg_status,
                )

                exit_cfg.click(fn=shutdown, inputs=[], outputs=[])

        interface.queue()
        _app, local_url, _share_url = interface.launch(
            inbrowser=False,
            prevent_thread_lock=True,
            server_name="127.0.0.1",
            js=_combined_head_js,
            css=_CUSTOM_CSS,
            head="",
        )
        print(f"Gradio server started at {local_url}")
        return local_url