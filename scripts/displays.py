# scripts/displays.py
# Gradio UI: Conversation tab and Configuration tab.

import os
import signal
import gradio as gr

from scripts import configure as cfg
from scripts.utilities import reset_session_state, browse_folder
from scripts.inference import prompt_response, generate_image


# -----------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------
def shutdown() -> None:
    print("Shutting down application...")
    os.kill(os.getpid(), signal.SIGTERM)


def reset_session():
    """Reload config defaults and clear session state."""
    cfg.load_config()
    reset_session_state()
    placeholder = "./data/new_session.jpg"
    if not os.path.exists(placeholder):
        placeholder = None
    return cfg.session_history, "", placeholder


def filter_model_output(raw: str) -> str:
    """Prefix the agent name if the model didn't already include it."""
    if ":" in raw[:40]:
        _, _, after = raw.partition(":")
        return f"{cfg.agent_name}: {after.strip()}"
    return f"{cfg.agent_name}: {raw}"


def chat_with_model(user_input: str):
    """Main conversation loop: converse → consolidate → image."""
    if not user_input.strip():
        return "Please type a message.", cfg.session_history, None

    if not cfg.text_model_loaded:
        return (
            "Text model is not loaded. Check the Configuration tab.",
            cfg.session_history,
            None,
        )

    cfg.human_input = f"{cfg.human_name}: {user_input.strip()}"

    # --- Step 1: converse --------------------------------------------------------
    result = prompt_response("converse")
    if "error" in result:
        return f"Error: {result['error']}", cfg.session_history, None

    filtered = filter_model_output(result["agent_response"])
    cfg.agent_output = filtered

    # --- Step 2: consolidate -----------------------------------------------------
    consolidate = prompt_response("consolidate")
    if "error" not in consolidate:
        cfg.session_history = consolidate["agent_response"]

    # --- Step 3: image -----------------------------------------------------------
    image_path = generate_image()
    cfg.latest_image_path = image_path

    return cfg.agent_output, cfg.session_history, image_path


# -----------------------------------------------------------------------
# Configuration callbacks
# -----------------------------------------------------------------------
def save_settings(
    new_agent_name,
    new_agent_role,
    new_human_name,
    new_scene_location,
    new_session_history,
    new_threads_percent,
    new_image_size,
    new_steps,
    new_sample_method,
    new_vram,
    new_text_folder,
    new_image_folder,
):
    cfg.agent_name = new_agent_name
    cfg.agent_role = new_agent_role
    cfg.human_name = new_human_name
    cfg.scene_location = new_scene_location
    cfg.session_history = new_session_history
    cfg.threads_percent = int(new_threads_percent)
    cfg.IMAGE_SIZE_OPTIONS["selected_size"] = new_image_size
    cfg.selected_steps = int(new_steps)
    cfg.selected_sample_method = new_sample_method
    cfg.vram_assigned = int(new_vram)
    cfg.text_model_folder = new_text_folder
    cfg.image_model_folder = new_image_folder
    cfg.save_config()
    return "Settings saved."


def browse_text_folder(current: str) -> str:
    return browse_folder(current)


def browse_image_folder(current: str) -> str:
    return browse_folder(current)


# -----------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------
def launch_gradio_interface() -> None:
    """Build and launch the Gradio Blocks interface."""

    # Optional hardware info
    def get_hardware_details() -> str:
        path = "./data/hardware_details.txt"
        if os.path.exists(path):
            with open(path, "r") as f:
                return f.read()
        return "No hardware details file found."

    with gr.Blocks(title="Rpg-Gradio-Gguf") as interface:
        gr.Markdown("# ⚔️ Rpg-Gradio-Gguf")

        with gr.Tabs():
            # ==============================================================
            # TAB 1 — Conversation
            # ==============================================================
            with gr.Tab("Conversation"):
                with gr.Row():
                    # Left column: chat I/O
                    with gr.Column(scale=1):
                        bot_response = gr.Textbox(
                            label="Agent Output",
                            lines=9,
                            interactive=False,
                        )
                        user_input = gr.Textbox(
                            label="Your Message",
                            lines=9,
                            placeholder="Type your message here...",
                            interactive=True,
                        )
                    # Middle column: running history
                    with gr.Column(scale=1):
                        session_display = gr.Textbox(
                            label="Consolidated History",
                            lines=21,
                            value=cfg.session_history,
                            interactive=False,
                        )
                    # Right column: generated image
                    with gr.Column(scale=1):
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

                with gr.Row():
                    send_btn = gr.Button("Send Message", variant="primary")
                    reset_btn = gr.Button("Restart Session")
                    exit_btn = gr.Button("Exit Program", variant="stop")

                send_btn.click(
                    fn=chat_with_model,
                    inputs=user_input,
                    outputs=[bot_response, session_display, generated_image],
                )
                reset_btn.click(
                    fn=reset_session,
                    inputs=[],
                    outputs=[session_display, bot_response, generated_image],
                )
                exit_btn.click(fn=shutdown, inputs=[], outputs=[])

            # ==============================================================
            # TAB 2 — Configuration
            # ==============================================================
            with gr.Tab("Configuration"):
                with gr.Row():
                    # Left: roleplay params
                    with gr.Column():
                        gr.Markdown("### Roleplay Parameters")
                        agent_name_in = gr.Textbox(
                            label="Agent Name", value=cfg.agent_name
                        )
                        agent_role_in = gr.Textbox(
                            label="Agent Role", value=cfg.agent_role
                        )
                        human_name_in = gr.Textbox(
                            label="Human Name", value=cfg.human_name
                        )
                        scene_location_in = gr.Textbox(
                            label="Scene Location", value=cfg.scene_location
                        )
                        session_history_in = gr.Textbox(
                            label="Default History",
                            value=cfg.session_history,
                            lines=4,
                        )

                    # Right: generation params
                    with gr.Column():
                        gr.Markdown("### Generation Parameters")
                        hardware_display = gr.Textbox(
                            label="Detected Hardware",
                            value=get_hardware_details(),
                            lines=3,
                            interactive=False,
                        )
                        threads_slider = gr.Slider(
                            label="CPU Thread Usage (%)",
                            minimum=10,
                            maximum=100,
                            step=10,
                            value=cfg.threads_percent,
                        )
                        vram_dropdown = gr.Dropdown(
                            label="VRAM Assigned (MB)",
                            choices=[str(v) for v in cfg.VRAM_OPTIONS],
                            value=str(cfg.vram_assigned),
                            type="value",
                        )
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
                        )
                        sample_dd = gr.Dropdown(
                            label="Sample Method",
                            choices=cfg.SAMPLE_METHOD_OPTIONS,
                            value=cfg.selected_sample_method,
                            type="value",
                        )

                gr.Markdown("### Model Folders")
                with gr.Row():
                    text_folder_in = gr.Textbox(
                        label="Text Model Folder",
                        value=cfg.text_model_folder,
                        scale=4,
                    )
                    text_browse_btn = gr.Button("Browse...", scale=1)
                with gr.Row():
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

                with gr.Row():
                    save_btn = gr.Button(
                        "Save Configuration", variant="primary"
                    )
                    save_status = gr.Textbox(
                        label="Status", interactive=False, scale=2
                    )

                save_btn.click(
                    fn=save_settings,
                    inputs=[
                        agent_name_in,
                        agent_role_in,
                        human_name_in,
                        scene_location_in,
                        session_history_in,
                        threads_slider,
                        image_size_dd,
                        steps_dd,
                        sample_dd,
                        vram_dropdown,
                        text_folder_in,
                        image_folder_in,
                    ],
                    outputs=save_status,
                )

    interface.launch(inbrowser=True)
