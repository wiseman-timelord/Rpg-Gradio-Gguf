# scripts/interface.py

import os
import signal
import data.temporary as temporary
from data.temporary import (agent_name, agent_role, human_name, session_history, threads_percent, rotation_counter, agent_output, human_input)
from scripts.utility import reset_session_state, write_to_yaml, read_yaml
from scripts.model import prompt_response
import gradio as gr
import threading
import time

def shutdown():
    print("Shutting down the application...")
    os.kill(os.getpid(), signal.SIGTERM)

def reset_session():
    """
    Resets the session history and related variables to default values in data.temporary.
    Does NOT save to persistent.yaml automatically. Keeps it ephemeral.
    """
    reset_session_state()  # Resets session_history, agent_output, human_input in data.temporary
    temporary.rotation_counter = 0
    # Return updated interface fields without writing to YAML
    return "", temporary.session_history

def apply_configuration(new_agent_name, new_agent_role, new_human_name):
    temporary.agent_name = new_agent_name
    temporary.agent_role = new_agent_role
    temporary.human_name = new_human_name
    return "Configuration applied successfully (Not saved yet)."

def save_configuration(new_agent_name, new_agent_role, new_human_name, new_threads_percent, new_session_history):
    """
    Saves agent configuration and session settings to persistent.yaml.
    This is explicitly triggered by the user.
    """
    temporary.agent_name = new_agent_name
    temporary.agent_role = new_agent_role
    temporary.human_name = new_human_name
    temporary.threads_percent = int(new_threads_percent)
    temporary.session_history = new_session_history

    # Now actually save to persistent.yaml
    from main_script import save_persistent_settings
    save_persistent_settings()
    return "Configuration and session settings saved successfully!"

def chat_with_model(user_input):
    """
    Handles the chat interaction: sends user input to the model and processes responses.
    """
    if not user_input.strip():
        return "Error: Input cannot be empty.", temporary.session_history

    # Update the human_input variable
    temporary.human_input = user_input.strip()
    print(f"[DEBUG] Updated human_input: {temporary.human_input}")  # Debugging line

    temporary.rotation_counter += 1

    # First, get the response from 'converse'
    converse_result = prompt_response('converse', temporary.rotation_counter)
    if 'error' in converse_result:
        return f"Error in model response: {converse_result['error']}", temporary.session_history

    temporary.agent_output = converse_result['agent_response']
    # Update session history with both input and output
    temporary.session_history += f"\n{temporary.human_input}\n{temporary.agent_output}"
    print(f"[DEBUG] Updated session history after converse:\n{temporary.session_history}")

    # Now call the 'consolidate' prompt
    consolidate_result = prompt_response('consolidate', temporary.rotation_counter)
    if 'error' in consolidate_result:
        return f"Error in model consolidation: {consolidate_result['error']}", temporary.session_history

    # The 'consolidate' response is already appended to session_history in `prompt_response`
    return temporary.agent_output, temporary.session_history


def launch_gradio_interface():
    # Ensure data is re-synced with the latest settings, although this might be redundant
    # if already handled in the background engine or initialization flow
    session_history_current = temporary.session_history

    with gr.Blocks() as interface:
        with gr.Tabs():
            # Conversation Tab
            with gr.Tab("Conversation"):
                gr.Markdown("# Chat-Ubuntu-Gguf")

                with gr.Row():
                    with gr.Column(scale=3):
                        bot_response = gr.Textbox(
                            label="AI Response",
                            lines=10,
                            value="",
                            interactive=False
                        )
                        user_input = gr.Textbox(
                            label="Your Input",
                            lines=2,
                            placeholder="Type your message here...",
                            interactive=True
                        )
                    with gr.Column(scale=1):
                        session_history_display = gr.Textbox(
                            label="Conversation History",
                            lines=15,
                            value=session_history_current,
                            interactive=False
                        )

                with gr.Row():
                    send_btn = gr.Button("Send Message")
                    reset_btn = gr.Button("Reset Session")
                    exit_btn = gr.Button("Exit Program")

                send_btn.click(
                    fn=chat_with_model,
                    inputs=user_input,
                    outputs=[bot_response, session_history_display]
                )
                reset_btn.click(
                    fn=reset_session,
                    inputs=[],
                    outputs=[bot_response, session_history_display]
                )
                exit_btn.click(
                    fn=shutdown,
                    inputs=[],
                    outputs=[]
                )

            # Configuration Tab
            with gr.Tab("Configuration"):
                gr.Markdown("# Configuration Settings")

                agent_name_input = gr.Textbox(label="Agent Name", value=temporary.agent_name)
                agent_role_input = gr.Textbox(label="Agent Role", value=temporary.agent_role)
                human_name_input = gr.Textbox(label="Human Name", value=temporary.human_name)

                session_history_input = gr.Textbox(
                    label="Default History",
                    value=temporary.session_history
                )

                threads_slider = gr.Slider(
                    label="Threads Usage (%)",
                    minimum=10,
                    maximum=100,
                    step=10,
                    value=temporary.threads_percent
                )

                with gr.Row():
                    apply_btn = gr.Button("Apply")
                    save_btn = gr.Button("Save")

                apply_btn.click(
                    fn=apply_configuration,
                    inputs=[agent_name_input, agent_role_input, human_name_input],
                    outputs=[]
                )
                save_btn.click(
                    fn=save_configuration,
                    inputs=[
                        agent_name_input,
                        agent_role_input,
                        human_name_input,
                        threads_slider,
                        session_history_input
                    ],
                    outputs=[]
                )

    interface.launch(inbrowser=True)
