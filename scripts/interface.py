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
    Soft reset of session variables
    """
    # Read the latest configuration from YAML
    read_config = read_yaml('./data/persistent.yaml')
    
    # Reset temporary variables to the values from the configuration
    temporary.agent_name = read_config.get('agent_name', 'Wise-Llama')
    temporary.agent_role = read_config.get('agent_role', 'A wise oracle of sorts')
    temporary.human_name = read_config.get('human_name', 'Human')
    temporary.session_history = read_config.get('session_history', "the conversation started")
    
    # Reset session-specific variables
    reset_session_state()

    # Return the updated values to reflect in UI inputs
    return temporary.session_history, "", ""

def apply_configuration(new_agent_name, new_agent_role, new_human_name):
    temporary.agent_name = new_agent_name
    temporary.agent_role = new_agent_role
    temporary.human_name = new_human_name

    # Return the updated values to reflect in UI inputs
    return new_agent_name, new_agent_role, new_human_name

def update_keys(new_threads_percent=None, new_agent_name=None, new_agent_role=None, new_human_name=None, new_session_history=None):
    """
    Updates temporary variables and saves them to persistent.yaml.
    """
    if new_threads_percent is not None:
        temporary.threads_percent = int(new_threads_percent)

    if new_agent_name is not None:
        temporary.agent_name = new_agent_name
    if new_agent_role is not None:
        temporary.agent_role = new_agent_role
    if new_human_name is not None:
        temporary.human_name = new_human_name
    if new_session_history is not None:
        temporary.session_history = new_session_history

    # Save updated values to YAML
    data_to_save = {
        'agent_name': temporary.agent_name,
        'agent_role': temporary.agent_role,
        'human_name': temporary.human_name,
        'threads_percent': temporary.threads_percent,
        'session_history': temporary.session_history
    }
    write_to_yaml(data_to_save, './data/persistent.yaml')
    return "Settings updated successfully!"


def filter_model_output(raw_output):
    """
    Filters the model output to extract the response after the first colon in the first 25 characters.
    """
    if ":" in raw_output[:25]:
        # Extract the text after the first colon
        response = raw_output.split(":", 1)[1].strip()
        return f"{temporary.agent_name}: {response}"
    else:
        return raw_output

def chat_with_model(user_input):
    """
    Handles the chat interaction: sends user input to the model and processes responses.
    """
    if not user_input.strip():
        return "Error: Input cannot be empty.", ""

    # Update the human_input variable
    temporary.human_input = f"{temporary.human_name}: {user_input.strip()}"

    # Process 'converse' prompt
    converse_result = prompt_response('converse', temporary.rotation_counter)
    if 'error' in converse_result:
        return f"Error in model response: {converse_result['error']}", ""

    # Filter the model's response
    filtered_response = filter_model_output(converse_result['agent_response'])

    # Update `agent_output` with the filtered response
    temporary.agent_output = filtered_response

    # Process 'consolidate' prompt
    consolidate_result = prompt_response('consolidate', temporary.rotation_counter)
    if 'error' in consolidate_result:
        return temporary.agent_output, f"Error in model consolidation: {consolidate_result['error']}"

    # Update `session_history` with only the latest response from 'consolidate'
    temporary.session_history = consolidate_result['agent_response']

    # Return results for UI update
    return temporary.agent_output, temporary.session_history

def launch_gradio_interface():
    def get_hardware_details():
        """
        Reads and returns the contents of the hardware details file.
        """
        try:
            with open('./data/hardware_details.txt', 'r') as file:
                return file.read()
        except FileNotFoundError:
            return "Hardware details file not found. Please run the setup installer."

    with gr.Blocks() as interface:
        with gr.Tabs():
            # Conversation Tab
            with gr.Tab("Conversation"):
                gr.Markdown("# Chat Interface")
                with gr.Row():
                    with gr.Column(scale=1):
                        bot_response = gr.Textbox(label="Agent Output:", lines=10, value="", interactive=False)
                        user_input = gr.Textbox(label="User Input:", lines=10, placeholder="Type your message here...", interactive=True)
                    with gr.Column(scale=1):
                        session_history_display = gr.Textbox(label="Consolidated History", lines=23, value=temporary.session_history, interactive=False)
                with gr.Row():
                    with gr.Column(scale=1):  # Send Message button twice as wide
                        send_btn = gr.Button("Send Message")
                    with gr.Column():  # Reset and Exit buttons half as wide each
                        reset_btn = gr.Button("Restart Session")
                        exit_btn = gr.Button("Exit Program")
                # Define button actions
                send_btn.click(fn=chat_with_model, inputs=user_input, outputs=[bot_response, session_history_display])
                reset_btn.click(fn=reset_session, inputs=[], outputs=[session_history_display, bot_response, user_input])
                exit_btn.click(fn=shutdown, inputs=[], outputs=[])

            # Parameters Tab
            with gr.Tab("Parameters"):
                gr.Markdown("# Roleplay Parameters")
                agent_name_input = gr.Textbox(label="Agent Name", value=temporary.agent_name)
                agent_role_input = gr.Textbox(label="Agent Role", value=temporary.agent_role)
                human_name_input = gr.Textbox(label="Human Name", value=temporary.human_name)
                session_history_input = gr.Textbox(
                    label="Default History",
                    value=temporary.session_history
                )
                update_roleplay_btn = gr.Button("Save Roleplay Keys")
                update_roleplay_btn.click(
                    fn=lambda new_agent_name, new_agent_role, new_human_name, new_session_history: update_keys(
                        new_agent_name=new_agent_name,
                        new_agent_role=new_agent_role,
                        new_human_name=new_human_name,
                        new_session_history=new_session_history
                    ),
                    inputs=[agent_name_input, agent_role_input, human_name_input, session_history_input],
                    outputs=[]
                )

            # Hardware Tab
            with gr.Tab("Hardware"):
                gr.Markdown("# Hardware Settings")
                hardware_details_display = gr.Textbox(
                    label="Detected Hardware Details",
                    value=get_hardware_details(),
                    lines=4,  # Increase vertical space by setting more lines
                    interactive=False
                )
                threads_slider = gr.Slider(
                    label="CPU Threads Usage (%)",
                    minimum=10,
                    maximum=100,
                    step=10,
                    value=temporary.threads_percent
                )

                update_hardware_btn = gr.Button("Save Hardware Keys")
                update_hardware_btn.click(
                    fn=lambda new_threads_percent: update_keys(new_threads_percent=new_threads_percent),
                    inputs=[threads_slider],
                    outputs=[]
                )

    interface.launch(inbrowser=True)

