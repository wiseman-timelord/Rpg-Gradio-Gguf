# scripts/interface.py

import os, signal
import data.temporary as temporary
from data.temporary import (agent_name, agent_role, human_name, session_history, threads_percent,
                            rotation_counter, agent_output, human_input, PROMPT_TO_SETTINGS, IMAGE_SIZE_OPTIONS)
from scripts.utility import reset_session_state, write_to_yaml, read_yaml
from scripts.model import prompt_response, generate_image_from_history
import gradio as gr
import threading, time

def shutdown():
    print("Shutting down the application...")
    os.kill(os.getpid(), signal.SIGTERM)

def reset_session():
    read_config = read_yaml('./data/persistent.yaml')
    temporary.agent_name = read_config.get('agent_name', 'Wise-Llama')
    temporary.agent_role = read_config.get('agent_role', 'A wise oracle of sorts')
    temporary.human_name = read_config.get('human_name', 'Human')
    temporary.session_history = read_config.get('session_history', "the conversation started")
    temporary.selected_steps = read_config.get('selected_steps', temporary.selected_steps)
    temporary.selected_sample_method = read_config.get('selected_sample_method', temporary.selected_sample_method)

    reset_session_state()
    # Return updated UI states, including the default image
    return temporary.session_history, "", "./data/new_session.jpg"


def apply_configuration(new_agent_name, new_agent_role, new_human_name):
    temporary.agent_name = new_agent_name
    temporary.agent_role = new_agent_role
    temporary.human_name = new_human_name

    # Return the updated values to reflect in UI inputs
    return new_agent_name, new_agent_role, new_human_name


def update_keys(new_threads_percent=None, new_agent_name=None, new_agent_role=None, new_human_name=None, new_session_history=None, new_image_size=None, new_steps_used=None, new_sample_method=None):
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
    if new_image_size is not None:
        temporary.IMAGE_SIZE_OPTIONS['selected_size'] = new_image_size  # Store as string
    if new_steps_used is not None:
        temporary.selected_steps = int(new_steps_used)  # Store as integer
    if new_sample_method is not None:
        temporary.selected_sample_method = new_sample_method  # Store as string

    data_to_save = {
        'agent_name': temporary.agent_name,
        'agent_role': temporary.agent_role,
        'human_name': temporary.human_name,
        'threads_percent': temporary.threads_percent,
        'session_history': temporary.session_history,
        'selected_steps': temporary.selected_steps,
        'selected_sample_method': temporary.selected_sample_method
    }
    write_to_yaml(data_to_save, './data/persistent.yaml')
    return "Settings updated successfully!"



def filter_model_output(raw_output):
    if ":" in raw_output[:25]:
        response = raw_output.split(":", 1)[1].strip()
        return f"{temporary.agent_name}: {response}"
    else:
        return raw_output

def chat_with_model(user_input):
    if not user_input.strip():
        return "Error: Input cannot be empty.", temporary.session_history, None

    temporary.human_input = f"{temporary.human_name}: {user_input.strip()}"

    converse_result = prompt_response('converse', temporary.rotation_counter)
    if 'error' in converse_result:
        return f"Error in model response: {converse_result['error']}", temporary.session_history, None

    filtered_response = filter_model_output(converse_result['agent_response'])
    temporary.agent_output = filtered_response

    consolidate_result = prompt_response('consolidate', temporary.rotation_counter)
    if 'error' in consolidate_result:
        return temporary.agent_output, temporary.session_history, None

    temporary.session_history = consolidate_result['agent_response']

    image_path = generate_image_from_history(temporary.session_history)
    temporary.latest_image_path = image_path

    return temporary.agent_output, temporary.session_history, image_path

def launch_gradio_interface():
    def get_hardware_details():
        try:
            with open('./data/hardware_details.txt', 'r') as file:
                return file.read()
        except FileNotFoundError:
            return "Hardware details file not found. Please run the setup installer."

    with gr.Blocks() as interface:
        with gr.Tabs():
            with gr.Tab("Conversation"):
                gr.Markdown("# Chat Interface")
                with gr.Row():
                    with gr.Column(scale=1):
                        bot_response = gr.Textbox(label="Agent Output:", lines=9, value="", interactive=False)
                        user_input = gr.Textbox(label="User Input:", lines=9, placeholder="Type your message here...", interactive=True)
                    with gr.Column(scale=1):
                        session_history_display = gr.Textbox(label="Consolidated History", lines=21, value=temporary.session_history, interactive=False)
                    with gr.Column(scale=1):
                        generated_image = gr.Image(
                            label="Generated Image",
                            type="filepath",
                            interactive=False,
                            height=490,
                            value="./data/new_session.jpg"  # Set default image upon initialization
                        )

                with gr.Row():
                    send_btn = gr.Button("Send Message", elem_id="send_message_button")
                    reset_btn = gr.Button("Restart Session", elem_id="restart_session_button")
                    exit_btn = gr.Button("Exit Program", elem_id="exit_program_button")

                send_btn.click(fn=chat_with_model, inputs=user_input, outputs=[bot_response, session_history_display, generated_image])
                reset_btn.click(fn=reset_session, inputs=[], outputs=[session_history_display, bot_response, generated_image])
                exit_btn.click(fn=shutdown, inputs=[], outputs=[])

            with gr.Tab("Configuration"):
                gr.Markdown("# Configuration Settings")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Roleplay Parameters")
                        agent_name_input = gr.Textbox(label="Agent Name", value=temporary.agent_name)
                        agent_role_input = gr.Textbox(label="Agent Role", value=temporary.agent_role)
                        human_name_input = gr.Textbox(label="Human Name", value=temporary.human_name)
                        session_history_input = gr.Textbox(
                            label="Default History",
                            value=temporary.session_history
                        )

                    with gr.Column():
                        gr.Markdown("### Image Generation Settings")
                        image_size_dropdown = gr.Dropdown(
                            label="Image Size",
                            choices=temporary.IMAGE_SIZE_OPTIONS['available_sizes'],
                            value=temporary.IMAGE_SIZE_OPTIONS['selected_size'],
                            type="value"
                        )
                        steps_dropdown = gr.Dropdown(
                            label="Steps Used",
                            choices=temporary.STEPS_OPTIONS,
                            value=temporary.selected_steps,
                            type="value"
                        )
                        sample_method_dropdown = gr.Dropdown(
                            label="Sample Method",
                            choices=[method[0] for method in temporary.SAMPLE_METHOD_OPTIONS],  # List of labels
                            value=temporary.selected_sample_method,    # String value
                            type="value"
                        )
                        gr.Markdown("### Hardware Settings")
                        hardware_details_display = gr.Textbox(
                            label="Detected Hardware Details",
                            value=get_hardware_details(),
                            lines=3,
                            interactive=False
                        )
                        threads_slider = gr.Slider(
                            label="CPU Threads Usage (%)",
                            minimum=11,
                            maximum=100,
                            step=10,
                            value=temporary.threads_percent
                        )

                # Save Configuration Button
                with gr.Row():
                    save_configuration_btn = gr.Button("Save Configuration", elem_id="save_configuration_button")

                # Define Save Configuration button action
                save_configuration_btn.click(
                    fn=lambda new_agent_name, new_agent_role, new_human_name, new_session_history, new_threads_percent, new_image_size, new_steps_used, new_sample_method: update_keys(
                        new_agent_name=new_agent_name,
                        new_agent_role=new_agent_role,
                        new_human_name=new_human_name,
                        new_session_history=new_session_history,
                        new_threads_percent=new_threads_percent,
                        new_image_size=new_image_size,
                        new_steps_used=new_steps_used,
                        new_sample_method=new_sample_method
                    ),
                    inputs=[
                        agent_name_input,
                        agent_role_input,
                        human_name_input,
                        session_history_input,
                        threads_slider,
                        image_size_dropdown,
                        steps_dropdown,
                        sample_method_dropdown
                    ],
                    outputs=[]
                )

    interface.launch(inbrowser=True)

