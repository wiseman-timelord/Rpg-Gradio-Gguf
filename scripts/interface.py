# .\scripts\interface.py

from scripts.utility import reset_session_state, write_to_yaml
from data.temporary import session_history, agent_output, agent_name, agent_role, human_name, threads_percent
import gradio as gr
import threading
import time
import signal
import os

def simulate_processing(delay=3):
    """
    Simulates processing time for AI response generation.
    """
    time.sleep(delay)

def shutdown():
    """
    Gracefully shuts down the application.
    """
    print("Shutting down the application...")
    os.kill(os.getpid(), signal.SIGTERM)  # Terminate the process

def reset_session():
    """
    Resets the session state and clears relevant UI components.
    """
    global agent_output, session_history, human_input
    reset_session_state()  # Reset globals
    agent_output = ""
    session_history = "The conversation started."
    human_input = ""
    return "", session_history

def apply_configuration(new_agent_name, new_agent_role, new_human_name):
    """
    Applies configuration changes by updating temporary variables.
    """
    global agent_name, agent_role, human_name
    agent_name = new_agent_name
    agent_role = new_agent_role
    human_name = new_human_name
    return "Configuration applied successfully!"

def save_configuration(new_agent_name, new_agent_role, new_human_name, new_threads_percent):
    """
    Saves configuration changes to both temporary variables and persistent.yaml.
    """
    global agent_name, agent_role, human_name, threads_percent
    agent_name = new_agent_name
    agent_role = new_agent_role
    human_name = new_human_name
    threads_percent = new_threads_percent

    write_to_yaml('agent_name', agent_name)
    write_to_yaml('agent_role', agent_role)
    write_to_yaml('human_name', human_name)
    write_to_yaml('threads_percent', threads_percent)

    return "Configuration and thread settings saved successfully!"

def save_threads_percent(new_threads_percent):
    """
    Saves the threads_percent setting to both temporary variables and persistent.yaml.
    """
    global threads_percent
    threads_percent = new_threads_percent
    write_to_yaml('threads_percent', threads_percent)
    return f"Threads percentage set to {threads_percent}%."

def chat_with_model(user_input):
    """
    Handles chat interaction with the AI model.
    Combines user input and session history for prompt, updates session history.
    """
    global session_history, agent_output

    try:
        if not user_input.strip():
            raise ValueError("Input cannot be empty.")

        # Simulate processing delay
        processing_thread = threading.Thread(target=simulate_processing)
        processing_thread.start()

        # Construct prompt
        prompt = f"{session_history}\nUser: {user_input}\nAI:"
        agent_output = f"Mock response to: {user_input}"  # Replace with actual AI logic
        session_history += f"\nUser: {user_input}\nAI: {agent_output}"

        return agent_output, session_history
    except Exception as e:
        return f"Error: {e}", session_history

def launch_gradio_interface():
    """
    Launches the Gradio interface with tabs for Conversation and Configuration.
    """
    with gr.Blocks() as interface:
        with gr.Tabs():
            # Tab 1: Conversation
            with gr.Tab("Conversation"):
                gr.Markdown("# Chat-Ubuntu-Gguf")

                # Conversation Layout
                with gr.Row():
                    with gr.Column(scale=3):
                        bot_response = gr.Textbox(
                            label="AI Response",
                            lines=10,
                            value=agent_output,
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
                            value=session_history,
                            interactive=False
                        )

                # Button Row at the bottom
                with gr.Row():
                    send_btn = gr.Button("Send Message")
                    reset_btn = gr.Button("Reset Session")
                    exit_btn = gr.Button("Exit Program")

                # Button Actions
                send_btn.click(
                    fn=chat_with_model,
                    inputs=user_input,
                    outputs=[bot_response, session_history_display]
                )
                reset_btn.click(
                    fn=reset_session,
                    inputs=[],
                    outputs=[bot_response, session_history_display]  # Two outputs
                )
                gr.HTML(
                    value="<script>function closeTab() { window.open('','_self').close(); }</script>"
                )
                exit_btn.click(
                    fn=shutdown,  # Python shutdown function
                    inputs=[],
                    outputs=[]
                )

            # Tab 2: Configuration
            with gr.Tab("Configuration"):
                gr.Markdown("# Configuration Settings")
                
                # Configuration Layout
                agent_name_input = gr.Textbox(label="Agent Name", value=agent_name)
                agent_role_input = gr.Textbox(label="Agent Role", value=agent_role)
                human_name_input = gr.Textbox(label="Human Name", value=human_name)
                
                # Thread Percentage Slider
                threads_slider = gr.Slider(
                    label="Threads Usage (%)",
                    minimum=10,
                    maximum=100,
                    step=10,
                    value=threads_percent
                )

                with gr.Row():
                    apply_btn = gr.Button("Apply")
                    save_btn = gr.Button("Save")

                # Button Actions for Configuration
                apply_btn.click(
                    fn=apply_configuration,
                    inputs=[agent_name_input, agent_role_input, human_name_input],
                    outputs=None
                )
                save_btn.click(
                    fn=save_configuration,
                    inputs=[agent_name_input, agent_role_input, human_name_input, threads_slider],
                    outputs=None
                )

    # Launch the interface
    interface.launch(inbrowser=True)

