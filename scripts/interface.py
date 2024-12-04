# .\scripts\interface.py

from scripts.utility import reset_session_state, write_to_yaml
from data.temporary import session_history, agent_output, agent_name, agent_role, human_name
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
    os.kill(os.getpid(), signal.SIGTERM)

def reset_session():
    """
    Resets the session state without modifying persistent settings.
    """
    reset_session_state()
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

def save_configuration(new_agent_name, new_agent_role, new_human_name):
    """
    Saves configuration changes to both temporary variables and persistent.yaml.
    """
    apply_configuration(new_agent_name, new_agent_role, new_human_name)
    write_to_yaml('agent_name', agent_name)
    write_to_yaml('agent_role', agent_role)
    write_to_yaml('human_name', human_name)
    return "Configuration saved successfully!"

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
                    outputs=[bot_response, session_history_display]
                )
                exit_btn.click(
                    fn=lambda: ("Goodbye!", ""),
                    inputs=[],
                    outputs=[bot_response, session_history_display]
                )

            # Tab 2: Configuration
            with gr.Tab("Configuration"):
                gr.Markdown("# Configuration Settings")
                # Configuration Layout
                agent_name_input = gr.Textbox(label="Agent Name", value=agent_name)
                agent_role_input = gr.Textbox(label="Agent Role", value=agent_role)
                human_name_input = gr.Textbox(label="Human Name", value=human_name)

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
                    inputs=[agent_name_input, agent_role_input, human_name_input],
                    outputs=None
                )

    interface.launch()


