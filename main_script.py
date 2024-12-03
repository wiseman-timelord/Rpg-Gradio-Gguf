# ./main_script.py

# Imports
import os
import time
import yaml
import argparse
import threading
import gradio as gr
from scripts import interface, model as agent_module, utility
from data.temporary import RAMFS_DIR, PERSISTENT_FILE, session_history, agent_name, agent_role, human_name
from scripts.utility import read_yaml, write_to_yaml

def load_and_initialize_model():
    """
    Loads and initializes the model as part of the main program flow.
    """
    print("Scanning models directory and initializing model...")
    process_selected_model(models_dir='./models')

def load_persistent_settings():
    """
    Loads persistent settings from persistent.yaml and updates the corresponding temporary variables.
    """
    global agent_name, agent_role, human_name
    persistent_data = read_yaml()
    agent_name = persistent_data.get('agent_name', 'Empty')
    agent_role = persistent_data.get('agent_role', 'Empty')
    human_name = persistent_data.get('human_name', 'Empty')

def save_persistent_settings():
    """
    Saves current settings to persistent.yaml.
    """
    write_to_yaml('agent_name', agent_name)
    write_to_yaml('agent_role', agent_role)
    write_to_yaml('human_name', human_name)

# Background Engine with YAML Monitoring
def background_engine():
    last_modified_time = os.path.getmtime(PERSISTENT_FILE) if os.path.exists(PERSISTENT_FILE) else None
    print("Background engine is running...")
    
    while True:
        try:
            if os.path.exists(PERSISTENT_FILE):
                current_modified_time = os.path.getmtime(PERSISTENT_FILE)
                if last_modified_time is None or current_modified_time > last_modified_time:
                    print("Persistent file updated. Reloading session...")
                    last_modified_time = current_modified_time
            time.sleep(5)
        except Exception as e:
            print(f"Error in background engine: {e}")

# Gradio Interface Launcher with Background Engine
def launch_gradio_with_background_engine():
    session_state = {'rotation_counter': 0}

    def chat_with_model(user_input):
        response, session_history = handle_chat_input(
            user_input, 
            session_state['rotation_counter']
        )
        session_state['rotation_counter'] = (session_state['rotation_counter'] + 1) % 4
        return response, session_history

    engine_thread = threading.Thread(target=background_engine, daemon=True)
    engine_thread.start()

    interface = gr.Interface(
        fn=chat_with_model,
        inputs=gr.Textbox(lines=2, label="Your Message"),
        outputs=[
            gr.Textbox(lines=4, label="Response"),
            gr.Textbox(lines=10, label="Session History")
        ],
        title="Chat-Ubuntu-Gguf",
    )
    interface.launch()

def reset_keys_to_empty():
    """
    Resets predefined keys in the YAML file to their default 'Empty' values.
    """
    keys = [
        'agent_name', 'agent_role', 'session_history',
        'human_input', 'agent_output_1', 'agent_output_2',
        'agent_output_3', 'context_length', 'syntax_type', 'model_path'
    ]
    for key in keys:
        write_to_yaml(key, "Empty")

def initialize_session():
    """
    Initializes session variables from temporary.py and loads persistent settings.
    """
    global session_history
    session_history = "the conversation started"  # Default state

if __name__ == "__main__":
    load_and_initialize_model()
    load_persistent_settings()
    launch_gradio_with_background_engine()

