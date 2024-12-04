# ./main_script.py

# Imports
import os
import time
import yaml
import argparse
import threading
import webbrowser
import gradio as gr
from scripts import interface, model as agent_module, utility
from data.temporary import PERSISTENT_FILE, session_history, agent_name, agent_role, human_name
from scripts.utility import read_yaml, write_to_yaml


def load_and_initialize_model():
    """
    Loads and initializes the model as part of the main program flow.
    """
    print("Scanning models directory and initializing model...")
    from scripts.model import process_selected_model  # Ensure import of new function
    process_selected_model(models_dir='./models')


def load_persistent_settings():
    """
    Loads persistent settings from persistent.yaml and updates the corresponding temporary variables.
    """
    global agent_name, agent_role, human_name, session_history

    if not os.path.exists(PERSISTENT_FILE):
        print(f"Error: Persistent file {PERSISTENT_FILE} not found. Please run the setup installer.")
        exit(1)

    persistent_data = read_yaml(PERSISTENT_FILE)
    agent_name = persistent_data.get('agent_name', 'Empty')
    agent_role = persistent_data.get('agent_role', 'Empty')
    human_name = persistent_data.get('human_name', 'Empty')
    session_history = persistent_data.get('session_history', session_history)  # Default to initial value


def save_persistent_settings():
    """
    Saves current settings to persistent.yaml.
    """
    data = {
        'agent_name': agent_name,
        'agent_role': agent_role,
        'human_name': human_name,
        'session_history': session_history
    }
    write_to_yaml(data, PERSISTENT_FILE)


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
    """
    Launch Gradio server and background engine, and open the browser.
    """
    # Start the background engine thread
    engine_thread = threading.Thread(target=background_engine, daemon=True)
    engine_thread.start()

    # Launch Gradio interface in the main thread
    interface.launch_gradio_interface()

    # Open the default browser explicitly to access Gradio
    webbrowser.open("http://127.0.0.1:7860", new=1)


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
        write_to_yaml(key, "Empty", PERSISTENT_FILE)


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

