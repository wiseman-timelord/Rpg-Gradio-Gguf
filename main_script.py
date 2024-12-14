# main_script.py

import os
import time
import yaml
import argparse
import threading
import webbrowser
import gradio as gr
import data.temporary  # for explicit writes back to temporary
from data.temporary import (session_history, agent_name, agent_role, human_name,
                            threads_percent, optimal_threads, model_used, large_language_model)
from scripts.utility import read_yaml, write_to_yaml
from scripts import interface, model as agent_module

def load_and_initialize_model():
    """
    Loads and initializes the model as part of the main program flow.
    """
    print("Scanning models directory and initializing model...")
    from scripts.model import process_selected_model
    process_selected_model()

def load_persistent_settings():
    """
    Loads persistent settings from persistent.yaml and updates the corresponding temporary variables.
    """
    try:
        persistent_data = read_yaml('./data/persistent.yaml')
    except FileNotFoundError:
        print("Error: Persistent file not found. Please run the setup installer.")
        exit(1)

    # Update temporary variables from persistent data
    data.temporary.agent_name = persistent_data.get('agent_name', data.temporary.agent_name)
    data.temporary.agent_role = persistent_data.get('agent_role', data.temporary.agent_role)
    data.temporary.human_name = persistent_data.get('human_name', data.temporary.human_name)
    data.temporary.session_history = persistent_data.get('session_history', "The conversation started")
    data.temporary.threads_percent = persistent_data.get('threads_percent', data.temporary.threads_percent)

    total_threads = os.cpu_count()
    data.temporary.optimal_threads = max(1, (total_threads * data.temporary.threads_percent) // 100)
    print(f"Using {data.temporary.optimal_threads} threads out of {total_threads} ({data.temporary.threads_percent}%)")


def save_persistent_settings():
    """
    Saves current settings from data.temporary to persistent.yaml.
    This is called when the user presses the "Save Settings" button, not automatically.
    """
    data_to_save = {
        'agent_name': data.temporary.agent_name,
        'agent_role': data.temporary.agent_role,
        'human_name': data.temporary.human_name,
        'threads_percent': data.temporary.threads_percent,
        'session_history': data.temporary.session_history
    }
    write_to_yaml(data_to_save, './data/persistent.yaml')
    print("Configuration and session settings saved successfully to persistent.yaml.")

def background_engine():
    """
    Background process to monitor updates to persistent.yaml.
    If the file is updated (i.e., user saved settings), reload settings.
    """
    last_modified_time = None
    print("Background engine is running...")
    while True:
        try:
            if os.path.exists('./data/persistent.yaml'):
                current_modified_time = os.path.getmtime('./data/persistent.yaml')
                if last_modified_time is None:
                    last_modified_time = current_modified_time
                else:
                    if current_modified_time > last_modified_time:
                        print("Persistent file updated. Reloading session settings...")
                        load_persistent_settings()
                        last_modified_time = current_modified_time
            time.sleep(5)
        except Exception as e:
            print(f"Error in background engine: {e}")

def launch_gradio_with_background_engine():
    """
    Launch Gradio server and background engine, and open the browser.
    """
    # Start the background engine thread
    engine_thread = threading.Thread(target=background_engine, daemon=True)
    engine_thread.start()

    # Launch Gradio interface in the main thread
    interface.launch_gradio_interface()

    # Open the default browser explicitly
    webbrowser.open("http://127.0.0.1:7860", new=1)

def initialize_session():
    """
    Initializes session variables from the YAML configuration file into data.temporary.
    This does not write anything back to YAML yet.
    """
    persistent_data = read_yaml('./data/persistent.yaml')
    data.temporary.session_history = persistent_data.get('session_history', "The conversation started.")

if __name__ == "__main__":
    load_persistent_settings()
    load_and_initialize_model()
    launch_gradio_with_background_engine()
