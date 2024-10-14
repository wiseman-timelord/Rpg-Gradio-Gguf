# .\scripts\utility.py

# imports
import yaml
import os
import time
import threading
import subprocess

# Define the ramfs directory
RAMFS_DIR = '/mnt/ramfs'

# Function to read YAML file
def read_yaml(file_path=os.path.join(RAMFS_DIR, 'persistent.yaml')):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: persistent.yaml not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading persistent.yaml: {e}")
        return None

# Function to write to YAML file
def write_to_yaml(key, value, file_path=os.path.join(RAMFS_DIR, 'persistent.yaml')):
    data = read_yaml(file_path) or {}
    data[key] = value
    try:
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file)
    except Exception as e:
        print(f"An error occurred while writing to persistent.yaml: {e}")

# Function to reset keys to empty
def reset_keys_to_empty():
    yaml_data = {
        'agent_name': "Empty",
        'agent_role': "Empty",
        'scenario_location': "Empty",
        'agent_emotion': "Empty",
        'session_history': "Empty",
        'human_input': "Empty",
        'agent_output_1': "Empty",
        'agent_output_2': "Empty",
        'agent_output_3': "Empty",
        'sound_event': "None",
        'context_length': "Empty",
        'syntax_type': "Empty",
        'model_path': "Empty"
    }
    for key, value in yaml_data.items():
        write_to_yaml(key, value)

# Function to calculate optimal threads
def calculate_optimal_threads():
    cpu_count = os.cpu_count()
    optimal_threads = cpu_count // 2 if cpu_count > 1 else 1
    print(f"Optimal threads calculated: {optimal_threads}")
    return optimal_threads

# Function to trigger sound event
def trigger_sound_event(event_name):
    sound_file = f"./data/sounds/{event_name}.wav"
    if os.path.exists(sound_file):
        play_wav(sound_file)

# Function to play WAV file
def play_wav(filename):
    subprocess.run(['aplay', filename], check=True) 

# Function for fancy delay
def fancy_delay(seconds):
    for i in range(seconds, 0, -1):
        print(f" ...{i} seconds remaining...", end='\r')
        time.sleep(1)
    print(" ...Done!               ")

# Function to shift responses
def shift_responses():
    data = read_yaml()
    data['agent_output_3'] = data['agent_output_2']
    data['agent_output_2'] = data['agent_output_1']
    data['agent_output_1'] = "Empty"
    for key, value in data.items():
        write_to_yaml(key, value)

# Function to clear debug logs
def clear_debug_logs():
    log_file = './data/logs/debug.log'
    if os.path.exists(log_file):
        os.remove(log_file)
        print("Debug logs cleared.")
    else:
        print("No debug logs found.")

# Function to extract context key from model name
def extract_context_key_from_model_name(model_name):
    match = re.search(r'ctx(\d+)', model_name, re.IGNORECASE)
    return match.group(1) if match else None
