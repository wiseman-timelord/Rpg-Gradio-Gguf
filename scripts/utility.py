# .\scripts\utility.py

# Imports
import yaml
import os
import time
from data.temporary import session_history, agent_output, human_input

# Define the RAMFS directory
RAMFS_DIR = '/mnt/ramfs'

# Function to read YAML file
def read_yaml(file_path='./data/persistent.yaml'):
    """
    Reads the YAML file and returns its contents as a dictionary.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error reading YAML: {e}")
        return {}

def write_to_yaml(key, value, file_path='./data/persistent.yaml'):
    """
    Writes a key-value pair to the YAML file.
    """
    try:
        data = read_yaml(file_path) or {}
        data[key] = value
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file)
    except Exception as e:
        print(f"Error writing to YAML: {e}")


# reset
def reset_session_state():
    """
    Resets session-specific variables to their default values.
    """
    global session_history, agent_output, human_input
    session_history = "the conversation started"
    agent_output = ""
    human_input = ""


# Function to calculate optimal threads
def calculate_optimal_threads(threads_percent=80):
    """
    Calculates the optimal number of threads to use based on the percentage provided.
    """
    cpu_count = os.cpu_count()
    optimal_threads = max(1, (cpu_count * threads_percent) // 100)
    print(f"Optimal threads based on {threads_percent}% of {cpu_count} cores: {optimal_threads}")
    return optimal_threads

def scan_models_directory(models_dir='./models'):
    """
    Scans the models directory for GGUF models and their corresponding JSON configs.
    """
    models = []
    for file in os.listdir(models_dir):
        if file.endswith('.gguf'):
            json_path = os.path.join(models_dir, 'model_config.json')
            if os.path.exists(json_path):
                models.append({
                    'model_path': os.path.join(models_dir, file),
                    'config_path': json_path
                })
    return models



