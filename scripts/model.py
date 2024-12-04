# .\scripts\model.py

# imports
from llama_cpp import Llama
from scripts import utility
from gguf_parser import GGUFParser
import os
import time
import re
import json
import subprocess  # Corrected typo
from data.temporary import MODE_TO_TEMPERATURE, PROMPT_TO_MAXTOKENS, PROMPT_TO_SETTINGS

# Define the ramfs directory
RAMFS_DIR = '/mnt/ramfs'

def process_selected_model(models_dir='./models'):
    """
    Scans the models directory, selects a model, and initializes it.
    """
    print(f"Scanning {models_dir} for models...")
    models = utility.scan_models_directory(models_dir)
    if not models:
        raise FileNotFoundError("No valid GGUF models found in the directory.")
    
    # Select the first available model (or add logic for user selection if needed)
    selected_model = models[0]
    print(f"Selected model: {selected_model['model_path']}")
    
    # Initialize the model
    initialize_model(models_dir=models_dir, optimal_threads=utility.calculate_optimal_threads())
    print("Model selection and initialization completed.")


# initialize the model
def initialize_model(models_dir='./models', optimal_threads=None):
    """
    Initializes a GGUF model, extracting metadata using gguf-parser.
    Optionally uses configuration files for additional overrides.
    """
    try:
        # Locate GGUF file
        gguf_files = [f for f in os.listdir(models_dir) if f.endswith('.gguf')]
        if not gguf_files:
            raise FileNotFoundError("No GGUF model files found in the directory.")
        
        model_path = os.path.join(models_dir, gguf_files[0])
        print(f"Using GGUF model file: {model_path}")

        # Parse metadata from GGUF file
        parser = GGUFParser(model_path)
        parser.parse()
        
        # Access metadata directly
        metadata = parser.metadata

        # Display GGUF metadata
        print("Metadata extracted from GGUF file:")
        for key, value in metadata.items():
            print(f"{key}: {value}")

        # Check for optional config files
        config_files = [os.path.join(models_dir, f) for f in ('model_config.json', 'config.json') if os.path.exists(os.path.join(models_dir, f))]
        if config_files:
            with open(config_files[0], 'r') as config_file:
                config_data = json.load(config_file)
            print("Configuration file data:", json.dumps(config_data, indent=2))

            # Merge config data into metadata
            metadata.update(config_data)

        # Include optimal_threads in metadata if provided
        if optimal_threads is not None:
            metadata['optimal_threads'] = optimal_threads
            print(f"Using optimal threads: {optimal_threads}")

        # Validate required parameters
        required_parameters = [
            'llama.context_length', 'llama.embedding_length', 'llama.feed_forward_length',
            'llama.attention.head_count', 'llama.attention.head_count_kv', 'llama.vocab_size',
            'general.architecture', 'general.name'
        ]

        missing_parameters = [param for param in required_parameters if param not in metadata]
        if missing_parameters:
            print("Warning: Missing required parameters:")
            for param in missing_parameters:
                print(f" - {param}")

        # Confirm all parameters for initialization
        print("\nFinal Parameters for Model Initialization:")
        for key, value in metadata.items():
            print(f"{key}: {value}")

        if missing_parameters:
            print("Missing parameters, using defaults.")
        else:
            print("All required parameters are present.")

        # Initialize model (use metadata as needed for your library)
        # model = YourModelLibrary(model_path=model_path, **metadata)

        print("Model initialized successfully.")
    except Exception as e:
        print(f"Error during model initialization: {e}")

def run_llama_cli(prompt, max_tokens, temperature):
    try:
        if not model:
            raise RuntimeError("Model is not initialized. Please load a valid model.")
        response = model(prompt, max_tokens=max_tokens, temperature=temperature)
        return response['choices'][0]['text']
    except Exception as e:
        print(f"Error during inference: {e}")
        utility.write_to_yaml('error_log', f"Inference error: {e}")
        return "Error: Could not generate a response."

# Function to read and format prompts
def read_and_format_prompt(file_name, data, task_name, syntax_type):
    syntax_type = utility.read_yaml().get(f'syntax_type_1', "{combined_input}")  # Always use chat syntax
    try:
        with open(file_name, "r") as file:
            lines = file.readlines()
        system_input = ""
        instruct_input = ""
        reading_system = False
        reading_instruct = False
        for line in lines:
            if "SYSTEM:" in line:
                reading_system = True
                reading_instruct = False
                continue
            elif "INSTRUCT:" in line:
                reading_system = False
                reading_instruct = True
                continue
            if reading_system:
                system_input += line.strip().format(**data) + " "
            elif reading_instruct:
                instruct_input += line.strip().format(**data) + " "
        
        # Use the provided syntax type for formatting
        formatted_prompt = syntax_type.format(combined_input=f"[INST] <<SYS>>\n{system_input}\n<</SYS>>\n{instruct_input}[/INST]")
        
        return formatted_prompt
    except FileNotFoundError:
        print(f"Error: {file_name} not found.")
        return None

# Function to log messages
def log_message(message, log_type, prompt_name=None, event_name=None, enable_logging=False):
    log_path = f'./data/{log_type}.log'
    if log_type == 'output' and not enable_logging:
        print("Logging is disabled!")
        return
    if os.path.exists(log_path):
        with open(log_path, 'a') as log_file:
            log_entry_name = prompt_name if prompt_name else 'processed_input'
            log_file.write(f"\n<-----------------------------{log_entry_name}_start--------------------------------->\n")
            log_file.write(message)
            log_file.write(f"\n<------------------------------{log_entry_name}_end---------------------------------->\n")
            if log_type == 'output':
                print(f"\n Logging {event_name}...")
                print(" ...Output logged.")
    else:
        print(f"File {log_path} not found. Logging failed.")         

# Function to parse the model's raw response
def parse_agent_response(raw_agent_response, data):
    print(" Parsing raw response...")
    cleaned_response = raw_agent_response.strip()
    cleaned_response = re.sub(r'^---\n*', '', cleaned_response, flags=re.MULTILINE)
    cleaned_response = re.sub(r'^\n+', '', cleaned_response, flags=re.MULTILINE)
    cleaned_response = re.sub(r"'\.'", '', cleaned_response, flags=re.MULTILINE)
    cleaned_response = re.sub(r'^### Solution:\n', '', cleaned_response, flags=re.MULTILINE)
    cleaned_response = re.sub(r'^### Summary:\n', '', cleaned_response, flags=re.MULTILINE)
    cleaned_response = re.sub(r'^### Response:\n', '', cleaned_response, flags=re.MULTILINE)
    cleaned_response = re.sub(r'^### Instruction:\n', '', cleaned_response, flags=re.MULTILINE)
    cleaned_response = re.sub(r'^### Example:\n', '', cleaned_response, flags=re.MULTILINE)
    cleaned_response = re.sub(r'^### Output:\n', '', cleaned_response, flags=re.MULTILINE)
    cleaned_response = re.sub(r'^### Example:\n', '', cleaned_response, flags=re.MULTILINE)
    cleaned_response = re.sub(r'^### Answer:\n', '', cleaned_response, flags=re.MULTILINE)
    cleaned_response = re.sub(r'^### Prompt Answer:\n', '', cleaned_response, flags=re.MULTILINE)
    cleaned_response = re.sub(r'^### Prompt Answer:\n', '', cleaned_response, flags=re.MULTILINE)
    cleaned_response = re.sub(r'^Please make sure.*\n?', '', cleaned_response, flags=re.MULTILINE)
    cleaned_response = re.sub(r'^(Sure, here\'s|Sure! Here is|Sure! Here\'s|Sure! here is).*\n?', '', cleaned_response, flags=re.MULTILINE)
    agent_name = data.get('agent_name', '')  
    cleaned_response = re.sub(rf'^### {agent_name}\n', '', cleaned_response, flags=re.MULTILINE)
    return cleaned_response

def prompt_response(task_name, rotation_counter, enable_logging=False, save_to=None):
    """
    Generates a response based on the task name and updates session variables.
    """
    data = utility.read_yaml()
    if data is None:
        return {"error": "Could not read config file."}

    # Retrieve prompt-specific settings
    prompt_settings = PROMPT_TO_SETTINGS.get(task_name, {})
    temperature = prompt_settings.get('temperature', 0.7)
    repeat_penalty = prompt_settings.get('repeat_penalty', 1.1)
    max_tokens = prompt_settings.get('max_tokens', 2000)

    prompt_file = f"./data/prompts/{task_name}.txt"
    formatted_prompt = read_and_format_prompt(prompt_file, data, task_name, None)
    if not os.path.exists(prompt_file) or formatted_prompt is None:
        return {"error": f"Prompt file {prompt_file} not found or failed to format."}

    raw_agent_response = run_llama_cli(formatted_prompt, max_tokens, temperature)

    if enable_logging:
        log_message(formatted_prompt, 'input', task_name, f"event {rotation_counter}", enable_logging)
        log_message(raw_agent_response, 'output', task_name, f"event {rotation_counter}", enable_logging)

    parsed_response = parse_agent_response(raw_agent_response, data)
    if save_to:
        utility.write_to_yaml(save_to, parsed_response)

    if task_name == 'consolidate':
        utility.write_to_yaml('session_history', parsed_response)

    return {'agent_response': parsed_response}

