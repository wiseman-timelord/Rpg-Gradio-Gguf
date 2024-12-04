# .\scripts\model.py

# imports
from llama_cpp import Llama
from scripts import utility
import os
import time
import re
import subprocess  # Corrected typo
from data.temporary import MODE_TO_TEMPERATURE, PROMPT_TO_MAXTOKENS


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
    Initializes a GGUF model using the configuration from model_config.json.
    """
    global model
    try:
        # Scan the models directory
        models = utility.scan_models_directory(models_dir)
        if not models:
            raise FileNotFoundError("No valid GGUF model and model_config.json found in the models directory.")
        
        # Select the first valid model (adjust if multiple models need handling)
        selected_model = models[0]
        model_path = selected_model['model_path']
        config_path = selected_model['config_path']

        # Load configuration
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)

        # Extract and override relevant parameters
        n_ctx = config["load_params"].get("n_ctx", 8192)
        n_batch = config["load_params"].get("n_batch", 512)
        use_mmap = config["load_params"].get("use_mmap", True)

        print(f"Initializing model: {model_path} with context: {n_ctx}, batch: {n_batch}")
        model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            use_mmap=use_mmap,
            use_mlock=False  # Override mlock to False
        )
        print("Model initialized successfully.")
        utility.write_to_yaml('model_path', model_path)
        utility.write_to_yaml('n_ctx', n_ctx)
        utility.write_to_yaml('n_batch', n_batch)

    except Exception as e:
        print(f"Error during model initialization: {e}")
        utility.write_to_yaml('error_log', f"Model initialization error: {e}")
        model = None


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

def prompt_response(task_name, rotation_counter, enable_logging=False, save_to=None, loaded_models=None):
    data = utility.read_yaml()
    if data is None:
        return {"error": "Could not read config file."}

    mode = 'RolePlaying' if task_name == 'converse' else 'TextProcessing'
    temperature = MODE_TO_TEMPERATURE[mode]
    
    prompt_file = f"./data/prompts/{task_name}.txt"
    formatted_prompt = read_and_format_prompt(prompt_file, data, task_name, None)  # Removed syntax_type argument
    if not os.path.exists(prompt_file) or formatted_prompt is None:
        return {"error": f"Prompt file {prompt_file} not found or failed to format."}

    max_tokens = PROMPT_TO_MAXTOKENS.get(task_name, 2000)

    raw_agent_response = run_llama_cli(formatted_prompt, max_tokens, temperature)

    if enable_logging:
        log_entry_name = f"{task_name}_response"
        log_message(formatted_prompt, 'input', log_entry_name, f"event {rotation_counter}", enable_logging)
        log_message(raw_agent_response, 'output', log_entry_name, f"event {rotation_counter}", enable_logging)

    parsed_response = parse_agent_response(raw_agent_response, data)
    if save_to:
        utility.write_to_yaml(save_to, parsed_response)

    new_session_history = None
    new_emotion = None
    if task_name == 'consolidate':
        new_session_history = parsed_response
        utility.write_to_yaml('session_history', new_session_history)
    elif task_name == 'emotions':
        emotion_keywords = ["Love", "Arousal", "Euphoria", "Surprise", "Curiosity", "Indifference", "Fatigue", "Discomfort", "Embarrassment", "Anxiety", "Stress", "Anger", "Hate"]
        found_emotions = [word for word in emotion_keywords if re.search(rf"\b{word}\b", parsed_response, re.IGNORECASE)]
        new_emotion = ", ".join(found_emotions)
        utility.write_to_yaml('agent_emotion', new_emotion)

    return {
        'agent_response': parsed_response,
        'new_session_history': new_session_history,
        'new_emotion': new_emotion
    }

# Define MODE_TO_TEMPERATURE and PROMPT_TO_MAXTOKENS dictionaries
MODE_TO_TEMPERATURE = {
    'RolePlaying': 0.7,
    'TextProcessing': 0.1
}

PROMPT_TO_MAXTOKENS = {
    'converse': 2000,
    'emotions': 500,
    'consolidate': 1000
}
