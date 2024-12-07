# .\scripts\model.py

from llama_cpp import Llama
from scripts import utility
from gguf_parser import GGUFParser
import os
import re
import json
from data.temporary import MODE_TO_TEMPERATURE, PROMPT_TO_SETTINGS, large_language_model, model_used
from scripts.utility import calculate_optimal_threads, read_yaml

def process_selected_model(models_dir='./models'):
    global large_language_model, model_used
    print(f"Scanning {models_dir} for models...")
    gguf_files = [f for f in os.listdir(models_dir) if f.endswith(".gguf")]
    if not gguf_files:
        raise FileNotFoundError("No valid GGUF models found in the directory.")

    model_path = os.path.join(models_dir, gguf_files[0])
    print(f"Selected model: {model_path}")

    json_files = [f for f in os.listdir(models_dir) if f.endswith(".json")]
    config_path = os.path.join(models_dir, json_files[0]) if json_files else None

    metadata = {}
    if config_path:
        with open(config_path, 'r') as file:
            metadata.update(json.load(file))

    parser = GGUFParser(model_path)
    parser.parse()
    metadata.update(parser.metadata)

    n_ctx = metadata.get("llama.context_length", 2048)
    optimal_threads = calculate_optimal_threads(read_yaml().get('threads_percent', 80))
    
    print(f"Initializing model with context length: {n_ctx}, threads: {optimal_threads}")
    large_language_model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=optimal_threads,
        n_batch=512,
        use_mmap=False,
        use_mlock=False
    )
    model_used = True
    print("Model initialized successfully.")

def initialize_model(model_path, n_ctx, n_threads):
    print("Initializing model...")
    large_language_model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=512,
        use_mmap=False,
        use_mlock=False
    )
    model_used = True
    print("Model initialized successfully.")

def run_llama_cli(prompt, max_tokens=2000, temperature=0.7, repeat_penalty=1.1):
    global large_language_model, model_used
    if not model_used or large_language_model is None:
        print(f"Model used: {model_used}, Model instance: {large_language_model}")
        raise RuntimeError("Model is not initialized. Please run the setup-installer and ensure a model is loaded.")

    try:
        response = large_language_model(prompt, max_tokens=max_tokens, temperature=temperature, repeat_penalty=repeat_penalty)
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                generated_text = choices[0].get("text", "").strip()
                return generated_text
            else:
                return "No output from model."
        else:
            return "Unexpected response format from model."
    except Exception as e:
        print(f"Error during inference: {e}")
        return f"Error: Could not generate a response due to: {e}"


def read_and_format_prompt(file_name, data, task_name):
    """
    Reads and formats the prompt template from the given file_name using the data dictionary.
    """
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
            elif "INSTRUCTION:" in line:
                reading_system = False
                reading_instruct = True
                continue
            if reading_system:
                system_input += line.strip().format(**data) + " "
            elif reading_instruct:
                instruct_input += line.strip().format(**data) + " "

        # Use a consistent syntax. For now let's embed in a standard template:
        # The code below was simplified to always wrap system and instruction in [INST] tags as per original code.
        formatted_prompt = f"[INST] <<SYS>>\n{system_input}\n<</SYS>>\n{instruct_input}[/INST]"
        return formatted_prompt
    except FileNotFoundError:
        print(f"Error: {file_name} not found.")
        return None


def log_message(message, log_type, prompt_name=None, event_name=None, enable_logging=False):
    # As before, optional logging functionality
    pass


def parse_agent_response(raw_agent_response, data):
    # Cleans up raw_agent_response as before
    cleaned_response = raw_agent_response.strip()
    # ... (All the regex replacements as before)
    cleaned_response = re.sub(r'^### .*:\n', '', cleaned_response, flags=re.MULTILINE)
    return cleaned_response


def prompt_response(task_name, rotation_counter, enable_logging=False, save_to=None):
    data = utility.read_yaml()
    if data is None:
        return {"error": "Could not read config file."}

    prompt_settings = PROMPT_TO_SETTINGS.get(task_name, {'temperature':0.7, 'repeat_penalty':1.1, 'max_tokens':2000})
    temperature = prompt_settings.get('temperature', 0.7)
    repeat_penalty = prompt_settings.get('repeat_penalty', 1.1)
    max_tokens = prompt_settings.get('max_tokens', 2000)

    prompt_file = f"./prompts/{task_name}.txt"  # Ensure prompt files are in ./prompts/
    formatted_prompt = read_and_format_prompt(prompt_file, data, task_name)
    if not os.path.exists(prompt_file) or formatted_prompt is None:
        return {"error": f"Prompt file {prompt_file} not found or failed to format."}

    raw_agent_response = run_llama_cli(formatted_prompt, max_tokens, temperature, repeat_penalty)
    parsed_response = parse_agent_response(raw_agent_response, data)

    if save_to:
        utility.write_to_yaml(save_to, parsed_response)

    # If we are consolidating, update session_history
    if task_name == 'consolidate':
        utility.write_to_yaml('session_history', parsed_response)

    return {'agent_response': parsed_response}
