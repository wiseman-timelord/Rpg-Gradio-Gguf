# scripts/model.py

from llama_cpp import Llama
from scripts import utility
from gguf_parser import GGUFParser
import os
import re
import json
from data.temporary import PROMPT_TO_SETTINGS, large_language_model, model_used
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
    global large_language_model, model_used
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
            template = file.read()

        # Replace placeholders with values from the data dictionary
        formatted_prompt = template.format(**data)
        return formatted_prompt
    except FileNotFoundError:
        print(f"Error: {file_name} not found.")
        return None
    except KeyError as e:
        print(f"Error: Missing key {e} in data for prompt formatting.")
        return None

def log_message(message, log_type, prompt_name=None, event_name=None, enable_logging=False):
    # Optional logging functionality can be implemented here
    pass

def parse_agent_response(raw_agent_response, data):
    # Cleans up raw_agent_response as before
    cleaned_response = raw_agent_response.strip()
    # Example: Remove any unwanted prefixes or formatting
    cleaned_response = re.sub(r'^### .*:\n', '', cleaned_response, flags=re.MULTILINE)
    return cleaned_response

def prompt_response(task_name, rotation_counter, enable_logging=False, save_to=None):
    from data import temporary  # Import here to avoid circular imports

    # Build runtime data
    runtime_data = {
        'agent_name': temporary.agent_name or "Agent",
        'agent_role': temporary.agent_role or "Assistant",
        'human_name': temporary.human_name or "User",
        'session_history': temporary.session_history or "No prior conversation.",
        'human_input': temporary.human_input or "No input provided.",
        'agent_output': temporary.agent_output or "No response generated."
    }

    prompt_settings = PROMPT_TO_SETTINGS.get(task_name)
    if not prompt_settings:
        print(f"[ERROR] No settings found for task '{task_name}'.")
        return {"error": f"No settings found for task '{task_name}'."}

    temperature = prompt_settings.get('temperature', 0.7)
    repeat_penalty = prompt_settings.get('repeat_penalty', 1.1)
    max_tokens = prompt_settings.get('max_tokens', 2000)

    prompt_file = f"./prompts/{task_name}.txt"

    formatted_prompt = read_and_format_prompt(prompt_file, runtime_data, task_name)
    if formatted_prompt is None:
        print(f"[ERROR] Prompt formatting failed for '{task_name}'.")
        return {"error": f"Prompt file {prompt_file} not found or failed to format."}

    # Print formatted prompt just before sending to the model
    print(f"[INFO] Sending prompt ({task_name}) to model:\n{formatted_prompt}\n")

    raw_agent_response = run_llama_cli(formatted_prompt, max_tokens, temperature, repeat_penalty)

    # Print raw model response
    print(f"[INFO] Raw model response for task '{task_name}':\n{raw_agent_response}\n")

    if "Error:" in raw_agent_response:
        print(f"[ERROR] Model returned an error for '{task_name}': {raw_agent_response}")
        return {"error": raw_agent_response}

    parsed_response = parse_agent_response(raw_agent_response, runtime_data)

    # Take only the first line of the response
    parsed_response = parsed_response.split('\n', 1)[0].strip()

    if save_to:
        utility.write_to_yaml(save_to, parsed_response)

    if task_name == 'consolidate':
        temporary.session_history += f"\n{parsed_response}"

    return {'agent_response': parsed_response}
