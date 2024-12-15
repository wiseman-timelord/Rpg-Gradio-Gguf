# scripts/model.py

from llama_cpp import Llama
from stable_diffusion_cpp import StableDiffusion
from scripts import utility
from gguf_parser import GGUFParser
import os, re, json, uuid
from data.temporary import PROMPT_TO_SETTINGS, large_language_model, model_used, IMAGE_SIZE_OPTIONS
from data import temporary
from scripts.utility import calculate_optimal_threads, read_yaml

def process_selected_model():
    """
    Scans the ./models/text and ./models/image directories to load the first available GGUF models.
    Initializes both the text and image models and assigns them to temporary globals.
    """
    # Load text model
    text_model_dir = './models/text'
    image_model_dir = './models/image'

    print(f"Scanning {text_model_dir} for GGUF text models...")
    text_gguf_files = [f for f in os.listdir(text_model_dir) if f.endswith(".gguf")]
    if not text_gguf_files:
        raise FileNotFoundError("No valid GGUF text models found in the directory.")
    text_model_path = os.path.join(text_model_dir, text_gguf_files[0])
    print(f"Selected text model: {text_model_path}")

    # Optional config JSON for text model
    json_files = [f for f in os.listdir(text_model_dir) if f.endswith(".json")]
    config_path = os.path.join(text_model_dir, json_files[0]) if json_files else None

    metadata = {}
    if config_path:
        with open(config_path, 'r') as file:
            metadata.update(json.load(file))

    parser = GGUFParser(text_model_path)
    parser.parse()
    metadata.update(parser.metadata)
    n_ctx = metadata.get("llama.context_length", 2048)
    threads_percent = read_yaml().get('threads_percent', 80)
    optimal_threads = calculate_optimal_threads(threads_percent)

    print(f"Initializing text model with context length: {n_ctx}, threads: {optimal_threads}")
    temporary.large_language_model = Llama(
        model_path=text_model_path,
        n_ctx=n_ctx,
        n_threads=optimal_threads,
        n_batch=512,
        use_mmap=False,
        use_mlock=False
    )
    temporary.model_used = True
    print("Text model initialized successfully.")

    # Load image model
    print(f"Scanning {image_model_dir} for Stable Diffusion GGUF models...")
    # Assuming that stable_diffusion_cpp supports GGUF models
    image_models = [f for f in os.listdir(image_model_dir) if f.endswith(".gguf")]
    if not image_models:
        print("No GGUF image models found, Stable Diffusion unavailable.")
        temporary.stable_diffusion_model = None
    else:
        image_model_path = os.path.join(image_model_dir, image_models[0])
        print(f"Selected image model: {image_model_path}")
        temporary.stable_diffusion_model = StableDiffusion(
            model_path=image_model_path,
            wtype="default"  # Weight type; 'default' auto-detects based on model file
        )
        print("Image model initialized successfully.")

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
    if not temporary.model_used or temporary.large_language_model is None:
        print(f"Model used: {temporary.model_used}, Model instance: {temporary.large_language_model}")
        raise RuntimeError("Model is not initialized.")

    try:
        response = temporary.large_language_model(prompt, max_tokens=max_tokens, temperature=temperature, repeat_penalty=repeat_penalty)
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

        # Escape any unescaped curly braces
        template = template.replace('{', '{{').replace('}', '}}').replace('{{{{', '{').replace('}}}}', '}')

        # Replace placeholders with values from the data dictionary
        formatted_prompt = template.format(**data)
        return formatted_prompt
    except FileNotFoundError:
        print(f"Error: {file_name} not found.")
        return None
    except KeyError as e:
        print(f"Error: Missing key {e} in data for prompt formatting.")
        return None
    except ValueError as ve:
        print(f"Error: Formatting issue in template file {file_name}: {ve}")
        return None


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

    # Ensure that the response does not contain image size information
    # This assumes that the agent does not send back 'size: ...' in its response
    # If it does, further processing may be needed
    parsed_response = parsed_response.split('\n', 1)[0].strip()

    if save_to:
        utility.write_to_yaml(save_to, parsed_response)

    if task_name == 'consolidate':
        # Ensure that only conversation text is appended without image size
        temporary.session_history += f"\n{parsed_response}"

    return {'agent_response': parsed_response}


def generate_image_from_history(session_history_prompt):
    """
    Generates an image based on the session_history prompt using stable_diffusion_model.
    Returns the path to the generated image.
    """
    if temporary.stable_diffusion_model is None:
        print("Stable Diffusion model not loaded, skipping image generation.")
        return None

    # Retrieve the selected image size from temporary.py
    selected_size_str = temporary.IMAGE_SIZE_OPTIONS.get('selected_size', "64x128")
    if selected_size_str not in temporary.IMAGE_SIZE_OPTIONS['available_sizes']:
        print(f"Invalid image size selected: {selected_size_str}. Defaulting to 64x128.")
        selected_size_str = "64x128"

    try:
        # Ensure only one 'x' is present
        if selected_size_str.count('x') == 1:
            width, height = map(int, selected_size_str.split('x'))
        else:
            # Handle cases where size might be appended multiple times
            parts = selected_size_str.split('x')
            width, height = map(int, parts[:2])
            print(f"Warning: Image size string '{selected_size_str}' has multiple 'x'. Using first two values: {width}x{height}")
    except ValueError:
        print(f"Error parsing image size '{selected_size_str}'. Defaulting to 64x128.")
        width, height = 64, 128

    # Retrieve the selected number of steps from temporary.py
    selected_steps = temporary.selected_steps
    if selected_steps not in temporary.STEPS_OPTIONS:
        print(f"Invalid number of steps selected: {selected_steps}. Defaulting to 2 steps.")
        selected_steps = 2

    # Retrieve the selected sample method from temporary.py
    selected_sample_method = temporary.selected_sample_method  # string

    # Construct a suitable prompt from session history
    # For example, we use the last line of the session_history as the image prompt
    prompt = session_history_prompt.strip().split('\n')[-1]
    # Remove 'size: ' prefix if present
    if prompt.lower().startswith('size:'):
        prompt = prompt.split(':', 1)[1].strip()

    # Incorporate scene_location into the prompt if necessary
    # Assuming that session_history already includes scene_location from the converse prompt

    if not prompt:
        prompt = "A generic illustrative scene."

    print(f"Generating image with prompt: '{prompt}', size: {width}x{height}, steps: {selected_steps}, sample_method: {selected_sample_method}")

    try:
        output_images = temporary.stable_diffusion_model.txt_to_img(
            prompt=prompt,
            width=width,                     # Set width based on selected size
            height=height,                   # Set height based on selected size
            sample_steps=selected_steps,     # Set steps based on user selection
            sample_method=selected_sample_method  # Set sample method as string
        )
    except TypeError as te:
        print(f"TypeError during image generation: {te}")
        return None
    except Exception as e:
        print(f"Error during image generation: {e}")
        return None

    if not os.path.exists('./generated'):
        os.makedirs('./generated', exist_ok=True)
    image_filename = f"{uuid.uuid4().hex}.png"
    image_path = os.path.join('./generated', image_filename)
    try:
        output_images[0].save(image_path)
        print(f"Image generated and saved to {image_path}")
        return image_path
    except Exception as e:
        print(f"Error saving generated image: {e}")
        return None

