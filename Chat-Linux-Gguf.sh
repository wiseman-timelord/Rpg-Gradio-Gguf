#!/bin/bash

# Define paths and files
PERSISTENT_FILE="./data/persistent.yaml"
VENV_PATH="./venv"
REQUIREMENTS_FILE="./data/requirements.txt"
HARDWARE_FILE="./data/hardware_details.txt"

# Global list of directories to create
FOLDERS_TO_CREATE=(
    "./models"
    "./models/text"
    "./models/image"
    "./generated"
)

# Function to check if running as root
check_sudo() {
    if [[ $EUID -ne 0 ]]; then
        echo "Error: Sudo Authorization Required!"
        sleep 3
        exit 1  # Directly exit
    else
        echo "Sudo authorization confirmed."
        sleep 1
    fi
}

# Function to create all necessary directories
create_folders() {
    for folder in "${FOLDERS_TO_CREATE[@]}"; do
        if [ ! -d "$folder" ]; then
            echo "Creating $folder directory..."
            mkdir -p "$folder"
            chmod 777 "$folder"
            echo "$folder directory created successfully."
        else
            echo "$folder directory already exists."
        fi
    done
}

# Function to create __init__.py in ./data
create_data_init_py() {
    local INIT_FILE="./data/__init__.py"
    echo "Creating or overwriting __init__.py in ./data"
    cat > "$INIT_FILE" <<EOL
# This file is auto-generated to mark this directory as a Python package.
EOL
    chmod 777 "$INIT_FILE"
    echo "__init__.py created successfully in ./data."
}

# Function to create requirements.txt in ./data
create_data_requirements() {
    local DATA_REQUIREMENTS_FILE="./data/requirements.txt"
    echo "Creating ./data/requirements.txt"
    cat > "$DATA_REQUIREMENTS_FILE" <<EOF
llama-cpp-python
gradio
watchdog
PyYAML
gguf-parser
stable-diffusion-cpp-python
EOF
    chmod 777 "$DATA_REQUIREMENTS_FILE"
    echo "./data/requirements.txt created successfully."
}

# Function to create hardware_details.txt in ./data
create_hardware_details() {
    local HARDWARE_FILE="./data/hardware_details.txt"
    echo "Detecting hardware information and creating hardware_details.txt in ./data"

    # Start writing simplified hardware details
    {
        echo -n "CPU Name : "
        grep -m 1 'model name' /proc/cpuinfo | awk -F: '{print $2}' | sed 's/^ //'

        echo -n "CPU Threads Total: "
        lscpu | grep '^CPU(s):' | awk '{print $2}'

        echo -n "Total System Ram: "
        free -h | grep 'Mem:' | awk '{print $2}'
    } > "$HARDWARE_FILE"

    chmod 777 "$HARDWARE_FILE"
    echo "Hardware details saved successfully to ./data/hardware_details.txt."
}

# Function to create persistent.yaml in ./data
create_persistent_yaml() {
    local PERSISTENT_FILE="./data/persistent.yaml"
    echo "Creating or overwriting persistent.yaml in ./data"
    cat > "$PERSISTENT_FILE" <<EOL
agent_name: Wise-Llama
agent_role: A wise oracle of sorts
human_name: Human
scene_location: the side of a mountain
selected_sample_method: euler_a
selected_steps: 2
session_history: The conversation started
threads_percent: 80
EOL
    chmod 777 "$PERSISTENT_FILE"
    echo "persistent.yaml created successfully in ./data."
}

# Function to create temporary.py in ./data
create_temporary_py() {
    local TEMPORARY_FILE="./data/temporary.py"
    echo "Creating or overwriting temporary.py in ./data"
    cat > "$TEMPORARY_FILE" <<EOL
# ./data/temporary.py

# Program variables
rotation_counter = 0
optimal_threads = 4
loaded_models = {}
large_language_model = None
model_used = False
stable_diffusion_model = None
latest_image_path = None

# Roleplay Settings
agent_name = "Wise-Llama"
agent_role = "A wise oracle of sorts"
human_name = "Human"
threads_percent = 80
agent_output = ""
human_input = ""
session_history = "the conversation started"
scene_location = "the side of a mountain"

# Image Size Options
IMAGE_SIZE_OPTIONS = {
    'available_sizes': ["192x256", "384x512", "512x768", "768x1024"],  # Updated to include only desired sizes
    'selected_size': "384x512"  # Default size as a string
}

# Steps Options
STEPS_OPTIONS = [1, 2, 4]  # Available step options
selected_steps = 2         # Default step

# Sample Method Options
SAMPLE_METHOD_OPTIONS = [
    ("EULER_A", "euler_a"),
    ("EULER", "euler"),
    ("HEUN", "heun"),
    ("DPM2", "dpm2"),
    ("DPMPP2S_A", "dpmpp2s_a"),
    ("DPMPP2M", "dpmpp2m"),
    ("DPMPP2Mv2", "dpmpp2mv2"),
    ("IPNDM", "ipndm"),
    ("IPNDM_V", "ipndm_v"),
    ("LCM", "lcm")
]
selected_sample_method = "heun"  # Default sample method, e.g., HEUN

SYNTAX_OPTIONS = [
    "{combined_input}",
    "User: {combined_input}",
    "User:\n{combined_input}"
]

PROMPT_TO_SETTINGS = {
    'converse': {
        'temperature': 0.5,
        'repeat_penalty': 1.2,
        'max_tokens': 250
    },
    'consolidate': {
        'temperature': 0.1,
        'repeat_penalty': 1.1,
        'max_tokens': 500
    }
}

EOL
    chmod 777 "$TEMPORARY_FILE"
    echo "temporary.py created successfully in ./data."
}

# Function to install Python requirements
install_requirements() {
    local DATA_REQUIREMENTS_FILE="./data/requirements.txt"
    
    echo "Installing Python libraries from $DATA_REQUIREMENTS_FILE to the virtual environment..."
    if [ -f "$DATA_REQUIREMENTS_FILE" ]; then
        "$VENV_PATH/bin/python3" -m pip install --upgrade pip
        "$VENV_PATH/bin/python3" -m pip install -r "$DATA_REQUIREMENTS_FILE"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install Python libraries from $DATA_REQUIREMENTS_FILE. Exiting..."
            End_Of_Script
        fi
        echo "Python libraries installed successfully."
    else
        echo "Error: $DATA_REQUIREMENTS_FILE not found. Exiting..."
        End_Of_Script
    fi
}

# Installer function
run_installer() {
    clear
    echo "================================================================================"
    echo "    Chat-Linux-Gguf - Installer"
    echo "================================================================================"
    echo ""
    echo "Running the Setup-Installer..."
    sleep 1

    # Create all necessary directories
    create_folders
    sleep 1

    # Create persistent.yaml
    create_persistent_yaml
    sleep 1

    # Create temporary.py in ./data
    create_temporary_py
    sleep 1

    # Create __init__.py in ./data
    create_data_init_py
    sleep 1

    # Create requirements.txt in ./data
    create_data_requirements
    sleep 1

    # Virtual environment check and creation
    if [ ! -d "$VENV_PATH" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "$VENV_PATH"
        if [ ! -f "$VENV_PATH/bin/pip" ]; then
            echo "Error: Virtual environment creation failed. Exiting..."
            End_Of_Script
        fi
        echo "Virtual environment created."
    else
        echo "Virtual environment already exists."
    fi
    sleep 1

    # Install Python libraries from ./data/requirements.txt
    install_requirements
    sleep 1

    # Create hardware details file
    create_hardware_details
    sleep 1

    echo "Setup-Installer processes have been completed."
    sleep 2
}

# Launch function
# Launch function
launch_program() {
    clear
    echo "================================================================================"
    echo "    Chat-Linux-Gguf - Launcher"
    echo "================================================================================"
    echo ""
    echo "Preparing to launch the main program..."

    echo "Verifying virtual environment..."
    if [ ! -d "$VENV_PATH" ]; then
        echo "Error: Virtual environment not found. Please run the setup-installer first."
        sleep 3
        End_Of_Script
    else
        echo "Virtual environment verified."
        sleep 1
    fi

    echo "Activating virtual environment and preparing to launch the main program..."
    source "$VENV_PATH/bin/activate"
    echo "Environment activated."
    sleep 1

    echo "Launching the main Python script..."
    sleep 2
    "$VENV_PATH/bin/python3" main_script.py --gui  # Corrected to use the Python from the virtual environment
    if [ $? -ne 0 ]; then
        echo "Main program exited. Shutting down..."
        End_Of_Script
    fi

    echo "Main program launched successfully."
    sleep 1
}



# Function to gracefully end the script
End_Of_Script() {
    echo "Performing cleanup operations..."
    
    # Deactivate the virtual environment if active
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo "Deactivating virtual environment..."
        deactivate
    fi

    # Notify user and exit gracefully
    echo "All cleanup operations complete. Exiting in 3 seconds..."
    sleep 3
    exit 0
}

# Check for sudo at the start of the script
check_sudo

# Menu system
while true; do
    clear
    echo "================================================================================"
    echo "    Chat-Linux-Gguf - Bash Menu"
    echo "================================================================================"
    echo ""
    echo ""
    echo "" 
    echo "" 
    echo "" 
    echo ""
    echo "    1. Launch Main Program"
    echo ""
    echo "    2. Run Setup-Installer"
    echo ""
    echo "" 
    echo "" 
    echo ""
    echo "" 
    echo ""
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo -n "Selection; Menu Options = 1-2, Exit Program = X: "
    read -r choice
    case "$choice" in
        1) launch_program ;;
        2) run_installer ;;
        X|x) End_Of_Script ;;
        *) echo "Invalid option, try again." ;;
    esac
    # Pause briefly before refreshing the menu
    sleep 2
done

