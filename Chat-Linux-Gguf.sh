#!/bin/bash

# Define paths and files
PERSISTENT_FILE="./data/persistent.yaml"
VENV_PATH="$(pwd)/venv"
REQUIREMENTS_FILE="./data/requirements.txt"

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

# Ensure the ./data directory exists
ensure_data_directory() {
    if [ ! -d "./data" ]; then
        echo "Creating ./data directory..."
        mkdir ./data
        chmod 777 ./data
        echo "./data directory created successfully."
    else
        echo "./data directory already exists."
    fi
}

# Function to create __init__.py in ./data
create_data_init_py() {
    ensure_data_directory  # Ensure the directory exists
    local INIT_FILE="./data/__init__.py"
    echo "Creating or overwriting __init__.py in ./data"
    cat > "$INIT_FILE" <<EOL
# This file is auto-generated to mark this directory as a Python package.
EOL
    chmod 777 "$INIT_FILE"
    echo "__init__.py created successfully in ./data."
}

create_data_requirements() {
    ensure_data_directory  # Ensure the ./data directory exists
    local DATA_REQUIREMENTS_FILE="./data/requirements.txt"
    
    # Copy requirements.txt to ./data/requirements.txt
    echo "Copying requirements.txt to ./data/requirements.txt"
    if [ -f "./requirements.txt" ]; then
        cp ./requirements.txt "$DATA_REQUIREMENTS_FILE"
        chmod 777 "$DATA_REQUIREMENTS_FILE"
        echo "requirements.txt copied successfully to ./data."
    else
        echo "Error: requirements.txt not found in the current directory. Exiting..."
        exit 1
    fi
}


# Function to create persistent.yaml
create_persistent_yaml() {
    ensure_data_directory  # Ensure the directory exists
    local PERSISTENT_FILE="./data/persistent.yaml"
    echo "Creating or overwriting persistent.yaml in ./data"
    cat > "$PERSISTENT_FILE" <<EOL
# ./data/persistent.yaml - default session config:
human_name: "Human"
agent_name: "Wise-Llama"
agent_role: "A wise oracle of sorts"
session_history = "The conversation started"
threads_percent: 80
EOL
    chmod 777 "$PERSISTENT_FILE"
    echo "persistent.yaml created successfully in ./data."
}

# Function to create temporary.py in ./data
create_temporary_py() {
    ensure_data_directory  # Ensure the directory exists
    local TEMPORARY_FILE="./data/temporary.py"
    echo "Creating or overwriting temporary.py in ./data"
    cat > "$TEMPORARY_FILE" <<EOL
# Temporary variables for Chat-Linux-Gguf

# General Variables
session_history = "the conversation started"
rotation_counter = 0
optimal_threads = 4

# Model Variables
loaded_models = {}
large_language_model = None  # Renamed model instance
model_used = False           # Flag to track if the model is used

# Configurable Keys
agent_name = "Wise-Llama"
agent_role = "A wise oracle of sorts"
human_name = "Human"
session_history = "the conversation started"
threads_percent = 80

# Other Keys
agent_output = ""
human_input = ""

# Syntax Options
SYNTAX_OPTIONS = [
    "{combined_input}",
    "User: {combined_input}",
    "User:\n{combined_input}"
]

# Consolidated Prompt Settings
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

    # Ensure ./logs directory exists
    if [ ! -d "./logs" ]; then
        echo "Creating ./logs directory..."
        mkdir ./logs
        chmod 777 ./logs
        echo "./logs directory created."
    else
        echo "./logs directory already exists."
    fi
    sleep 1

    # Ensure ./models directory exists
    if [ ! -d "./models" ]; then
        echo "Creating ./models directory..."
        mkdir ./models
        chmod 777 ./models
        echo "./models directory created."
    else
        echo "./models directory already exists."
    fi
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

    echo "Setup-Installer processes have been completed."
    sleep 2
}



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
    python3 main_script.py --gui
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
done

