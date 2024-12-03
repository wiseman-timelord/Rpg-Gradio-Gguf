#!/bin/bash

# Define paths and files
PERSISTENT_FILE="./data/persistent.yaml"  # Updated location for persistent.yaml
VENV_PATH="$(pwd)/venv"
REQUIREMENTS_FILE="./requirements.txt"

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

# Function to create __init__.py in ./data
create_data_init_py() {
    INIT_FILE="./data/__init__.py"
    if [ ! -f "$INIT_FILE" ]; then
        echo "Creating __init__.py in ./data"
        touch "$INIT_FILE"
        echo "# This file is auto-generated to mark this directory as a Python package." > "$INIT_FILE"
        chmod 644 "$INIT_FILE"
        echo "__init__.py created successfully in ./data."
    else
        echo "__init__.py already exists in ./data."
    fi
}

# Function to create persistent.yaml
create_persistent_yaml() {
    if [[ ! -f "$PERSISTENT_FILE" ]]; then
        echo "Creating persistent.yaml in ./data"
        cat > "$PERSISTENT_FILE" <<EOL
# DEFAULT_CONFIG:
human_name: "Human"
agent_name: "Wise-Llama"
agent_role: "Wise Oracle"
EOL
        chmod 666 "$PERSISTENT_FILE"
        echo "persistent.yaml created with default configuration."
    else
        echo "persistent.yaml already exists in ./data."
    fi
}

# Function to create temporary.py in ./data
create_temporary_py() {
    TEMPORARY_FILE="./data/temporary.py"
    if [ ! -f "$TEMPORARY_FILE" ]; then
        echo "Creating temporary.py in ./data"
        cat > "$TEMPORARY_FILE" <<EOL
# Temporary variables for Chat-Ubuntu-Gguf

# General Variables
session_history = "the conversation started"  # Default: "the conversation started"
rotation_counter = 0

# Model Variables
loaded_models = {}
llm = None

# Configurable Keys
agent_name = "Empty"
agent_role = "Empty"
human_name = "Empty"

# Other Keys
agent_output = ""
human_input = ""

# Model Mapping
MODE_TO_TEMPERATURE = {
    'RolePlaying': 0.7,
    'TextProcessing': 0.1
}

PROMPT_TO_MAXTOKENS = {
    'converse': 2000,
    'consolidate': 1000
}

# Syntax Options
SYNTAX_OPTIONS_DISPLAY = [
    "{combined_input}",
    "User: {combined_input}",
    "User:\\n{combined_input}",
    "### Human: {combined_input}",
    "### Human:\\n{combined_input}",
    "### Instruction: {combined_input}",
    "### Instruction:\\n{combined_input}",
    "{system_input}. USER: {instruct_input}",
    "{system_input}\\nUser: {instruct_input}"
]
SYNTAX_OPTIONS = SYNTAX_OPTIONS_DISPLAY
EOL
        chmod 644 "$TEMPORARY_FILE"
        echo "temporary.py created successfully."
    else
        echo "temporary.py already exists in ./data."
    fi
}

# Installer function
run_installer() {
    echo "Running the Setup-Installer..."
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

    # Install Python libraries directly to the virtual environment
    echo "Installing Python libraries from $REQUIREMENTS_FILE to the virtual environment..."
    "$VENV_PATH/bin/python3" -m pip install --upgrade pip
    "$VENV_PATH/bin/python3" -m pip install -r "$REQUIREMENTS_FILE"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install Python libraries. Exiting..."
        End_Of_Script
    fi
    echo "Python libraries installed."
    sleep 1

    echo "Setup-Installer processes have been completed."
    sleep 2
}

# Launch function
launch_program() {
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
        echo "Error: Main program execution failed. Exiting..."
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
    echo "    Chat-Ubuntu-Gguf"
    echo "================================================================================"
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
    echo ""  "--------------------------------------------------------------------------------"
    echo -n "Selection; Menu Options = 1-2, Exit Program = X: "
    read -r choice
    case "$choice" in
        1) launch_program ;;
        2) run_installer ;;
        X|x) End_Of_Script ;;
        *) echo "Invalid option, try again." ;;
    esac
done

