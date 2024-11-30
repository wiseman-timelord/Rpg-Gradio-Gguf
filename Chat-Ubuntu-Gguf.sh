#!/bin/bash

# Define paths and files
TMPFS_DIR="/mnt/ramfs"
PERSISTENT_FILE="$TMPFS_DIR/persistent.yaml"
VENV_PATH="$(pwd)/venv"
REQUIREMENTS_FILE="./data/requirements.txt"

# Function to check if running as root
check_sudo() {
    if [[ $EUID -ne 0 ]]; then
        echo "Error: Sudo Authorization Required!"
        sleep 3
        End_Of_Script
    else
        echo "Sudo authorization confirmed."
        sleep 1
    fi
}

# Functions to mount and unmount tmpfs
mount_tmpfs() {
    if ! mountpoint -q "$TMPFS_DIR"; then
        sudo mkdir -p "$TMPFS_DIR"
        sudo mount -t tmpfs -o size=100M tmpfs "$TMPFS_DIR"
        sudo chmod 777 "$TMPFS_DIR"
        echo "TMPFS mounted at $TMPFS_DIR"
    else
        echo "TMPFS is already mounted at $TMPFS_DIR"
    fi

    if [[ ! -f "$PERSISTENT_FILE" ]]; then
        echo "Creating persistent.yaml in $TMPFS_DIR"
        cat > "$PERSISTENT_FILE" <<EOL
# DEFAULT_CONFIG:
human_name: "Human"
agent_name: "Wise-Llama"
agent_role: "Wise Oracle"
EOL
        sudo chmod 666 "$PERSISTENT_FILE"
        echo "Persistent.yaml created with default configuration."
    fi
}

unmount_tmpfs() {
    if mountpoint -q "$TMPFS_DIR"; then
        sudo umount "$TMPFS_DIR"
        sudo rmdir "$TMPFS_DIR"
        echo "TMPFS unmounted from $TMPFS_DIR"
    else
        echo "TMPFS is not mounted at $TMPFS_DIR"
    fi
}

# Installer function
run_installer() {
    echo "Running the Setup-Installer..."

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
    sleep 3
}

# Launch function
launch_program() {
    echo "Preparing to launch the main program..."
    mount_tmpfs

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
    
    # Unmount tmpfs if mounted
    echo "Checking for mounted TMPFS..."
    if mountpoint -q "$TMPFS_DIR"; then
        echo "Unmounting TMPFS..."
        unmount_tmpfs
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
    echo "    1. Launch Main Program"
    echo "    2. Run Setup-Installer"
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

