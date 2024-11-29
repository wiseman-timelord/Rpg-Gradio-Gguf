#!/bin/bash

# Define paths and files
TMPFS_DIR="/mnt/ramfs"
PERSISTENT_FILE="$TMPFS_DIR/persistent.yaml"
VENV_PATH="/absolute/path/to/venv"  # Modify this path to the correct location

# Function to check if running as root
check_sudo() {
    if [[ $EUID -ne 0 ]]; then
        echo "Error: Sudo Authorization Required!"
        sleep 3
        exit 1
    else
        echo "Sudo authorization confirmed."
        sleep 1
    fi
}

# Function to gracefully exit the script
exit_script() {
    echo "Exiting Launcher-Installer..."
    # Check if virtual environment is activated and deactivate
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo "Deactivating virtual environment..."
        deactivate
    fi
    sleep 3
    exit 0
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
        echo "{}" > "$PERSISTENT_FILE"
        sudo chmod 666 "$PERSISTENT_FILE"
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

    # Virtual environment check and installation
    if [ ! -d "$VENV_PATH" ]; then
        echo "Creating virtual environment..."
        python3 -m venv $VENV_PATH
        echo "Virtual environment created."
    else
        echo "Virtual environment already exists."
    fi
    sleep 1

    # Activate virtual environment
    echo "Activating virtual environment..."
    source $VENV_PATH/bin/activate
    echo "Virtual environment activated."
    sleep 1

    echo "Updating system..."
    sudo apt update
    echo "System updated."

    echo "Installing dos2unix..."
    sudo apt install -y dos2unix
    echo "dos2unix installed."
    sleep 1

    echo "Installing OpenCL and Vulkan drivers..."
    sudo apt install -y mesa-opencl-icd vulkan-tools
    echo "Drivers installed."
    sleep 1

    echo "Installing Python libraries from requirements.txt..."
    pip install -r ./data/requirements.txt
    echo "Python libraries installed."
    sleep 1

    echo "Setup-Installer processes have been completed."
    sleep 5
    echo "Returning to the main menu..."
}

# Launch function
launch_program() {
    echo "Preparing to launch the main program..."
    mount_tmpfs

    echo "Verifying virtual environment..."
    if [ ! -d "$VENV_PATH" ]; then
        echo "Error: Virtual environment not found. Please run the setup-installer first."
        sleep 3
        return
    else
        echo "Virtual environment verified."
        sleep 1
    fi

    echo "Activating virtual environment and preparing to launch the main program..."
    source $VENV_PATH/bin/activate
    echo "Environment activated."
    sleep 1

    echo "Launching the main Python script..."
    sleep 5
    python "$VENV_PATH/bin/python" "main.py"  # Adjust this to your specific Python script

    echo "Main program launched successfully."
    sleep 1
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
        X|x) exit_script ;;
        *) echo "Invalid option, try again." ;;
    esac
done

