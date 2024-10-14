#!/bin/bash

# Check for admin privileges
if [[ $EUID -ne 0 ]]; then
    echo "This script requires administrative privileges."
    echo "Please run this script as root or using sudo."
    exit 1
fi
echo "Running with Admin rights."

# Initialization Block
ScriptDirectory=$(dirname "$(readlink -f "$0")")
pushd "$ScriptDirectory" > /dev/null

# Custom Banner
echo "*******************************************************************************************************************"
echo "    Chat-UbuntuLLama - Launcher"
echo "*******************************************************************************************************************"
echo
echo "Working Dir: $ScriptDirectory"
echo

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Launch the Python script
echo "Launching the Python script..."
python3 window_1.py

# Deactivate the virtual environment
echo "Deactivating the virtual environment..."
deactivate

echo
echo "Launcher Process Completed Normally."
exit 0
