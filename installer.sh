#!/bin/bash

# Check for admin privileges
if [[ $EUID -ne 0 ]]; then
    echo "This script requires administrative privileges."
    echo "Please run this script as root or using sudo."
    exit 1
fi
echo "Running with Admin rights."

# Globals
PYTHON_EXE_TO_USE=$(which python3)
PIP_EXE_TO_USE=$(which pip3)
REQUIRED_LIBRARIES=(
    "llama-cpp-python--"
    "plotly--"
    "watchdog--"
    "pyyaml--"
)

# Initialization Block
ScriptDirectory=$(dirname "$(readlink -f "$0")")
pushd "$ScriptDirectory" > /dev/null
if [[ -z "$PYTHON_EXE_TO_USE" ]]; then
    echo "Error: Python 3 not found. Please ensure it is installed."
    exit 1
fi

# Custom Banner
echo "*******************************************************************************************************************"
echo "                                          Chat-UbuntuLLama - Installer"
echo "*******************************************************************************************************************"
echo
echo "Working Dir: $ScriptDirectory"
echo

# Create Directories
echo "Creating necessary directories..."
mkdir -p ./data/libraries ./data/cache ./models

# Folder Maintenance
echo "Emptying ./data/libraries..."
rm -rf ./data/libraries/*

# Function to convert `--` to `==` for pip installation
convert_to_pip_format() {
    local library_list=("$@")
    for lib in "${library_list[@]}"; do
        echo "${lib//--/==}"
    done
}

# Install dos2unix if necessary
if ! command -v dos2unix &> /dev/null; then
    echo "dos2unix is not installed. Installing dos2unix..."
    sudo apt update
    sudo apt install -y dos2unix
fi

# Convert script line endings to LF if necessary
if file "$0" | grep -q 'CRLF line terminators'; then
    echo "Converting script line endings to LF..."
    dos2unix "$0"
fi

# Create and activate a virtual environment
echo "Creating and activating a virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Requirements
echo "Installing Pip Requirements..."
for lib in $(convert_to_pip_format "${REQUIRED_LIBRARIES[@]}"); do
    pip install "$lib"
done

# Install llama-cpp-python with specific flags for AMD AVX2 and non-ROCm AMDGPU
echo "Installing llama-cpp-python with specific flags for AMD AVX2 and non-ROCm AMDGPU..."
pip install llama-cpp-python --force-reinstall --no-binary :all: --config-settings CMAKE_ARGS="-DLLAMA_AVX2=ON -DLLAMA_CLBLAST=OFF"

# Install OpenCL and Vulkan Drivers
echo "Installing OpenCL and Vulkan Drivers..."
sudo apt update
sudo apt install -y mesa-opencl-icd vulkan-tools

echo
echo "Installation Processes Completed Normally."
exit 0
