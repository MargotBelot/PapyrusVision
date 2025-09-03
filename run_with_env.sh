#!/bin/bash
export COPYFILE_DISABLE=1
# PapyrusVision Environment Activation Script
# This script activates the virtual environment and runs the specified command

# Path to the virtual environment
ENV_PATH="$HOME/papyrus_env"

# Check if virtual environment exists
if [ ! -d "$ENV_PATH" ]; then
    echo "Error: Virtual environment not found at $ENV_PATH"
    echo "Please run the installation script first."
    exit 1
fi

# Activate the virtual environment and run the command
source "$ENV_PATH/bin/activate"

# If no arguments provided, just activate the environment
if [ $# -eq 0 ]; then
    echo "PapyrusVision virtual environment activated!"
    echo "You can now run Python scripts or install additional packages."
    echo "To deactivate, type 'deactivate'"
    exec bash
else
    # Run the provided command with the activated environment
    exec "$@"
fi
