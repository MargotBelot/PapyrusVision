#!/bin/bash
echo "Starting PapyrusVision Hieroglyphs App..."
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/papyrus_env/bin/activate"
streamlit run apps/streamlit_hieroglyphs_app.py
