#!/bin/bash
# Setup script for Poker Environment
# Usage: ./setup.sh

set -e

echo "=== Poker Environment Setup ==="

# Check Python version
PYTHON_CMD=""
for cmd in python3.11 python3.12 python3.13 python3; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON_CMD=$cmd
            echo "Found Python $version at $(which $cmd)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3.11+ is required but not found."
    echo ""
    echo "Install options:"
    echo "  macOS:   brew install python@3.11"
    echo "  Ubuntu:  sudo apt install python3.11 python3.11-venv"
    echo "  pyenv:   pyenv install 3.11"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate and install dependencies
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run an experiment:"
echo "  python run_experiment.py --agent random --hands 100 --seed 42 --out logs/test.jsonl -v"
echo ""
echo "To run tests:"
echo "  pytest poker_env/tests/ -v"
