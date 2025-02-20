#!/bin/bash

# Exit on error
set -e

# Install dependencies
echo "Installing system dependencies..."
apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    software-properties-common \

# Add deadsnakes PPA if not already added
if ! grep -q "^deb.*deadsnakes" /etc/apt/sources.list /etc/apt/sources.list.d/*; then
    echo "Adding deadsnakes PPA..."
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
fi

# Install Python 3.13 if not already installed
if ! command -v python3.13 &> /dev/null; then
    echo "Installing Python 3.13..."
    apt-get install -y python3.13-full
    python3.13 -m ensurepip --upgrade
fi

# Create and activate virtual environment
VENV_PATH="./venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python3.13 -m venv "$VENV_PATH"
fi

# Activate virtual environment and install requirements
echo "Activating virtual environment and installing requirements..."
source "$VENV_PATH/bin/activate"
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found"
fi

echo "Setup complete! To activate the virtual environment, run: source $VENV_PATH/bin/activate"

