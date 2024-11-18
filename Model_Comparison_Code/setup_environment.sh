#!/bin/bash

# Check for root privileges
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root or use sudo"
    exit
fi

# Update and install system packages
echo "Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install required system packages
echo "Installing system dependencies..."
sudo apt-get install -y python3 python3-pip python3-venv build-essential libopenblas-dev liblapack-dev gfortran

# Check and update grep
echo "Checking GNU grep version..."
if grep --version | grep -q "GNU grep"; then
    current_version=$(grep --version | head -n 1 | awk '{print $NF}')
    if dpkg --compare-versions "$current_version" ge "3.4"; then
        echo "GNU grep is up-to-date (version $current_version)."
    else
        echo "GNU grep is outdated (version $current_version). Updating..."
        wget https://ftp.gnu.org/gnu/grep/grep-3.10.tar.gz -O grep-latest.tar.gz
        tar -xvzf grep-latest.tar.gz
        cd grep-3.10
        ./configure
        make
        sudo make install
        cd ..
        rm -rf grep-3.10 grep-latest.tar.gz
        echo "GNU grep updated to the latest version."
    fi
else
    echo "GNU grep is not installed or not recognized. Installing latest GNU grep..."
    wget https://ftp.gnu.org/gnu/grep/grep-3.10.tar.gz -O grep-latest.tar.gz
    tar -xvzf grep-latest.tar.gz
    cd grep-3.10
    ./configure
    make
    sudo make install
    cd ..
    rm -rf grep-3.10 grep-latest.tar.gz
    echo "GNU grep installed successfully."
fi

# Create a virtual environment
echo "Setting up virtual environment..."
if [ ! -d "env" ]; then
    python3 -m venv env
fi

# Activate the virtual environment
source env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required Python packages
echo "Installing Python dependencies..."
pip install pandas numpy matplotlib scikit-learn statsmodels keras-self-attention

# Create required directories
echo "Setting up directories..."
mkdir -p Simulate_Cleaned_Merged saved_models

# Create placeholder files
echo "Creating placeholder files..."
touch Simulate_Cleaned_Merged/Station1_simulated_cleaned_merged_data.csv
touch Simulate_Cleaned_Merged/Station2_simulated_cleaned_merged_data.csv

# Verify installation
echo "Verifying installation..."
python3 -c "
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm
print('All required Python libraries installed successfully!')
"

# Instructions for the user
echo "Setup complete!"
echo "To activate the environment, run: source env/bin/activate"
echo "Run your script using: ./run_model_comparison.sh"
