#!/bin/bash
# setup_env.sh - Script to set up the environment for Gemma fine-tuning

# Print execution steps
set -x

# Create and activate virtual environment (if needed)
if [ ! -d "gemma_env" ]; then
    echo "Creating virtual environment..."
    python -m venv gemma_env
fi

# Activate environment (source this script instead of executing it)
# source gemma_env/bin/activate

# Uninstall conflicting packages
pip uninstall -y datasets
pip uninstall -y torchvision

# Install required packages
pip install transformers==4.36.2
pip install peft==0.7.1
pip install accelerate==0.24.1
pip install huggingface_hub==0.19.4
pip install pillow==10.0.1
pip install pymupdf==1.23.7
pip install tqdm==4.66.1
pip install tensorboard==2.14.1
pip install einops==0.7.0

# Install Hugging Face datasets with correct version
pip install huggingface_datasets==2.16.0

# Install PyTorch with compatible version
pip install torch==2.0.1 torchvision==0.15.2

# Create alias for datasets import
echo "from huggingface_datasets import Dataset" > huggingface_datasets.py

echo "Environment setup complete!"
echo "Please source this script to activate the environment: source gemma_env/bin/activate"