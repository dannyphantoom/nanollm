#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored status messages
print_status() {
    echo -e "${GREEN}[+] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[!] $1${NC}"
}

print_error() {
    echo -e "${RED}[-] $1${NC}"
}

# Check for python3 command
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    print_error "Python 3 is not installed. Please install it first."
    print_status "On Ubuntu/Debian, run: sudo apt install python3-full python3-pip python3-venv"
    exit 1
fi

# Check if running in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_warning "No virtual environment detected. It's recommended to use one."
    read -p "Create a new virtual environment? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Creating virtual environment..."
        # Check if python3-venv is installed
        if ! dpkg -l | grep -q python3-venv; then
            print_warning "python3-venv is not installed. Installing it now..."
            sudo apt-get update && sudo apt-get install -y python3-venv
        fi
        $PYTHON_CMD -m venv venv
        if [ $? -ne 0 ]; then
            print_error "Failed to create virtual environment. Please install python3-venv manually."
            exit 1
        fi
        source venv/bin/activate
        if [ $? -ne 0 ]; then
            print_error "Failed to activate virtual environment."
            exit 1
        fi
        print_status "Installing pip in virtual environment..."
        $PYTHON_CMD -m ensurepip --upgrade
        $PYTHON_CMD -m pip install --upgrade pip --break-system-packages
    else
        print_error "A virtual environment is required for this project."
        exit 1
    fi
fi

# Check Python version
python_version=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
major_version=$($PYTHON_CMD -c 'import sys; print(sys.version_info[0])')
minor_version=$($PYTHON_CMD -c 'import sys; print(sys.version_info[1])')

if [ "$major_version" -lt 3 ] || ([ "$major_version" -eq 3 ] && [ "$minor_version" -lt 8 ]); then
    print_error "Python version must be 3.8 or higher. Current version: $python_version"
    exit 1
fi

# Check CUDA availability
print_status "Checking CUDA availability..."
cuda_available=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "$cuda_available" = "True" ]; then
    print_status "CUDA is available!"
    device="cuda"
else
    print_warning "CUDA is not available. Training will be slow on CPU."
    device="cpu"
fi

# Install dependencies
print_status "Installing dependencies..."
$PYTHON_CMD -m pip install --upgrade pip --break-system-packages

# First install PyTorch
print_status "Installing PyTorch..."
if [ "$device" = "cuda" ]; then
    $PYTHON_CMD -m pip install torch --index-url https://download.pytorch.org/whl/cu118 --break-system-packages
else
    $PYTHON_CMD -m pip install torch --break-system-packages
fi

# Then install other dependencies
print_status "Installing other dependencies..."
# Read requirements.txt and install everything except torch and flash-attn
grep -v "^torch\|^flash-attn" requirements.txt | while read -r package; do
    if [ ! -z "$package" ]; then
        $PYTHON_CMD -m pip install "$package" --break-system-packages
    fi
done

# Finally install flash-attn
print_status "Installing flash-attn..."
$PYTHON_CMD -m pip install flash-attn --break-system-packages

if [ $? -ne 0 ]; then
    print_warning "Failed to install flash-attn. This is optional and the model will still work without it."
fi

# Check if torch is installed with CUDA support
if [ "$device" = "cuda" ]; then
    torch_cuda=$($PYTHON_CMD -c "import torch; print(torch.version.cuda is not None)" 2>/dev/null || echo "False")
    if [ "$torch_cuda" = "False" ]; then
        print_warning "PyTorch is installed without CUDA support. Reinstalling with CUDA..."
        $PYTHON_CMD -m pip uninstall -y torch
        $PYTHON_CMD -m pip install torch --index-url https://download.pytorch.org/whl/cu118 --break-system-packages
    fi
fi

# Check if wandb is configured
if ! wandb login --check > /dev/null 2>&1; then
    print_warning "Weights & Biases not configured. Some features may be limited."
    print_warning "You can configure it later by running: wandb login"
fi

# Create necessary directories
mkdir -p checkpoints
mkdir -p logs

# Function to check if training was successful
check_training_success() {
    if ls checkpoints/best/checkpoint_epoch_*.pt 1> /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Start training
print_status "Starting training..."
$PYTHON_CMD scripts/train_model.py \
    --config config/config.json \
    --dataset wikitext \
    2>&1 | tee logs/training.log

# Check if training was successful
if check_training_success; then
    print_status "Training completed successfully!"
else
    print_error "Training may have failed. Check logs/training.log for details."
    exit 1
fi

# Start the chat interface
print_status "Starting NanoLLM Chat Interface..."
if ls checkpoints/best/checkpoint_epoch_*.pt 1> /dev/null 2>&1; then
    $PYTHON_CMD GUI/main.py
else
    print_error "Could not find a trained model. Please check if training completed successfully."
    exit 1
fi 