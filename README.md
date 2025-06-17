# NanoLLM

A lightweight implementation of a transformer-based language model that can be trained on consumer hardware. This implementation includes modern architecture choices and optimizations while maintaining readability and extensibility.

## Features

- Transformer architecture with modern improvements:
  - Rotary Position Embeddings (RoPE)
  - Flash Attention support (optional)
  - SwiGLU activation functions
  - Cosine learning rate scheduling
  - Gradient clipping
- BPE tokenizer with efficient implementation
- Training with validation and checkpointing
- Weights & Biases integration for experiment tracking
- Memory-efficient attention implementation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nanollm.git
cd nanollm

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
nanollm/
├── config/           # Configuration files
├── model/           # Model architecture
├── tokenizer/       # BPE tokenizer implementation
├── train/           # Training utilities
├── inference/       # Inference and sampling
└── scripts/         # Training and utility scripts
```

## Usage

### Training

```bash
# Train from scratch
python scripts/train_model.py

# Resume training from checkpoint
python scripts/train_model.py --resume_from checkpoints/checkpoint_epoch_10.pt
```

### Inference

```python
from model import TransformerModel
from tokenizer import BPETokenizer
from inference.sampler import sample

# Load model and tokenizer
model = TransformerModel.from_pretrained('checkpoints/best')
tokenizer = BPETokenizer.from_pretrained('tokenizer/vocab.json')

# Generate text
prompt = "Once upon a time"
generated = sample(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8)
print(generated)
```

## Training Tips

1. **Hardware Requirements**:
   - Minimum: 8GB GPU VRAM
   - Recommended: 16GB+ GPU VRAM
   - CPU training is supported but not recommended for large models

2. **Optimization**:
   - Use Flash Attention if available for better memory efficiency
   - Adjust batch size based on your GPU memory
   - Start with a small model and gradually scale up

3. **Hyperparameters**:
   - Default learning rate: 3e-4
   - Gradient clipping: 1.0
   - Weight decay: 0.1

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details 