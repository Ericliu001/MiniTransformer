# Mini-GPT: Transformer Implementation

A minimal, educational implementation of the GPT (Generative Pre-trained Transformer) architecture based on the seminal paper **"Attention Is All You Need"** by Vaswani et al. (2017).

## Overview

This project implements a decoder-only Transformer model (GPT-style) from scratch using only NumPy, with extensive comments explaining each component. It's designed for educational purposes to help understand how modern language models work.

## Architecture

The implementation strictly follows the Transformer architecture described in "Attention Is All You Need" with the following key components:

### Core Components

1. **Scaled Dot-Product Attention**
   - The fundamental attention mechanism: `Attention(Q, K, V) = softmax(QK^T/√d_k)V`
   - Allows the model to focus on relevant parts of the input sequence

2. **Multi-Head Attention**
   - Splits the model into multiple attention heads
   - Each head learns different aspects of the relationships in the data
   - Outputs are concatenated and projected back to model dimension

3. **Position-wise Feed-Forward Networks**
   - Two-layer fully connected network applied to each position
   - Uses GELU activation function (common in modern transformers)
   - Formula: `FFN(x) = GELU(xW1 + b1)W2 + b2`

4. **Positional Encoding**
   - Sinusoidal position encodings to inject sequence order information
   - Uses sine and cosine functions of different frequencies
   - Allows the model to leverage position information

5. **Layer Normalization**
   - Stabilizes training by normalizing activations
   - Applied after each sub-layer with residual connections

6. **Residual Connections**
   - Skip connections around each sub-layer
   - Enables training of deep networks by allowing gradient flow

### Model Architecture

```
Input Text
    ↓
Token Embedding + Positional Encoding
    ↓
┌─────────────────────────────────────┐
│  Transformer Block 1                │
│  ├─ Multi-Head Self-Attention       │
│  ├─ Add & Layer Norm                │
│  ├─ Feed-Forward Network            │
│  └─ Add & Layer Norm                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Transformer Block 2                │
│  └─ (same structure)                │
└─────────────────────────────────────┘
    ↓
    ... (more blocks)
    ↓
Layer Normalization
    ↓
Output Projection to Vocabulary
    ↓
Softmax → Next Token Prediction
```

## Features

- **Pure NumPy Implementation**: No deep learning frameworks required
- **Comprehensive Comments**: Every function and component is thoroughly documented
- **Modular Design**: Clean separation between components
- **Training Pipeline**: Complete training script with data loading and preprocessing
- **Text Generation**: Autoregressive text generation with temperature sampling
- **Model Persistence**: Save and load trained models
- **Flexible Tokenization**: Supports both word-level and character-level tokenization

## Installation

### Requirements

- Python 3.7 or higher
- NumPy

```bash
pip install numpy
```

## Usage

### 1. Training a Model

Train a model on your own text data:

```bash
python train.py --data data/sample.txt --model_dir models/my_model --epochs 30
```

#### Training Arguments

- `--data`: Path to input text file (.txt)
- `--model_dir`: Directory to save the trained model
- `--tokenization`: Tokenization type (`word` or `char`, default: `word`)
- `--d_model`: Model dimension (default: 128)
- `--num_heads`: Number of attention heads (default: 4)
- `--num_layers`: Number of transformer layers (default: 4)
- `--d_ff`: Feed-forward network dimension (default: 512)
- `--max_seq_length`: Maximum sequence length (default: 128)
- `--learning_rate`: Learning rate (default: 0.001)
- `--epochs`: Number of training epochs (default: 20)
- `--min_token_frequency`: Minimum token frequency for vocabulary (default: 2)

#### Example with Custom Parameters

```bash
python train.py \
    --data data/sample.txt \
    --model_dir models/large_model \
    --d_model 256 \
    --num_heads 8 \
    --num_layers 6 \
    --d_ff 1024 \
    --epochs 50 \
    --learning_rate 0.0005
```

### 2. Generating Text

#### Single Prompt Mode

Generate text from a single prompt:

```bash
python generate.py --model_dir models/my_model --prompt "machine learning is"
```

#### Interactive Mode

Start an interactive session for continuous text generation:

```bash
python generate.py --model_dir models/my_model --interactive
```

#### Generation Arguments

- `--model_dir`: Directory containing the trained model
- `--prompt`: Text prompt for generation (single prompt mode)
- `--interactive`: Run in interactive mode
- `--max_new_tokens`: Maximum number of tokens to generate (default: 50)
- `--temperature`: Sampling temperature (default: 0.8)
  - Higher values (e.g., 1.2) → more random/creative
  - Lower values (e.g., 0.5) → more deterministic/focused
- `--quiet`: Suppress verbose output

#### Examples

```bash
# Generate with higher temperature (more creative)
python generate.py --model_dir models/my_model --prompt "the future of AI" --temperature 1.2

# Generate more tokens
python generate.py --model_dir models/my_model --prompt "attention mechanism" --max_new_tokens 100

# Quiet mode (just output the text)
python generate.py --model_dir models/my_model --prompt "deep learning" --quiet
```

## Project Structure

```
MiniTransformer/
├── mini_gpt.py           # Core model implementation
│   ├── MiniGPT class
│   ├── MultiHeadAttention
│   ├── PositionWiseFeedForward
│   ├── TransformerBlock
│   └── Helper functions (softmax, layer_norm, etc.)
│
├── data_utils.py         # Data loading and preprocessing
│   ├── Tokenizer class
│   ├── Data loading functions
│   └── Statistics utilities
│
├── train.py              # Training script
├── generate.py           # Text generation script
│
├── data/                 # Training data directory
│   └── sample.txt        # Sample training data
│
├── models/               # Saved models (created during training)
│   └── [model_name]/
│       ├── model.pkl
│       ├── tokenizer.pkl
│       └── config.txt
│
├── apps/                 # Legacy educational examples
│   ├── self_attention.py
│   ├── position_encoding.py
│   ├── mini_transformer.py
│   └── word_embedding.py
│
└── README.md
```

## Key Concepts Explained

### 1. Attention Mechanism

Attention allows the model to focus on different parts of the input when producing each output. The key innovation is computing attention scores between all positions in the sequence.

**Intuition**: When predicting the next word, the model can look back at all previous words and decide which ones are most relevant.

### 2. Multi-Head Attention

Instead of having one attention function, multi-head attention runs multiple attention operations in parallel. Each "head" can learn to focus on different aspects of the relationships between words.

**Intuition**: Like having multiple perspectives - one head might focus on grammar, another on semantics, etc.

### 3. Positional Encoding

Since transformers process all positions in parallel (unlike RNNs), we need to inject information about position. We use sinusoidal functions that allow the model to learn relative positions.

**Intuition**: Adding a "position fingerprint" to each word so the model knows word order.

### 4. Autoregressive Generation

The model generates text one token at a time, using previously generated tokens as context for the next prediction.

**Intuition**: Like writing a sentence word by word, where each new word depends on the words you've already written.

## Understanding the Paper Implementation

This implementation maps directly to concepts from "Attention Is All You Need":

| Paper Concept | Implementation |
|---------------|----------------|
| Scaled Dot-Product Attention (§3.2.1) | `scaled_dot_product_attention()` |
| Multi-Head Attention (§3.2.2) | `MultiHeadAttention` class |
| Position-wise FFN (§3.3) | `PositionWiseFeedForward` class |
| Positional Encoding (§3.5) | `create_positional_encoding()` |
| Encoder Stack | Not used (decoder-only for GPT) |
| Decoder Stack | `TransformerBlock` + `MiniGPT` |

## Training Tips

1. **Start Small**: Begin with smaller models (d_model=128, num_layers=4) to test quickly
2. **More Data = Better Results**: Larger, more diverse text datasets improve generation quality
3. **Adjust Temperature**: Lower temperature (0.5-0.7) for more coherent text, higher (1.0-1.2) for creativity
4. **Epochs vs Data Size**: More epochs needed for smaller datasets; fewer for large datasets
5. **Learning Rate**: Start with 0.001, reduce if training is unstable

## Limitations

This is an **educational implementation** with intentional simplifications:

- No proper backpropagation through all layers (simplified training)
- No advanced optimizers (Adam, AdamW)
- No dropout or regularization
- No batch processing
- Limited to smaller models due to NumPy constraints
- Training is slower than framework-based implementations

For production use cases, consider:
- PyTorch or TensorFlow implementations
- Pre-trained models like GPT-2, GPT-3, or Llama
- Proper optimization and regularization techniques

## Examples

### Example 1: Quick Start

```bash
# Train on sample data
python train.py --data data/sample.txt --model_dir models/sample --epochs 20

# Generate text
python generate.py --model_dir models/sample --prompt "machine learning"
```

### Example 2: Custom Dataset

```bash
# Create your own text file: my_data.txt
# Train with more layers and heads
python train.py \
    --data my_data.txt \
    --model_dir models/custom \
    --d_model 256 \
    --num_heads 8 \
    --num_layers 6 \
    --epochs 30

# Generate interactively
python generate.py --model_dir models/custom --interactive
```

### Example 3: Character-Level Model

```bash
# Train character-level model (better for small datasets)
python train.py \
    --data data/sample.txt \
    --model_dir models/char_model \
    --tokenization char \
    --epochs 50

# Generate
python generate.py --model_dir models/char_model --prompt "the" --max_new_tokens 200
```

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention is all you need**. In *Advances in neural information processing systems* (pp. 5998-6008).

2. Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). **Improving language understanding by generative pre-training**.

3. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). **Language models are unsupervised multitask learners**.

## Educational Resources

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) by Harvard NLP
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original paper

## Contributing

This is an educational project. Feel free to:
- Submit issues for bugs or unclear documentation
- Propose improvements to code clarity or comments
- Share interesting training results or examples

## License

See LICENSE file for details.

## Acknowledgments

This implementation is inspired by the groundbreaking work of Vaswani et al. and the broader transformer research community. The code prioritizes clarity and educational value over performance.

---

**Note**: This is a teaching tool. For production applications, use established deep learning frameworks and pre-trained models.
