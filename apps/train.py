"""
Training Script for Mini-GPT

This script trains a Mini-GPT model on a text file and saves the trained model to disk.

Usage:
    python train.py --data <path_to_text_file> --model_dir <output_directory>

Example:
    python train.py --data data/sample.txt --model_dir models/my_model
"""

import argparse
import os
import sys
from mini_gpt import MiniGPT
from data_utils import (
    Tokenizer,
    load_text_file,
    prepare_training_data,
    save_tokenizer,
    print_data_statistics
)


def train_model(
    data_file: str,
    model_dir: str,
    tokenization_type: str = "word",
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    d_ff: int = 512,
    max_seq_length: int = 128,
    learning_rate: float = 0.001,
    epochs: int = 20,
    min_token_frequency: int = 2
) -> None:
    """
    Train a Mini-GPT model on text data.

    Args:
        data_file: Path to the input text file
        model_dir: Directory to save the trained model and tokenizer
        tokenization_type: Type of tokenization ("word" or "char")
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        d_ff: Feed-forward network dimension
        max_seq_length: Maximum sequence length
        learning_rate: Learning rate for training
        epochs: Number of training epochs
        min_token_frequency: Minimum frequency for tokens to be included in vocabulary
    """
    print("=" * 80)
    print("Mini-GPT Training Script")
    print("=" * 80)
    print()

    # Create output directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Step 1: Load text data
    print(f"Loading data from: {data_file}")
    try:
        text = load_text_file(data_file)
        print(f"Loaded {len(text):,} characters")
        print()
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Step 2: Build vocabulary and tokenizer
    print("Building vocabulary...")
    tokenizer = Tokenizer(tokenization_type=tokenization_type)
    tokenizer.build_vocabulary(text, min_frequency=min_token_frequency)
    print()

    # Print data statistics
    print_data_statistics(text, tokenizer)

    # Step 3: Prepare training data
    print("Preparing training sequences...")
    sequences = prepare_training_data(text, tokenizer, max_seq_length=max_seq_length)
    print(f"Created {len(sequences):,} training sequences")
    print(f"Sequence length: up to {max_seq_length} tokens")
    print()

    # Step 4: Initialize model
    print("Initializing Mini-GPT model...")
    print(f"  Model dimension (d_model): {d_model}")
    print(f"  Number of attention heads: {num_heads}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Feed-forward dimension: {d_ff}")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Maximum sequence length: {max_seq_length}")
    print(f"  Learning rate: {learning_rate}")
    print()

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        learning_rate=learning_rate
    )

    # Step 5: Train the model
    print("=" * 80)
    print("Starting Training")
    print("=" * 80)
    print()

    model.train(sequences, epochs=epochs, verbose=True)

    print()
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print()

    # Step 6: Save model and tokenizer
    model_path = os.path.join(model_dir, "model.pkl")
    tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")

    print("Saving model and tokenizer...")
    model.save(model_path)
    save_tokenizer(tokenizer, tokenizer_path)
    print()

    # Save training configuration
    config_path = os.path.join(model_dir, "config.txt")
    with open(config_path, 'w') as f:
        f.write("Mini-GPT Training Configuration\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Data file: {data_file}\n")
        f.write(f"Tokenization type: {tokenization_type}\n")
        f.write(f"Model dimension: {d_model}\n")
        f.write(f"Number of heads: {num_heads}\n")
        f.write(f"Number of layers: {num_layers}\n")
        f.write(f"Feed-forward dimension: {d_ff}\n")
        f.write(f"Max sequence length: {max_seq_length}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Vocabulary size: {tokenizer.vocab_size}\n")
        f.write(f"Training sequences: {len(sequences)}\n")

    print(f"Training configuration saved to: {config_path}")
    print()

    # Step 7: Test the model with a few examples
    print("=" * 80)
    print("Testing Model")
    print("=" * 80)
    print()

    test_prompts = [
        "the",
        "machine learning",
        "attention is"
    ]

    for prompt in test_prompts:
        try:
            # Encode prompt
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

            # Generate text
            generated_ids = model.generate(prompt_ids, max_new_tokens=10, temperature=0.8)

            # Decode generated text
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            print(f"Prompt: '{prompt}'")
            print(f"Generated: '{generated_text}'")
            print()
        except Exception as e:
            print(f"Error generating text for prompt '{prompt}': {e}")
            print()

    print("=" * 80)
    print("All Done!")
    print("=" * 80)
    print()
    print(f"Model saved to: {model_dir}")
    print(f"To generate text, run:")
    print(f"  python generate.py --model_dir {model_dir} --prompt 'your prompt here'")


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train a Mini-GPT model on text data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the input text file (.txt)"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory to save the trained model"
    )

    # Model architecture arguments
    parser.add_argument(
        "--tokenization",
        type=str,
        default="word",
        choices=["word", "char"],
        help="Tokenization type: word-level or character-level"
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
        help="Model dimension (embedding size)"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Number of attention heads (must divide d_model)"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=512,
        help="Feed-forward network dimension"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )

    # Training arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--min_token_frequency",
        type=int,
        default=2,
        help="Minimum frequency for tokens to be included in vocabulary"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.d_model % args.num_heads != 0:
        print(f"Error: d_model ({args.d_model}) must be divisible by num_heads ({args.num_heads})")
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)

    # Train the model
    train_model(
        data_file=args.data,
        model_dir=args.model_dir,
        tokenization_type=args.tokenization,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        min_token_frequency=args.min_token_frequency
    )


if __name__ == "__main__":
    main()
