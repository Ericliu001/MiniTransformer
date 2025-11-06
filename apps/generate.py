"""
Text Generation Script for Mini-GPT

This script loads a trained Mini-GPT model and generates text based on user prompts.

Usage:
    python generate.py --model_dir <model_directory> --prompt "your prompt"

Example:
    python generate.py --model_dir models/my_model --prompt "machine learning is"
    python generate.py --model_dir models/my_model --interactive
"""

import argparse
import os
import sys
from mini_gpt import MiniGPT
from data_utils import load_tokenizer


def generate_text(
    model_dir: str,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    verbose: bool = True
) -> str:
    """
    Generate text using a trained Mini-GPT model.

    Args:
        model_dir: Directory containing the trained model and tokenizer
        prompt: Text prompt to start generation
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (higher = more random)
        verbose: Whether to print generation details

    Returns:
        Generated text
    """
    # Load model and tokenizer
    model_path = os.path.join(model_dir, "model.pkl")
    tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    if verbose:
        print("Loading model and tokenizer...")

    model = MiniGPT.load(model_path)
    tokenizer = load_tokenizer(tokenizer_path)

    if verbose:
        print(f"Model vocabulary size: {tokenizer.vocab_size}")
        print(f"Model dimension: {model.d_model}")
        print(f"Number of layers: {model.num_layers}")
        print(f"Number of heads: {model.num_heads}")
        print()

    # Encode prompt
    if verbose:
        print(f"Prompt: '{prompt}'")

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    if len(prompt_ids) == 0:
        if verbose:
            print("Warning: Empty prompt after tokenization. Using beginning-of-sequence token.")
        prompt_ids = [tokenizer.token_to_id[tokenizer.bos_token]]

    if verbose:
        print(f"Prompt tokens: {[tokenizer.id_to_token[tid] for tid in prompt_ids]}")
        print()
        print("Generating text...")
        print("-" * 60)

    # Generate text
    generated_ids = model.generate(
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

    # Decode generated text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text


def interactive_mode(model_dir: str, max_new_tokens: int = 50, temperature: float = 0.8) -> None:
    """
    Run interactive text generation mode.

    Args:
        model_dir: Directory containing the trained model
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
    """
    print("=" * 80)
    print("Mini-GPT Interactive Text Generation")
    print("=" * 80)
    print()
    print("Loading model...")

    # Load model and tokenizer once
    model_path = os.path.join(model_dir, "model.pkl")
    tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")

    model = MiniGPT.load(model_path)
    tokenizer = load_tokenizer(tokenizer_path)

    print(f"Model loaded successfully!")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print()
    print("=" * 80)
    print("Enter prompts to generate text. Type 'quit' or 'exit' to stop.")
    print("=" * 80)
    print()

    while True:
        # Get user input
        prompt = input("Prompt: ").strip()

        if prompt.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not prompt:
            print("Please enter a valid prompt.")
            print()
            continue

        try:
            # Encode prompt
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

            if len(prompt_ids) == 0:
                print("Warning: Prompt not recognized. Using beginning-of-sequence token.")
                prompt_ids = [tokenizer.token_to_id[tokenizer.bos_token]]

            # Generate text
            generated_ids = model.generate(
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

            # Decode generated text
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Display result
            print("-" * 60)
            print(f"Generated: {generated_text}")
            print("-" * 60)
            print()

        except Exception as e:
            print(f"Error during generation: {e}")
            print()


def main():
    """Main entry point for the generation script."""
    parser = argparse.ArgumentParser(
        description="Generate text using a trained Mini-GPT model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the trained model and tokenizer"
    )

    # Generation mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for generation"
    )
    group.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    # Generation parameters
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random, lower = more deterministic)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)

    if args.temperature <= 0:
        print("Error: Temperature must be positive")
        sys.exit(1)

    try:
        if args.interactive:
            # Interactive mode
            interactive_mode(
                model_dir=args.model_dir,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )
        else:
            # Single prompt mode
            print("=" * 80)
            print("Mini-GPT Text Generation")
            print("=" * 80)
            print()

            generated_text = generate_text(
                model_dir=args.model_dir,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                verbose=not args.quiet
            )

            if args.quiet:
                # Just print the generated text
                print(generated_text)
            else:
                print()
                print("=" * 80)
                print("Generated Text:")
                print("=" * 80)
                print(generated_text)
                print()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
