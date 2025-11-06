"""
Data Utilities for Mini-GPT

This module provides utilities for:
1. Loading and preprocessing text data from .txt files
2. Building vocabulary from text
3. Tokenizing text into token IDs
4. Creating training batches
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import re


# ============================================================================
# TOKENIZATION
# ============================================================================

class Tokenizer:
    """
    Simple character-level or word-level tokenizer.

    For educational purposes, we implement a basic tokenizer. In production,
    you would typically use a more sophisticated tokenizer like BPE (Byte Pair Encoding)
    or WordPiece as used in GPT-2/GPT-3.
    """

    def __init__(self, tokenization_type: str = "word"):
        """
        Initialize tokenizer.

        Args:
            tokenization_type: Either "word" or "char" for word-level or character-level
        """
        assert tokenization_type in ["word", "char"], "tokenization_type must be 'word' or 'char'"
        self.tokenization_type = tokenization_type

        # Vocabulary mappings
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"  # Beginning of sequence
        self.eos_token = "<EOS>"  # End of sequence

        self.vocab_size = 0

    def build_vocabulary(self, text: str, min_frequency: int = 1) -> None:
        """
        Build vocabulary from text.

        Args:
            text: Input text to build vocabulary from
            min_frequency: Minimum frequency for a token to be included in vocabulary
        """
        # Tokenize text
        if self.tokenization_type == "char":
            tokens = list(text)
        else:  # word-level
            # Simple word tokenization: lowercase and split on whitespace/punctuation
            tokens = self._tokenize_text(text)

        # Count token frequencies
        token_counts = Counter(tokens)

        # Build vocabulary starting with special tokens
        self.token_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3
        }

        # Add tokens that meet minimum frequency
        idx = len(self.token_to_id)
        for token, count in token_counts.most_common():
            if count >= min_frequency:
                if token not in self.token_to_id:
                    self.token_to_id[token] = idx
                    idx += 1

        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)

        print(f"Vocabulary built: {self.vocab_size} tokens")

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Lowercase
        text = text.lower()

        # Add spaces around punctuation for better tokenization
        text = re.sub(r'([.,!?;:])', r' \1 ', text)

        # Split on whitespace
        tokens = text.split()

        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to token IDs.

        Args:
            text: Input text to encode
            add_special_tokens: Whether to add BOS and EOS tokens

        Returns:
            List of token IDs
        """
        # Tokenize
        if self.tokenization_type == "char":
            tokens = list(text)
        else:
            tokens = self._tokenize_text(text)

        # Convert tokens to IDs
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.token_to_id[self.bos_token])

        for token in tokens:
            token_id = self.token_to_id.get(token, self.token_to_id[self.unk_token])
            token_ids.append(token_id)

        if add_special_tokens:
            token_ids.append(self.token_to_id[self.eos_token])

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text
        """
        special_token_ids = {
            self.token_to_id[self.pad_token],
            self.token_to_id[self.unk_token],
            self.token_to_id[self.bos_token],
            self.token_to_id[self.eos_token]
        }

        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_token_ids:
                continue

            token = self.id_to_token.get(token_id, self.unk_token)
            tokens.append(token)

        # Join tokens
        if self.tokenization_type == "char":
            return "".join(tokens)
        else:
            # Join words with spaces, but handle punctuation
            text = " ".join(tokens)
            # Remove spaces before punctuation
            text = re.sub(r'\s+([.,!?;:])', r'\1', text)
            return text


# ============================================================================
# DATA LOADING
# ============================================================================

def load_text_file(filepath: str) -> str:
    """
    Load text from a file.

    Args:
        filepath: Path to the text file

    Returns:
        Text content as a string
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error reading file {filepath}: {str(e)}")


def prepare_training_data(
    text: str,
    tokenizer: Tokenizer,
    max_seq_length: int = 128
) -> List[List[int]]:
    """
    Prepare training data by tokenizing text and creating sequences.

    Args:
        text: Input text
        tokenizer: Tokenizer instance
        max_seq_length: Maximum sequence length

    Returns:
        List of tokenized sequences
    """
    # Encode the entire text
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Split into sequences of max_seq_length
    sequences = []
    for i in range(0, len(token_ids) - 1, max_seq_length):
        # Get a chunk of max_seq_length + 1 tokens
        # (we need +1 because we'll use tokens[:-1] as input and tokens[1:] as target)
        sequence = token_ids[i:i + max_seq_length + 1]

        # Only include sequences with at least 2 tokens
        if len(sequence) >= 2:
            sequences.append(sequence)

    return sequences


def save_tokenizer(tokenizer: Tokenizer, filepath: str) -> None:
    """
    Save tokenizer vocabulary to a file.

    Args:
        tokenizer: Tokenizer instance
        filepath: Path to save the tokenizer
    """
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {filepath}")


def load_tokenizer(filepath: str) -> Tokenizer:
    """
    Load tokenizer from a file.

    Args:
        filepath: Path to the tokenizer file

    Returns:
        Loaded tokenizer instance
    """
    import pickle
    with open(filepath, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer loaded from {filepath}")
    return tokenizer


# ============================================================================
# STATISTICS AND ANALYSIS
# ============================================================================

def print_data_statistics(text: str, tokenizer: Tokenizer) -> None:
    """
    Print statistics about the text data.

    Args:
        text: Input text
        tokenizer: Tokenizer instance
    """
    print("=" * 60)
    print("Data Statistics")
    print("=" * 60)

    # Character-level statistics
    print(f"Total characters: {len(text):,}")
    print(f"Unique characters: {len(set(text)):,}")

    # Token-level statistics
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"Total tokens: {len(token_ids):,}")
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")

    # Show some example tokens
    print("\nExample tokens (first 20):")
    example_ids = token_ids[:20]
    example_tokens = [tokenizer.id_to_token[tid] for tid in example_ids]
    print(f"Token IDs: {example_ids}")
    print(f"Tokens: {example_tokens}")

    print("\nSpecial tokens:")
    print(f"  PAD: {tokenizer.token_to_id[tokenizer.pad_token]}")
    print(f"  UNK: {tokenizer.token_to_id[tokenizer.unk_token]}")
    print(f"  BOS: {tokenizer.token_to_id[tokenizer.bos_token]}")
    print(f"  EOS: {tokenizer.token_to_id[tokenizer.eos_token]}")
    print("=" * 60)
    print()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Data Utilities for Mini-GPT")
    print()

    # Example: Create a sample text file for testing
    sample_text = """
    The transformer architecture has revolutionized natural language processing.
    Attention mechanisms allow models to focus on relevant parts of the input.
    Deep learning continues to advance the field of artificial intelligence.
    Training large language models requires significant computational resources.
    """

    # Create tokenizer
    print("Creating word-level tokenizer...")
    tokenizer = Tokenizer(tokenization_type="word")
    tokenizer.build_vocabulary(sample_text, min_frequency=1)

    # Print statistics
    print_data_statistics(sample_text, tokenizer)

    # Example encoding and decoding
    print("Example: Encoding and decoding")
    test_sentence = "The transformer architecture is powerful."
    encoded = tokenizer.encode(test_sentence)
    decoded = tokenizer.decode(encoded)

    print(f"Original: {test_sentence}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print()

    # Example: Prepare training data
    print("Preparing training data...")
    sequences = prepare_training_data(sample_text, tokenizer, max_seq_length=10)
    print(f"Created {len(sequences)} training sequences")
    print(f"Example sequence (first): {sequences[0]}")
