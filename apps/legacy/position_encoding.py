import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(seq_len, d_model):
    """
    Compute positional encoding using only math functions.

    Args:
        seq_len (int): Number of positions (sequence length).
        d_model (int): Embedding size.

    Returns:
        numpy.ndarray: Positional encoding matrix of shape (seq_len, d_model).
    """
    positions = np.arange(seq_len)[:, np.newaxis]  # Shape: (seq_len, 1)
    indices = np.arange(d_model)[np.newaxis, :]    # Shape: (1, d_model)

    # Compute the denominator term 10000^(2i/d_model)
    div_term = np.power(10000.0, (2 * (indices // 2)) / d_model)

    # Compute the position encoding matrix
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(positions / div_term[:, 0::2])  # Apply sin to even indices
    pe[:, 1::2] = np.cos(positions / div_term[:, 1::2])  # Apply cos to odd indices

    return pe

# Example usage
seq_length = 10  # Small sequence length for display
d_model = 16     # Smaller embedding dimension for readability
pos_encoding = positional_encoding(seq_length, d_model)

# Print the first few rows
print("Positional Encoding Sample (First 5 Positions):\n")
for row in pos_encoding[:5]:
    print(row)


# Visualizing the position encoding
plt.figure(figsize=(10, 6))
plt.imshow(pos_encoding, cmap="viridis", aspect="auto")
plt.colorbar(label="Encoding Value")
plt.xlabel("Embedding Dimension")
plt.ylabel("Position in Sequence")
plt.title("Positional Encoding Heatmap")
plt.show()