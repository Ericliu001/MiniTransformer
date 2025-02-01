import numpy as np
import random
from collections import defaultdict

# Sample dataset of 50 sentences
sentences = [
    "The cat sits on the mat", "The dog barks at night", "Birds fly in the sky",
    "The sun shines bright", "Rain falls from the clouds", "The river flows to the sea",
    "A child plays in the park", "Leaves fall in autumn", "Stars twinkle at night",
    "A book rests on the table", "The moon orbits the Earth", "Fish swim in the water",
    "The wind blows strongly", "Mountains rise above the land", "The clock ticks every second",
    "Snow covers the ground", "Bees collect nectar from flowers", "A train moves on tracks",
    "The phone rings loudly", "Clouds drift in the sky", "The ocean waves crash on shore",
    "Trees grow tall and strong", "A plane flies above the clouds", "The candle flickers in the dark",
    "The bridge spans across the river", "A dog wags its tail happily", "People walk on the sidewalk",
    "Music plays from the radio", "A spider weaves a web", "A baby laughs with joy",
    "The butterfly flutters its wings", "A fire burns in the fireplace", "A squirrel gathers acorns",
    "The stars shine in the night", "A key unlocks the door", "The chef cooks delicious food",
    "A kite flies in the wind", "The mirror reflects an image", "The train station is busy",
    "A clock shows the time", "The artist paints a masterpiece", "A cat naps on the windowsill",
    "The bird sings in the morning", "A fish jumps out of the water", "A bicycle rolls down the street",
    "The lamp lights up the room", "The puppy chews on a toy", "The leaves rustle in the breeze",
    "A horse gallops across the field", "A fox sneaks through the woods"
]

# Step 1: Tokenization and Vocabulary
word_counts = defaultdict(int)
for sentence in sentences:
    for word in sentence.lower().split():
        word_counts[word] += 1

vocab = list(word_counts.keys())
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for word, i in word_to_index.items()}
vocab_size = len(vocab)

# Step 2: Initialize Embeddings (GPT-style single embedding matrix)
embedding_dim = 50  # embedding_dim (embedding dimension) is the size of the vector representation used to encode each word (or token) in a neural network. It determines how many numerical values represent a word in a high-dimensional space.
word_embeddings = np.random.rand(vocab_size, embedding_dim) * 0.01  # Small random values


# Step 3: Positional Encoding
def positional_encoding(seq_length, embedding_dim):
    pos_enc = np.zeros((seq_length, embedding_dim))
    for pos in range(seq_length):
        for i in range(0, embedding_dim, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** (i / embedding_dim)))
            pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / embedding_dim)))
    return pos_enc


max_seq_length = max(len(sentence.split()) for sentence in sentences)
pos_encodings = positional_encoding(max_seq_length, embedding_dim)


# Step 4: Scaled Dot-Product Attention (GPT-style Self-Attention)
def scaled_dot_product_attention(Q, K, V):
    """Compute attention weights and output"""
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)  # Scaled dot product
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)  # Softmax
    return np.dot(attention_weights, V)  # Output weighted sum


# Step 5: Training Loop (Self-Attention to update embeddings)
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    total_loss = 0
    random.shuffle(sentences)

    for sentence in sentences:
        words = sentence.lower().split()
        token_ids = [word_to_index[word] for word in words]

        # Extract embeddings & apply positional encoding
        X = word_embeddings[token_ids] + pos_encodings[:len(token_ids)]

        # Apply self-attention
        Q, K, V = X, X, X  # In GPT, Q=K=V from same input
        attention_output = scaled_dot_product_attention(Q, K, V)

        # Compute loss (Difference between attention output and original embeddings)
        loss = np.sum((attention_output - X) ** 2)
        total_loss += loss

        # Backpropagation (Gradient update on embeddings)
        gradients = (attention_output - X)
        for i, token_id in enumerate(token_ids):
            word_embeddings[token_id] -= learning_rate * gradients[i]

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Step 6: Normalize Embeddings for Similarity Calculation
word_embeddings = word_embeddings / np.linalg.norm(word_embeddings, axis=1, keepdims=True)


# Step 7: Find Similar Words using Cosine Similarity
def get_similar_words(word, top_n=5):
    if word not in word_to_index:
        return "Word not in vocabulary"

    word_vec = word_embeddings[word_to_index[word]]
    similarities = np.dot(word_embeddings, word_vec)  # Compute cosine similarity
    sorted_indices = np.argsort(-similarities)  # Sort by highest similarity

    return [index_to_word[i] for i in sorted_indices[:top_n]]


while True:
    # Example: Find words similar to "cat"
    user_input = input("Enter a word: ").lower()
    print(f"Words similar to '{user_input}': {get_similar_words(user_input)}")
