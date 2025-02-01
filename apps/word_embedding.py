import numpy as np
import random
from collections import defaultdict

# Sample dataset of 50 sentences
sentences = [
    "The cat sits on the mat",
    "The dog barks at night",
    "Birds fly in the sky",
    "The sun shines bright",
    "Rain falls from the clouds",
    "The river flows to the sea",
    "A child plays in the park",
    "Leaves fall in autumn",
    "Stars twinkle at night",
    "A book rests on the table",
    "The moon orbits the Earth",
    "Fish swim in the water",
    "The wind blows strongly",
    "Mountains rise above the land",
    "The clock ticks every second",
    "Snow covers the ground",
    "Bees collect nectar from flowers",
    "A train moves on tracks",
    "The phone rings loudly",
    "Clouds drift in the sky",
    "The ocean waves crash on shore",
    "Trees grow tall and strong",
    "A plane flies above the clouds",
    "The candle flickers in the dark",
    "The bridge spans across the river",
    "A dog wags its tail happily",
    "People walk on the sidewalk",
    "Music plays from the radio",
    "A spider weaves a web",
    "A baby laughs with joy",
    "The butterfly flutters its wings",
    "A fire burns in the fireplace",
    "A squirrel gathers acorns",
    "The stars shine in the night",
    "A key unlocks the door",
    "The chef cooks delicious food",
    "A kite flies in the wind",
    "The mirror reflects an image",
    "The train station is busy",
    "A clock shows the time",
    "The artist paints a masterpiece",
    "A cat naps on the windowsill",
    "The bird sings in the morning",
    "A fish jumps out of the water",
    "A bicycle rolls down the street",
    "The lamp lights up the room",
    "The puppy chews on a toy",
    "The leaves rustle in the breeze",
    "A horse gallops across the field",
    "A fox sneaks through the woods",
]

# Tokenization
word_counts = defaultdict(int)
for sentence in sentences:
    for word in sentence.lower().split():
        word_counts[word] += 1

# Create word-to-index mapping
vocab = list(word_counts.keys())
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for word, i in word_to_index.items()}
vocab_size = len(vocab)

# Initialize Embedding Matrices
embedding_dim = 50
embeddings = np.random.rand(vocab_size, embedding_dim) * 0.01  # Target embeddings
context_embeddings = np.random.rand(vocab_size, embedding_dim) * 0.01  # Context embeddings

# Generate Training Data (Skip-Gram)
window_size = 2
training_data = []

for sentence in sentences:
    words = sentence.lower().split()
    word_indices = [word_to_index[word] for word in words]

    for center_idx in range(len(word_indices)):
        center_word = word_indices[center_idx]
        for w in range(-window_size, window_size + 1):
            context_idx = center_idx + w
            if 0 <= context_idx < len(word_indices) and center_idx != context_idx:
                training_data.append((center_word, word_indices[context_idx]))

# Training with Gradient Descent and Negative Sampling
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_skipgram(epochs=1000, learning_rate=0.01, neg_samples=5):
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(training_data)

        for target, context in training_data:
            # Positive sample
            positive_score = sigmoid(np.dot(embeddings[target], context_embeddings[context]))
            positive_loss = -np.log(positive_score)

            # Update embeddings
            gradient = learning_rate * (1 - positive_score)
            embeddings[target] += gradient * context_embeddings[context]
            context_embeddings[context] += gradient * embeddings[target]

            # Negative Sampling
            for _ in range(neg_samples):
                negative_word = random.randint(0, vocab_size - 1)
                negative_score = sigmoid(np.dot(embeddings[target], context_embeddings[negative_word]))
                negative_loss = -np.log(1 - negative_score)

                # Update for negative sample
                gradient = learning_rate * (-negative_score)
                embeddings[target] += gradient * context_embeddings[negative_word]
                context_embeddings[negative_word] += gradient * embeddings[target]

            total_loss += positive_loss + (neg_samples * negative_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

train_skipgram()

# Find Similar Words
def get_similar_words(word, top_n=5):
    if word not in word_to_index:
        return "Word not in vocabulary"

    word_vec = embeddings[word_to_index[word]]
    similarities = np.dot(embeddings, word_vec) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(word_vec)
    )
    sorted_indices = np.argsort(-similarities)

    return [index_to_word[i] for i in sorted_indices[:top_n]]

# Example
print("Words similar to 'cat':", get_similar_words("cat"))

# Example: Find words similar to "cat"
# Test the model
while True:
    user_input = input("Enter a sentence fragment: ")
    print("Words similar to " + user_input, get_similar_words(user_input))
