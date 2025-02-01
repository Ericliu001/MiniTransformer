import numpy as np
import random
from collections import defaultdict

# Sample dataset of 50 sentences
import numpy as np
import random
from collections import defaultdict

# Expanded dataset with 100 interconnected sentences
sentences = [
    "The ocean stretches endlessly beyond the horizon, where the sky meets the waves, and the ocean whispers secrets that only the wind can hear.",
    "Waves crash against the shore with relentless energy, carving the coastline over centuries, while waves retreat quietly, leaving behind stories written in the sand.",
    "The tide rises with the pull of the moon, creeping onto the land with quiet determination, and the tide falls back again, revealing treasures buried beneath the shifting sands.",
    "The deep sea hides mysteries beyond human reach, where sunlight fades into eternal darkness, and the deep sea nurtures creatures unseen by the surface world.",
    "The ocean carries the weight of time, holding the memories of ancient civilizations lost beneath its surface, and the ocean breathes with the rhythm of the earth itself.",
    "Storms brew above the restless sea, whipping the waves into mountains of water, while storms pass over the calm ocean, leaving behind the silence of a cleansed sky.",
    "Sailors follow the currents that weave through the ocean’s vast expanse, trusting the stars to guide them, while sailors fear the unknown, where the depths hide dangers unseen.",
    "A lighthouse stands tall against the relentless tides, casting its guiding light across the waters, and a lighthouse watches over lost ships, offering hope in the darkest nights.",
    "Seashells scatter across the shore, washed up from distant places, and seashells carry the echoes of waves, whispering their stories to those who listen.",
    "The ocean reflects the sky above, mirroring the colors of sunrise and sunset, yet the ocean holds its own secrets, hidden beneath the ever-changing surface.",
    "Fishermen cast their nets into the depths, hoping for a bountiful catch, while fishermen respect the ocean’s power, knowing it gives and takes as it pleases.",
    "A lonely island rises from the water, its cliffs battered by relentless waves, while a lonely island shelters life, untouched by the world beyond its shores.",
    "The sun sinks into the sea at dusk, painting the sky with fire, while the sun rises from the horizon at dawn, igniting the waters with golden light.",
    "Whales sing through the ocean’s depths, their voices echoing for miles, and whales travel great distances, following the ancient pathways of their ancestors.",
    "The sea breeze carries the scent of salt and adventure, filling the sails of wandering ships, while the sea breeze whispers through the dunes, shaping the land as it moves.",
    "Coral reefs bloom beneath the waves, bustling with life in a hidden paradise, yet coral reefs suffer under the weight of time, crumbling as the waters grow warmer.",
    "The ocean floor holds forgotten relics, where sunken ships rest in silence, and the ocean floor is a world unseen, untouched by light and unknown to most.",
    "Waves chase each other across the surface, dancing in an endless rhythm, while waves break upon the shore, smoothing stones with each gentle touch.",
    "A message in a bottle drifts with the current, carrying a story across the sea, while a message in a bottle waits to be found, a mystery wrapped in glass.",
    "Seagulls circle above the crashing waves, their cries carried by the wind, while seagulls dive into the waters, seeking fish hidden beneath the rippling surface.",
    "The ocean teems with life, from the smallest plankton to the largest whales, yet the ocean keeps its balance, a delicate harmony of predator and prey.",
    "A ship sails toward the horizon, disappearing into the unknown, while a shipwreck rests beneath the waves, its journey forever unfinished.",
    "The sound of the ocean soothes the restless mind, waves lapping against the shore in a steady rhythm, while the sound of the ocean echoes in empty shells, a song of the sea.",
    "Dolphins leap from the water, playful and free, chasing the boats that cross their path, while dolphins travel in pods, bound by loyalty to one another.",
    "The ocean changes with the seasons, growing restless under stormy skies, yet the ocean finds peace, still and glassy beneath the warmth of summer.",
    "The depths are filled with creatures of wonder, glowing in the darkness where light cannot reach, while the depths remain unexplored, holding mysteries yet to be uncovered.",
    "A storm builds over the horizon, dark clouds rolling over the sea, and a storm crashes upon the waves, stirring the waters into chaos.",
    "The moon watches over the tides, pulling the waters as it moves, while the moon’s reflection ripples across the surface, a shimmering path of light.",
    "The seaweed sways with the currents, dancing to the ocean’s song, while the seaweed shelters creatures that hide within its tangled embrace.",
    "A sailor's heart belongs to the sea, longing for the open waters, yet a sailor's soul remembers home, yearning for the land once left behind.",
    "A pearl forms in the depths, hidden within the oyster’s embrace, while a pearl is discovered, a treasure from the heart of the sea.",
    "The ocean roars in fury when storms awaken its wrath, waves towering over ships, while the ocean whispers in peace when dawn calms its restless waters.",
    "A surfer rides the waves, balancing between movement and stillness, while a surfer waits for the perfect wave, watching the ocean for a moment of harmony.",
    "The ocean connects distant lands, carrying explorers across its vast waters, while the ocean separates worlds, standing as a barrier between those who long to meet.",
    "A tide pool cradles tiny creatures, a miniature world left behind by the sea, while a tide pool vanishes with the waves, returning to the embrace of the ocean.",
    "The ocean holds a history deeper than any book, recording the passage of time in its shifting sands, yet the ocean forgets, washing away footprints before they can remain.",
    "The scent of salt lingers in the air, clinging to clothes and skin, while the scent of salt calls to travelers, reminding them of distant shores.",
    "Waves write stories upon the sand, erasing them before they can be read, while waves sing lullabies to the shore, whispering with every retreat.",
    "A diver descends into the blue abyss, entering a world unlike any other, while a diver resurfaces, carrying memories of wonders unseen by those above.",
    "The ocean listens to the voices of sailors, carrying their songs across the waves, while the ocean keeps their secrets, burying them beneath the tide.",
    "A storm shakes the waters, stirring the depths into turmoil, while a storm passes, leaving the ocean calm but forever changed.",
    "A harbor welcomes weary travelers, offering shelter from the restless sea, while a harbor watches them leave, knowing they will always return to the waves.",
    "The ocean is a mother to the world, nurturing life in its vast embrace, yet the ocean is a force of nature, powerful enough to take back what it gives.",
    "A mermaid sings beneath the waves, her song lost to the surface world, while a mermaid dreams of the shore, longing for what lies beyond the water.",
    "The endless blue stretches far beyond what the eye can see, promising adventure and danger, yet the endless blue is never truly empty, filled with life unseen."
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

# Step 2: Initialize Embeddings
embedding_dim = 50  # Dimension size
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


# Step 4: Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)  # Softmax
    return np.dot(attention_weights, V), attention_weights


# Cross-Entropy Loss Function
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9))  # Small value to prevent log(0)


# One-hot encoding function
def one_hot_encoding(indices, vocab_size):
    one_hot = np.zeros((len(indices), vocab_size))
    for i, idx in enumerate(indices):
        one_hot[i, idx] = 1
    return one_hot


# Step 5: Training Loop
learning_rate = 0.01
epochs = 1000

# Add a transformation matrix from embedding space to vocab space
W_vocab = np.random.rand(embedding_dim, vocab_size) * 0.01  # (50, vocab_size)

for epoch in range(epochs):
    total_loss = 0
    random.shuffle(sentences)

    for sentence in sentences:
        words = sentence.lower().split()
        token_ids = [word_to_index[word] for word in words]

        # Extract embeddings & apply positional encoding
        X = word_embeddings[token_ids] + pos_encodings[:len(token_ids)]

        # Apply self-attention
        Q, K, V = X, X, X  # GPT-style (Q=K=V)
        attention_output, attention_weights = scaled_dot_product_attention(Q, K, V)

        # Project attention output to vocab space
        logits = np.dot(attention_output, W_vocab)  # Shape (seq_len, vocab_size)

        # Compute softmax over the projected output
        softmax_output = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

        # Convert original words to one-hot
        y_true = one_hot_encoding(token_ids, vocab_size)  # Shape (seq_len, vocab_size)

        # Compute cross-entropy loss
        loss = cross_entropy_loss(y_true, softmax_output)
        total_loss += loss

        # Backpropagation (Gradient update)
        gradients = softmax_output - y_true  # Shape (seq_len, vocab_size)
        dW_vocab = np.dot(attention_output.T, gradients)  # Gradient for W_vocab

        # Map gradient back to embedding space
        grad_embedding = np.dot(gradients, W_vocab.T)  # Shape (seq_len, embedding_dim)

        # Update embeddings
        for i, token_id in enumerate(token_ids):
            word_embeddings[token_id] -= learning_rate * grad_embedding[i]  # Now correctly shaped

        # Update transformation matrix
        W_vocab -= learning_rate * dW_vocab

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Step 6: Normalize Embeddings for Similarity Calculation
word_embeddings = word_embeddings / np.linalg.norm(word_embeddings, axis=1, keepdims=True)


# Step 7: Find Similar Words using Cosine Similarity
def get_similar_words(word, top_n=5):
    if word not in word_to_index:
        return "Word not in vocabulary"

    word_vec = word_embeddings[word_to_index[word]]
    similarities = np.dot(word_embeddings, word_vec)
    sorted_indices = np.argsort(-similarities)

    return [index_to_word[i] for i in sorted_indices[:top_n]]


while True:
    user_input = input("Enter a word: ").lower()
    print(f"Words similar to '{user_input}': {get_similar_words(user_input)}")
