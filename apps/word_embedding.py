import numpy as np
import random
from collections import defaultdict

# =============================================================================
# STEP 0: Prepare a Sample Dataset
# -----------------------------------------------------------------------------
# Here we define a list of sentences that serve as our "training" dataset.
# Each sentence describes a scene related to the ocean. In a real model,
# you would use a much larger and diverse dataset.
# =============================================================================

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

# =============================================================================
# STEP 1: Tokenization and Vocabulary Building
# -----------------------------------------------------------------------------
# We tokenize the sentences by splitting on spaces, convert words to lowercase,
# and count their frequency. Then we create mappings from words to indices and
# vice versa. This will allow us to convert words to numerical representations.
# =============================================================================

word_counts = defaultdict(int)
for sentence in sentences:
    for word in sentence.lower().split():
        word_counts[word] += 1

# Create a vocabulary list from the unique words
vocab = list(word_counts.keys())

# Create mappings between words and their corresponding indices
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for word, i in word_to_index.items()}

# Total number of unique words
vocab_size = len(vocab)

# =============================================================================
# STEP 2: Initialize Word Embeddings
# -----------------------------------------------------------------------------
# Each word is assigned a small random vector (embedding) of a specified dimension.
# These embeddings will be updated during training.
# =============================================================================

embedding_dim = 50  # Size of the embedding vector for each word
# Initialize embeddings with small random values
word_embeddings = np.random.rand(vocab_size, embedding_dim) * 0.01

# =============================================================================
# STEP 3: Positional Encoding
# -----------------------------------------------------------------------------
# Since the order of words in a sentence matters, we add positional information.
# This function creates a sinusoidal positional encoding matrix.
# =============================================================================

def positional_encoding(seq_length, embedding_dim):
    """
    Compute sinusoidal positional encodings.
    
    Args:
        seq_length (int): Maximum sequence length (number of words in a sentence).
        embedding_dim (int): Dimension of the embeddings.
    
    Returns:
        A (seq_length x embedding_dim) numpy array containing positional encodings.
    """
    pos_enc = np.zeros((seq_length, embedding_dim))
    for pos in range(seq_length):
        for i in range(0, embedding_dim, 2):
            # Use sine for even indices in the embedding
            pos_enc[pos, i] = np.sin(pos / (10000 ** (i / embedding_dim)))
            # Use cosine for odd indices in the embedding
            pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / embedding_dim)))
    return pos_enc

# Determine the maximum sentence length in the dataset
max_seq_length = max(len(sentence.split()) for sentence in sentences)
# Pre-compute positional encodings for sequences up to max_seq_length
pos_encodings = positional_encoding(max_seq_length, embedding_dim)

# =============================================================================
# STEP 4: Self-Attention Mechanism
# -----------------------------------------------------------------------------
# The scaled dot-product attention calculates attention weights between words
# and produces a new representation for each word based on all words in the sentence.
# In GPT, the same values are used for queries (Q), keys (K), and values (V).
# =============================================================================

def scaled_dot_product_attention(Q, K, V):
    """
    Compute the scaled dot-product attention.
    
    Args:
        Q (numpy.ndarray): Query matrix.
        K (numpy.ndarray): Key matrix.
        V (numpy.ndarray): Value matrix.
        
    Returns:
        A tuple of:
         - The attention output.
         - The attention weights.
    """
    d_k = Q.shape[-1]
    # Compute raw attention scores and scale them
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    # Apply softmax to obtain normalized attention weights
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    # Multiply weights by V to get the final attention output
    return np.dot(attention_weights, V), attention_weights

# =============================================================================
# Helper Functions for Loss and Encoding
# -----------------------------------------------------------------------------

def cross_entropy_loss(y_true, y_pred):
    """
    Compute cross-entropy loss.
    
    Args:
        y_true (numpy.ndarray): True one-hot encoded labels.
        y_pred (numpy.ndarray): Predicted probabilities.
        
    Returns:
        Scalar loss value.
    """
    # Add a small constant to prevent taking the log of 0
    return -np.sum(y_true * np.log(y_pred + 1e-9))

def one_hot_encoding(indices, vocab_size):
    """
    Convert a list of indices into one-hot encoded vectors.
    
    Args:
        indices (list): List of integer indices.
        vocab_size (int): Size of the vocabulary.
        
    Returns:
        A (number_of_indices x vocab_size) numpy array.
    """
    one_hot = np.zeros((len(indices), vocab_size))
    for i, idx in enumerate(indices):
        one_hot[i, idx] = 1
    return one_hot

# =============================================================================
# STEP 5: Training Loop
# -----------------------------------------------------------------------------
# In this simplified training loop, we update the word embeddings and a projection
# matrix that maps the attention outputs to vocabulary space. Note that in a real
# GPT model, you would have additional layers (e.g., multi-head attention, feed-forward
# layers, layer normalization, residual connections) and use an optimizer like Adam.
# =============================================================================

learning_rate = 0.01
epochs = 1000

# Transformation matrix to map from embedding space to vocabulary scores.
# This simulates the final linear layer in GPT that produces logits for each word.
W_vocab = np.random.rand(embedding_dim, vocab_size) * 0.01

for epoch in range(epochs):
    total_loss = 0
    # Shuffle sentences to introduce randomness during training
    random.shuffle(sentences)

    for sentence in sentences:
        # Tokenize the sentence: convert to lowercase and split into words
        words = sentence.lower().split()
        # Convert words into their corresponding token indices
        token_ids = [word_to_index[word] for word in words]

        # Retrieve the embeddings for the tokens and add positional encoding
        # (Only use as many positional encodings as there are tokens in the sentence)
        X = word_embeddings[token_ids] + pos_encodings[:len(token_ids)]

        # Apply self-attention (using Q = K = V in this simple example)
        Q, K, V = X, X, X
        attention_output, attention_weights = scaled_dot_product_attention(Q, K, V)

        # Project the attention output into the vocabulary space to obtain logits
        logits = np.dot(attention_output, W_vocab)  # Shape: (sequence_length, vocab_size)

        # Compute softmax probabilities from the logits
        softmax_output = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

        # Create one-hot encoded labels for the true tokens
        y_true = one_hot_encoding(token_ids, vocab_size)

        # Calculate the cross-entropy loss for the sentence
        loss = cross_entropy_loss(y_true, softmax_output)
        total_loss += loss

        # ---------------- Backpropagation ----------------
        # Compute the gradient of the loss with respect to the logits.
        # For cross-entropy with softmax, this gradient is (softmax_output - y_true).
        gradients = softmax_output - y_true

        # Compute gradient for the projection matrix W_vocab
        dW_vocab = np.dot(attention_output.T, gradients)

        # Backpropagate the gradient to the attention output (and therefore to the embeddings)
        grad_embedding = np.dot(gradients, W_vocab.T)

        # Update each word's embedding using the computed gradient.
        # Note: In a real model, gradients would be accumulated through all layers.
        for i, token_id in enumerate(token_ids):
            word_embeddings[token_id] -= learning_rate * grad_embedding[i]

        # Update the transformation matrix that projects to vocab space.
        W_vocab -= learning_rate * dW_vocab

    # Print the total loss every 100 epochs to monitor training progress.
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# =============================================================================
# STEP 6: Normalize the Word Embeddings
# -----------------------------------------------------------------------------
# Normalizing the embeddings allows us to compare them using cosine similarity.
# =============================================================================

word_embeddings = word_embeddings / np.linalg.norm(word_embeddings, axis=1, keepdims=True)

# =============================================================================
# STEP 7: Finding Similar Words
# -----------------------------------------------------------------------------
# This function calculates cosine similarity between the embedding of a given word
# and all other word embeddings to find and return the top N most similar words.
# =============================================================================

def get_similar_words(word, top_n=5):
    """
    Retrieve words most similar to the given word based on cosine similarity.
    
    Args:
        word (str): The input word.
        top_n (int): Number of similar words to return.
        
    Returns:
        List of words similar to the input word.
    """
    if word not in word_to_index:
        return "Word not in vocabulary"

    # Get the normalized embedding of the input word
    word_vec = word_embeddings[word_to_index[word]]
    # Compute cosine similarities (dot product works because vectors are normalized)
    similarities = np.dot(word_embeddings, word_vec)
    # Get indices of the most similar words (sorted in descending order)
    sorted_indices = np.argsort(-similarities)

    return [index_to_word[i] for i in sorted_indices[:top_n]]

# =============================================================================
# Interactive Loop for Testing the Model
# -----------------------------------------------------------------------------
# The loop below allows the user to input a word and see a list of similar words.
# =============================================================================

while True:
    user_input = input("Enter a word: ").lower()
    print(f"Words similar to '{user_input}': {get_similar_words(user_input)}")