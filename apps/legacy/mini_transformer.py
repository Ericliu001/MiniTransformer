# -------------------------
# 1. SAMPLE DATA
# -------------------------
data = [
    "i love machine learning",
    "transformers are powerful",
    "this is a transformer model",
    "deep learning is amazing",
    "attention is all you need",
    "pytorch is great for ai",
    "natural language processing is fun",
    "the future of ai is transformers",
    "chatbots are becoming smarter",
    "neural networks mimic the brain",
    "big data drives innovation",
    "cloud computing enables scalability",
    "artificial intelligence is evolving",
    "gradient descent optimizes models",
    "vector embeddings capture meaning",
    "self supervised learning is promising",
    "computer vision detects objects",
    "reinforcement learning learns policies",
    "data science is an exciting field",
    "speech recognition is improving",
    "generative ai creates content",
    "large language models understand context",
    "zero shot learning is impressive",
    "transformers revolutionized ai",
    "mathematics is fundamental to ai",
    "programming requires logical thinking",
    "python is widely used in ai",
    "autonomous vehicles use deep learning",
    "gpu acceleration speeds up training",
    "ethics in ai is important",
    "data preprocessing improves accuracy",
    "multi modal learning integrates information",
    "convolutional networks analyze images",
    "hyperparameter tuning improves models",
    "artificial general intelligence is a goal",
    "meta learning adapts to new tasks",
    "probabilistic models handle uncertainty",
    "bayesian inference updates beliefs",
    "sequence to sequence models translate text",
    "few shot learning requires less data",
    "unsupervised learning finds patterns",
    "tokenization prepares text for models",
    "long short term memory networks handle sequences",
    "backpropagation trains neural networks",
    "decision trees are interpretable",
    "ensemble methods improve predictions",
    "text embeddings capture semantics",
    "contrastive learning improves representations",
    "fine tuning adapts pre trained models"
]

import numpy as np


class MiniTransformer:
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, lr=0.01):
        """Initializes a simplified Transformer model."""
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.lr = lr  # Learning rate

        # Initialize token embeddings (random small values)
        self.token_embeddings = np.random.randn(vocab_size, d_model) * 0.01

        # Positional Encoding: Adds positional information
        self.positional_encoding = self.create_positional_encoding(100, d_model)

        # Initialize weights for multi-head attention and feed-forward layers
        self.attention_weights = [self.init_attention() for _ in range(num_layers)]
        self.ffn_weights = [self.init_ffn() for _ in range(num_layers)]

    def create_positional_encoding(self, max_length, d_model):
        """Creates positional encoding using sine and cosine functions."""
        pos = np.arange(max_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((max_length, d_model))
        pe[:, 0::2] = np.sin(pos * div_term)
        pe[:, 1::2] = np.cos(pos * div_term)
        return pe

    def init_attention(self):
        """Initializes weight matrices for self-attention."""
        return {
            'W_q': np.random.randn(self.d_model, self.d_model) * 0.01,
            'W_k': np.random.randn(self.d_model, self.d_model) * 0.01,
            'W_v': np.random.randn(self.d_model, self.d_model) * 0.01,
            'W_o': np.random.randn(self.d_model, self.d_model) * 0.01
        }

    def init_ffn(self):
        """Initializes feed-forward network weights."""
        return {
            'W1': np.random.randn(self.d_model, self.d_ff) * 0.01,
            'W2': np.random.randn(self.d_ff, self.d_model) * 0.01
        }

    def scaled_dot_product_attention(self, Q, K, V):
        """Computes the scaled dot-product attention."""
        d_k = Q.shape[-1]
        scores = np.dot(Q, K.T) / np.sqrt(d_k)
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        return np.dot(attention_weights, V)

    def multi_head_attention(self, x, layer):
        """Computes multi-head self-attention for a Transformer layer."""
        W_q, W_k, W_v, W_o = self.attention_weights[layer].values()
        Q, K, V = np.dot(x, W_q), np.dot(x, W_k), np.dot(x, W_v)
        attn_out = self.scaled_dot_product_attention(Q, K, V)
        return np.dot(attn_out, W_o)

    def feed_forward(self, x, layer):
        """Computes feed-forward network transformation."""
        W1, W2 = self.ffn_weights[layer].values()
        return np.dot(np.maximum(0, np.dot(x, W1)), W2)

    def forward(self, x):
        """Passes input through Transformer layers."""
        x = self.token_embeddings[x] + self.positional_encoding[:len(x)]
        for layer in range(self.num_layers):
            x = self.multi_head_attention(x, layer) + x  # Residual connection
            x = self.feed_forward(x, layer) + x  # Another residual connection
        return x

    def train(self, data, epochs=50):
        """
        Trains the token embeddings using a simple self-supervised approach.

        - Uses mean squared error (MSE) loss between predicted and actual next word embeddings.
        - Updates embeddings via gradient descent.
        """
        for epoch in range(epochs):
            total_loss = 0

            for sentence in data:
                words = sentence.split()
                for i in range(len(words) - 1):  # Predict next word in sequence
                    input_word = words[i]
                    target_word = words[i + 1]

                    if input_word not in word_to_index or target_word not in word_to_index:
                        continue  # Skip unknown words

                    input_id = word_to_index[input_word]
                    target_id = word_to_index[target_word]

                    # Forward pass
                    predicted_embedding = self.forward([input_id])[-1]

                    # Compute loss (MSE between predicted and actual embedding)
                    actual_embedding = self.token_embeddings[target_id]
                    loss = np.mean((predicted_embedding - actual_embedding) ** 2)
                    total_loss += loss

                    # Compute gradient and update embedding
                    grad = 2 * (predicted_embedding - actual_embedding)
                    self.token_embeddings[input_id] -= self.lr * grad

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    def softmax(self, x):
        """Computes softmax function for a vector."""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)

    def predict_next(self, input_text):
        """
        Predicts the next word using trained embeddings and softmax sampling.
        """
        input_ids = [word_to_index.get(word, 0) for word in input_text.split()]
        output = self.forward(input_ids)

        last_embedding = output[-1]  # Last token embedding

        # Compute similarity scores
        similarities = np.dot(self.token_embeddings, last_embedding)

        # Apply softmax to similarity scores to get probabilities
        probabilities = self.softmax(similarities)

        # Sample next word based on probability distribution
        next_word_id = np.random.choice(self.vocab_size, p=probabilities)

        return index_to_word[next_word_id]


# Tokenization
unique_words = sorted(set(word for sentence in data for word in sentence.split()))
word_to_index = {word: i for i, word in enumerate(unique_words)}
index_to_word = {i: word for word, i in word_to_index.items()}

# Model parameters
vocab_size = len(unique_words)
d_model = 16
num_heads = 2
num_layers = 2
d_ff = 32

# Initialize and train the model
transformer = MiniTransformer(vocab_size, d_model, num_heads, num_layers, d_ff)
transformer.train(data, epochs=100)

# Test the model
while True:
    user_input = input("Enter a sentence fragment: ")
    if user_input.lower() == "exit":
        break
    predicted_word = transformer.predict_next(user_input)
    print(f"Input: '{user_input}' -> Next word prediction: '{predicted_word}'")
