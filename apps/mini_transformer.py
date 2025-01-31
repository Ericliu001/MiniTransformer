import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define some sample sentences (dummy dataset for quick training)
sentences = [
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

# Tokenization: Create a vocabulary set from the sentences
vocab = set(word for sentence in sentences for word in sentence.split())
vocab.add("<PAD>")  # Add a padding token
vocab_size = len(vocab)

# Create a mapping from words to indices and vice versa
vocab_dict = {word: i for i, word in enumerate(vocab)}
reverse_vocab_dict = {i: word for word, i in vocab_dict.items()}

# Convert sentences to sequences of token indices
sequences = [[vocab_dict[word] for word in sentence.split()] for sentence in sentences]

# Define hyperparameters
embedding_dim = 32  # Size of word embeddings
num_heads = 2  # Number of attention heads in Transformer
hidden_dim = 64  # Hidden layer size in feedforward network
num_layers = 2  # Number of Transformer encoder layers
max_seq_len = max(len(seq) for seq in sequences)  # Find max sequence length for padding

# Pad sequences to max length using the padding token index
for seq in sequences:
    while len(seq) < max_seq_len:
        seq.append(vocab_dict["<PAD>"])

# Convert sequences to a PyTorch tensor
tensor_data = torch.tensor(sequences)

# Function to create positional encoding

def positional_encoding(seq_len, d_model):
    """Creates a positional encoding matrix."""
    pos = torch.arange(seq_len).unsqueeze(1)  # Position indices
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))  # Scaling factor
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div_term)  # Apply sine to even indices
    pe[:, 1::2] = torch.cos(pos * div_term)  # Apply cosine to odd indices
    return pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)

# Define a simple Transformer model class
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_seq_len):
        super(MiniTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Word embedding layer
        self.pos_encoding = positional_encoding(max_seq_len, embedding_dim)  # Positional encoding
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=0.1),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)  # Output layer for classification

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.shape[1], :]  # Add positional encoding
        x = self.encoder_layers(x)  # Pass through Transformer encoder
        x = self.fc_out(x)  # Output logits for each word in the vocabulary
        return x

# Instantiate model, optimizer, and loss function
model = MiniTransformer(vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_seq_len)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
criterion = nn.CrossEntropyLoss()  # Loss function for classification

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(tensor_data[:, :-1])  # Input sequence (excluding last word)
    loss = criterion(output.reshape(-1, vocab_size), tensor_data[:, 1:].reshape(-1))  # Predict next word
    loss.backward()  # Compute gradients
    optimizer.step()  # Update model parameters
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Function to predict the next word based on user input
def predict_next_word(input_text):
    tokens = [vocab_dict.get(word, 0) for word in input_text.split()]  # Tokenize input text
    tokens = torch.tensor(tokens).unsqueeze(0)  # Convert to tensor
    with torch.no_grad():
        output = model(tokens)  # Generate model output
        next_word_idx = output[0, -1].argmax().item()  # Get the most likely next word index
        return reverse_vocab_dict[next_word_idx]  # Convert back to word

# Interactive loop to test predictions
while True:
    user_input = input("Enter a sentence fragment: ")
    if user_input.lower() == "exit":
        break
    print("Predicted next word:", predict_next_word(user_input))
