import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

# Tokenization (simple)
vocab = set(word for sentence in sentences for word in sentence.split())
vocab.add("<PAD>")  # Explicitly add a padding token
vocab_size = len(vocab)
vocab_dict = {word: i for i, word in enumerate(vocab)}
reverse_vocab_dict = {i: word for word, i in vocab_dict.items()}

# Convert sentences to sequences
sequences = [[vocab_dict[word] for word in sentence.split()] for sentence in sentences]

# Define hyperparameters
embedding_dim = 32
num_heads = 2
hidden_dim = 64
num_layers = 2
max_seq_len = max(len(seq) for seq in sequences)

# Padding sequences to max length
for seq in sequences:
    while len(seq) < max_seq_len:
        seq.append(vocab_dict["<PAD>"])  # Using "." as padding token

tensor_data = torch.tensor(sequences)

def positional_encoding(seq_len, d_model):
    """Creates a positional encoding matrix."""
    pos = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_seq_len):
        super(MiniTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = positional_encoding(max_seq_len, embedding_dim)
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=0.1),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.shape[1], :]
        x = self.encoder_layers(x)
        x = self.fc_out(x)
        return x

# Model instance
model = MiniTransformer(vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_seq_len)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop (simple example)
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(tensor_data[:, :-1])  # Input sequence
    loss = criterion(output.reshape(-1, vocab_size), tensor_data[:, 1:].reshape(-1))  # Predict next word
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Predict the next word based on user input
def predict_next_word(input_text):
    tokens = [vocab_dict.get(word, 0) for word in input_text.split()]
    tokens = torch.tensor(tokens).unsqueeze(0)  # Convert to tensor
    with torch.no_grad():
        output = model(tokens)
        next_word_idx = output[0, -1].argmax().item()
        return reverse_vocab_dict[next_word_idx]

# Example prediction
while True:
    user_input = input("Enter a sentence fragment: ")
    if user_input.lower() == "exit":
        break
    print("Predicted next word:", predict_next_word(user_input))
