import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from torch.nn.utils.rnn import pad_sequence

# 1️⃣ Create a Tiny Dataset (200 words)
sentences = [
    "I love machine learning",
    "Deep learning is powerful",
    "Transformers are great for NLP",
    "PyTorch is easy to use",
    "The cat sat on the mat",
    "AI is the future of technology",
    "We are building a small transformer",
    "This model predicts the next word",
    "Natural language processing is cool",
    "Data science is an exciting field",
    "Neural networks can recognize patterns",
    "Reinforcement learning helps in robotics",
    "Language models improve text generation",
    "Self-attention captures word relationships",
    "Optimizers like Adam speed up training",
    "Large datasets improve model accuracy",
    "Speech recognition converts voice to text",
    "GPT models understand human language",
    "RNNs process sequences step by step",
    "BERT learns deep contextual embeddings",
    "Computers analyze images using CNNs",
    "Gradient descent optimizes neural networks",
    "Tokenization breaks text into smaller pieces",
    "Pretraining enhances model performance",
    "Machine translation converts languages",
    "Data augmentation improves robustness",
    "Overfitting happens with small datasets",
    "Zero-shot learning generalizes to new tasks",
    "Attention mechanisms revolutionized NLP",
    "Recurrent networks struggle with long-term dependencies",
    "Transformer models parallelize computation",
    "Convolutional layers detect image features",
    "Supervised learning needs labeled data",
    "Unsupervised learning finds hidden patterns",
    "Semi-supervised learning uses some labels",
    "Batch normalization stabilizes training",
    "Dropout prevents neural networks from overfitting",
    "Transfer learning fine-tunes models",
    "Meta-learning enables learning to learn",
    "Loss functions measure model performance",
    "Activation functions introduce non-linearity",
    "Softmax turns logits into probabilities",
    "Regularization techniques prevent overfitting",
    "Autoregressive models predict sequential data",
    "Masked language models learn bidirectional context",
    "Self-supervised learning leverages unlabeled data",
    "Feature extraction identifies important information",
    "Neural architectures evolve over time",
    "Graph neural networks analyze relationships",
    "Computational efficiency matters in deep learning",
    "Embedding layers map words into vector space",
    "Fine-tuning adapts models to specific tasks",
    "End-to-end learning minimizes manual engineering",
    "Beam search improves sequence generation",
    "Sparse attention scales to long sequences",
    "Contrastive learning distinguishes representations",
    "Knowledge distillation transfers model efficiency",
    "Latent representations capture hidden patterns",
    "Data pipelines automate preprocessing",
    "Ethical AI ensures responsible usage",
    "Multi-modal learning integrates different data types",
    "OpenAI advances artificial intelligence research",
    "Quantum computing might enhance machine learning",
    "Neurosymbolic AI combines logic and learning",
    "Adaptive learning rates accelerate convergence",
    "Neural embeddings capture semantic meaning",
    "Transformers outperform traditional models",
    "Clustering techniques group similar data points",
    "Curriculum learning structures training",
    "Explainable AI improves interpretability",
    "Hyperparameter tuning optimizes performance",
    "Evolutionary algorithms simulate natural selection",
    "Self-organizing maps cluster data naturally",
    "LSTMs handle longer sequences than RNNs",
    "CNNs extract hierarchical features",
    "Adversarial training strengthens model robustness",
    "Gradient clipping prevents exploding gradients",
    "Weight initialization affects model stability",
    "Self-play enables reinforcement learning agents",
    "Robotics combines AI and engineering",
    "Autonomous systems require adaptive learning",
    "Sim-to-real transfer improves robotic applications",
    "Neural rendering synthesizes realistic images",
    "3D vision advances depth perception",
    "Multi-task learning improves efficiency",
    "Uncertainty estimation enhances AI reliability",
    "Bayesian neural networks quantify uncertainty",
    "Efficient Transformers reduce computational cost",
    "Token embeddings capture word meaning",
    "Masked pretraining improves contextual understanding",
    "Energy-based models predict probability distributions",
    "Sparsity constraints improve model efficiency",
    "Neuroevolution optimizes neural networks",
    "Automated machine learning simplifies model selection",
    "Knowledge graphs store structured relationships",
    "Deep reinforcement learning powers game AI",
    "Multi-agent systems simulate real-world interactions",
    "Memory-augmented networks retain information",
    "Symbolic regression finds mathematical expressions",
    "Graph embeddings capture structural relationships",
    "Attention heads focus on relevant information",
    "Spiking neural networks mimic biological neurons",
    "Neural differential equations model continuous processes"
]


# Tokenization: Convert words into unique indices
words = set(" ".join(sentences).split())  # Extract unique words
word2idx = {word: idx for idx, word in enumerate(words)}  # Assign unique ID to each word
idx2word = {idx: word for word, idx in word2idx.items()}  # Reverse mapping for predictions
vocab_size = len(word2idx)  # Vocabulary size

# Convert sentences into sequences of token IDs
dataset = []
for sentence in sentences:
    tokens = [word2idx[word] for word in sentence.split()]
    for i in range(len(tokens) - 1):  # Create (input, target) pairs
        dataset.append((tokens[:i+1], tokens[i+1]))

# Custom Dataset Loader with Padding
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

# Collate function for padding sequences
def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence([torch.tensor(seq).clone().detach() for seq in inputs], batch_first=True, padding_value=0)
    targets = torch.tensor(targets, dtype=torch.long)
    return inputs, targets

# Load dataset with dynamic padding
dataloader = DataLoader(TextDataset(dataset), batch_size=2, shuffle=True, collate_fn=collate_fn)

# 2️⃣ Define the Mini Transformer Model
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=16, num_heads=2, num_layers=1):
        super().__init__()

        # Embedding Layer: Converts token indices into dense vectors
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional Encoding: Helps retain word order information
        self.positional_encoding = nn.Parameter(torch.randn(1, 10, d_model))  # Fixed max length = 10

        # Transformer Encoder: Applies self-attention and feedforward layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True),
            num_layers
        )

        # Fully Connected Layer: Predicts the next word from transformer output
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x)  # Apply transformer encoder layers
        return self.fc(x[:, -1, :])  # Use only last token for next-word prediction

# Initialize model
model = MiniTransformer(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 3️⃣ Train the Transformer Model
for epoch in range(10):
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# 4️⃣ Interactive User Input for Next-Word Prediction
def predict_next_word(input_text):
    """Predicts the next word based on user input using the trained model."""
    tokens = [word2idx[word] for word in input_text.split() if word in word2idx]
    x = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        output = model(x)
        predicted_word = idx2word[torch.argmax(output).item()]
    return predicted_word

# Ask for user input in a loop
while True:
    user_input = input("Enter a phrase (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    print("Predicted next word:", predict_next_word(user_input))
