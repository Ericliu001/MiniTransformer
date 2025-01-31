import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# -------------------------
# 1. SAMPLE DATA
# -------------------------
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

# -------------------------
# 2. VOCAB BUILDING & TOKENIZATION
# -------------------------
def build_vocab(sentences, min_freq=1):
    """
    Build a vocabulary dictionary from the given sentences.
    - min_freq can help filter rare words if needed
    """
    word_freq = {}
    for sentence in sentences:
        for word in sentence.split():
            word_freq[word] = word_freq.get(word, 0) + 1

    # Filter by min_freq if desired
    words = [w for w, freq in word_freq.items() if freq >= min_freq]

    # Add special tokens
    special_tokens = ["<PAD>", "<UNK>"]
    vocab_list = special_tokens + sorted(words)  # Sorting ensures stable index order

    vocab_dict = {word: idx for idx, word in enumerate(vocab_list)}
    reverse_dict = {idx: word for word, idx in vocab_dict.items()}
    return vocab_dict, reverse_dict

def encode_sentences(sentences, vocab_dict):
    """
    Convert each sentence to a list of token IDs using vocab_dict.
    Unrecognized words become <UNK>.
    """
    unk_idx = vocab_dict["<UNK>"]
    encoded = []
    for sentence in sentences:
        tokens = []
        for word in sentence.split():
            tokens.append(vocab_dict.get(word, unk_idx))
        encoded.append(tokens)
    return encoded

def pad_sequences(sequences, pad_idx):
    """
    Pad all sequences to the same length (max length in the batch).
    """
    max_len = max(len(seq) for seq in sequences)
    padded_seqs = []
    for seq in sequences:
        padded = seq + [pad_idx] * (max_len - len(seq))
        padded_seqs.append(padded)
    return padded_seqs

# -------------------------
# 3. PYTORCH DATASET
# -------------------------
class TextDataset(Dataset):
    def __init__(self, encoded_seqs):
        """
        encoded_seqs is a list of integer lists.
        We'll be doing next-word prediction: each sequence is input[: -1], label[1 :].
        """
        self.data = encoded_seqs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        # Input is seq[:-1], target is seq[1:]
        input_ids = seq[:-1]
        target_ids = seq[1:]
        return torch.tensor(input_ids), torch.tensor(target_ids)

def collate_fn(batch):
    """
    Collate function to handle dynamic padding within a batch.
    We'll find the largest sequence in the batch and pad all accordingly.
    """
    # Separate input/target
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Find max len
    max_len = max(len(seq) for seq in inputs)

    # Pad
    padded_inputs = []
    padded_targets = []
    for inp, tgt in zip(inputs, targets):
        padded_inp = torch.cat([inp, torch.zeros(max_len - len(inp), dtype=torch.long)])
        padded_tgt = torch.cat([tgt, torch.zeros(max_len - len(tgt), dtype=torch.long)])
        padded_inputs.append(padded_inp)
        padded_targets.append(padded_tgt)

    return torch.stack(padded_inputs), torch.stack(padded_targets)

def create_dataloader(encoded_seqs, batch_size=4, shuffle=True):
    dataset = TextDataset(encoded_seqs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

# -------------------------
# 4. POSITIONAL ENCODING
# -------------------------
def positional_encoding(seq_len, d_model):
    """
    Creates a positional encoding matrix of shape [1, seq_len, d_model].
    """
    pos = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe.unsqueeze(0)

# -------------------------
# 5. MODEL DEFINITION
# -------------------------
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_seq_len):
        super(MiniTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = positional_encoding(max_seq_len, embedding_dim)  # shape [1, max_seq_len, d_model]

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True  # Let the input shape be (batch, seq_len, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final output layer
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len)
        We first embed x, add positional encodings, pass through encoder, then project to vocab size.
        """
        seq_len = x.shape[1]  # dynamic sequence length
        # Embedding + positional
        embeddings = self.embedding(x)  # shape: (batch, seq_len, embedding_dim)
        # Slice positional encoding to current seq_len
        pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
        out = embeddings + pos_enc  # shape: (batch, seq_len, embedding_dim)

        # Transformer encoder
        out = self.encoder(out)  # shape: (batch, seq_len, embedding_dim)

        # Final classifier
        out = self.fc_out(out)  # shape: (batch, seq_len, vocab_size)
        return out

# -------------------------
# 6. TRAINING LOOP
# -------------------------
def train_model(
        model,
        train_loader,
        val_loader,
        vocab_size,
        epochs=20,
        lr=1e-3,
        device="cpu"
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignoring padded tokens if 0 is <PAD>

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)  # (batch, seq_len, vocab_size)

            # Flatten for cross-entropy: B x S -> B*S
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Compute average train loss
        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# -------------------------
# 7. PUTTING IT ALL TOGETHER
# -------------------------
if __name__ == "__main__":
    # Build vocab & encode
    vocab_dict, reverse_vocab_dict = build_vocab(sentences)
    encoded = encode_sentences(sentences, vocab_dict)

    # Pad them all to the same size to determine max_seq_len for positional encoding
    pad_idx = vocab_dict["<PAD>"]
    padded = pad_sequences(encoded, pad_idx)
    max_seq_len = len(padded[0])  # after padding, each seq has this length

    # Train/Val split
    split_idx = int(len(padded) * 0.8)
    train_data = padded[:split_idx]
    val_data = padded[split_idx:]

    # Create DataLoaders
    train_loader = create_dataloader(train_data, batch_size=4, shuffle=True)
    val_loader = create_dataloader(val_data, batch_size=4, shuffle=False)

    # Define model hyperparams
    embedding_dim = 32
    num_heads = 2
    hidden_dim = 64
    num_layers = 2
    vocab_size = len(vocab_dict)

    # Initialize model
    model = MiniTransformer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )

    # Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_size=vocab_size,
        epochs=30,
        lr=1e-3,
        device="cpu"
    )

    # -------------------------
    # 8. INFERENCE EXAMPLE
    # -------------------------
    def predict_next_word(model, input_text, vocab_dict, reverse_vocab_dict, device="cpu"):
        """
        Predict the next token given an input text.
        """
        unk_idx = vocab_dict["<UNK>"]
        pad_idx = vocab_dict["<PAD>"]

        # Encode input
        tokens = []
        for w in input_text.split():
            tokens.append(vocab_dict.get(w, unk_idx))
        tokens = torch.tensor([tokens], dtype=torch.long).to(device)  # shape: (1, seq_len)

        model.eval()
        with torch.no_grad():
            logits = model(tokens)  # shape: (1, seq_len, vocab_size)
            next_token_logits = logits[0, -1]  # shape: (vocab_size,)
            next_idx = torch.argmax(next_token_logits).item()
            return reverse_vocab_dict[next_idx]

    # Test with a quick example
    while True:
        user_input = input("Enter a sentence fragment: ")
        if user_input.lower() == "exit":
            break
        predicted_word = predict_next_word(model, user_input, vocab_dict, reverse_vocab_dict)
        print(f"Input: '{user_input}' -> Next word prediction: '{predicted_word}'")