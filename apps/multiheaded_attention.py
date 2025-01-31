import math
import torch
import torch.nn as nn

class MyMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        A custom multi-head attention layer implemented from scratch.

        Args:
            embed_dim (int): Dimensionality of the input embeddings (d_model).
            num_heads (int): Number of attention heads.

        The input to forward() should be of shape [batch_size, seq_len, embed_dim].
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, (
            "embed_dim must be divisible by num_heads."
        )

        # Each head will have dimensionality head_dim.
        self.head_dim = embed_dim // num_heads

        # Linear layers to project input into Q, K, V.
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Final linear layer to combine all heads' outputs.
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass for multi-head self-attention.

        Args:
            x (Tensor): Shape (batch_size, seq_len, embed_dim),
                        where embed_dim = d_model.

        Returns:
            Tensor of shape (batch_size, seq_len, embed_dim):
            The attention output for each position in the sequence.
        """
        # 1) Project the input x into queries, keys, and values.
        Q = self.W_q(x)  # (B, S, embed_dim)
        K = self.W_k(x)  # (B, S, embed_dim)
        V = self.W_v(x)  # (B, S, embed_dim)

        # 2) Reshape Q, K, V for multi-head attention.
        # We split embed_dim into (num_heads, head_dim).
        B, S, E = Q.shape  # batch_size, seq_len, embed_dim
        H = self.num_heads
        head_dim = self.head_dim

        # Reshape: (B, S, E) -> (B, S, H, head_dim) -> (B, H, S, head_dim)
        Q = Q.view(B, S, H, head_dim).transpose(1, 2)  # (B, H, S, head_dim)
        K = K.view(B, S, H, head_dim).transpose(1, 2)  # (B, H, S, head_dim)
        V = V.view(B, S, H, head_dim).transpose(1, 2)  # (B, H, S, head_dim)

        # 3) Scaled Dot-Product Attention per head.
        # scores shape: (B, H, S, S) because:
        #   Q: (B, H, S, head_dim)
        #   K^T: (B, H, head_dim, S)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

        # Apply softmax along the last dimension (the "sequence" dimension)
        attn_weights = torch.softmax(scores, dim=-1)  # (B, H, S, S)

        # Multiply by V: (B, H, S, S) x (B, H, S, head_dim) -> (B, H, S, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # 4) Reshape attn_output back to (B, S, E).
        attn_output = attn_output.transpose(1, 2)  # (B, S, H, head_dim)
        attn_output = attn_output.reshape(B, S, E) # (B, S, embed_dim)

        # 5) Apply the final linear projection.
        out = self.out_proj(attn_output)  # (B, S, embed_dim)
        return out