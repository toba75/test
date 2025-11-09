"""Transformer blocks for GPT architecture."""

import torch
import torch.nn as nn
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking.

    Args:
        d_model: Model dimension (embedding size)
        n_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional causal mask of shape (seq_len, seq_len)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        # (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_head)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        # (batch, n_heads, seq_len, d_head) @ (batch, n_heads, d_head, seq_len)
        # -> (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, d_head)
        # -> (batch, n_heads, seq_len, d_head)
        out = torch.matmul(attn_weights, v)

        # Reshape back to (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        return self.out_proj(out)



class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    Args:
        d_model: Model dimension
        d_ff: Hidden dimension (typically 4 * d_model)
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear2(x)


class TransformerBlock(nn.Module):
    """Transformer decoder block with pre-norm architecture.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass with pre-norm and residual connections.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional causal mask of shape (seq_len, seq_len)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Pre-norm + attention + residual
        attn_out = self.attn(self.ln1(x), mask)
        x = x + self.dropout(attn_out)

        # Pre-norm + FFN + residual
        ffn_out = self.ffn(self.ln2(x))
        return x + self.dropout(ffn_out)

