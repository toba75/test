"""GPT-style decoder model for stock return prediction.

This module implements a lightweight GPT decoder following the StockGPT paper:
- Sequence length: 256
- Embedding dimension: 128
- Number of layers: 4
- Number of heads: 4
- Dropout: 0.2
- Vocabulary size: 402 tokens
- Total parameters: ~0.93M

Reference: docs/ssrn-4787199.pdf
"""

import torch
import torch.nn as nn
from torch import Tensor

from .blocks import TransformerBlock
from .mask import create_causal_mask


class StockGPT(nn.Module):
    """GPT-style decoder for stock return prediction.

    Args:
        vocab_size: Size of token vocabulary (402)
        seq_len: Maximum sequence length (256)
        d_model: Model dimension (128)
        n_layers: Number of transformer blocks (4)
        n_heads: Number of attention heads (4)
        dropout: Dropout probability (0.2)
    """

    def __init__(
        self,
        vocab_size: int = 402,
        seq_len: int = 256,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layers = n_layers

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Positional embeddings (learned)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        d_ff = 4 * d_model  # Feed-forward dimension
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)

        # Output projection to vocabulary
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Create and register causal mask (fixed for all sequences)
        # Register as buffer so it moves with model to different devices
        self.register_buffer(
            'causal_mask',
            create_causal_mask(seq_len),
            persistent=False
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"StockGPT initialized with {n_params:,} parameters")

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following GPT-2 style."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, tokens: Tensor) -> Tensor:
        """Forward pass.

        Args:
            tokens: Token indices of shape (batch, seq_len)

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = tokens.shape
        assert seq_len <= self.seq_len, f"Sequence length {seq_len} exceeds maximum {self.seq_len}"

        # Token embeddings
        tok_emb = self.token_emb(tokens)  # (batch, seq_len, d_model)

        # Positional embeddings
        positions = torch.arange(seq_len, dtype=torch.long, device=tokens.device)
        pos_emb = self.pos_emb(positions)  # (seq_len, d_model)

        # Combine embeddings
        x = self.dropout(tok_emb + pos_emb)

        # Get causal mask for this sequence length
        mask = self.causal_mask[:seq_len, :seq_len]

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        return self.head(x)  # (batch, seq_len, vocab_size)


    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    vocab_size: int = 402,
    seq_len: int = 256,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    dropout: float = 0.2,
) -> StockGPT:
    """Create a StockGPT model with default or custom parameters.

    Args:
        vocab_size: Size of token vocabulary (default: 402)
        seq_len: Maximum sequence length (default: 256)
        d_model: Model dimension (default: 128)
        n_layers: Number of transformer blocks (default: 4)
        n_heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.2)

    Returns:
        Initialized StockGPT model
    """
    return StockGPT(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
    )
