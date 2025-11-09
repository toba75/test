"""Causal masking utilities for GPT model."""

import torch
from torch import Tensor


def create_causal_mask(seq_len: int, device: torch.device | None = None) -> Tensor:
    """Create a causal (triangular) attention mask.
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on. If None, uses CPU.
    
    Returns:
        Boolean mask of shape (seq_len, seq_len) where True means attend,
        False means mask out. The mask is lower triangular (including diagonal).
    """
    # Create lower triangular matrix (including diagonal)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask
