"""Dataset for training StockGPT.

Implements sequence sampling with probability proportional to history length.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..tokens.discretizer import ReturnDiscretizer


class StockReturnsDataset(Dataset):
    """Dataset for stock return sequences.

    Samples random sequences of length seq_len from stock histories,
    with probability proportional to the available history length.

    Args:
        returns_df: DataFrame with columns [symbol, date, return]
        discretizer: ReturnDiscretizer instance
        seq_len: Sequence length (256)
        split: Data split ('train', 'val', or 'test')
        start_date: Start date for split
        end_date: End date for split
    """

    def __init__(
        self,
        returns_df: pd.DataFrame,
        discretizer: ReturnDiscretizer,
        seq_len: int = 256,
        split: str = 'train',
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        self.returns_df = returns_df.copy()
        self.discretizer = discretizer
        self.seq_len = seq_len
        self.split = split

        # Filter by date range if provided
        if start_date is not None:
            self.returns_df = self.returns_df[
                self.returns_df['date'] >= pd.to_datetime(start_date)
            ]
        if end_date is not None:
            self.returns_df = self.returns_df[
                self.returns_df['date'] <= pd.to_datetime(end_date)
            ]

        # Group by symbol and store return sequences
        self._prepare_sequences()

        # Compute sampling weights (proportional to history length)
        self._compute_weights()

    def _prepare_sequences(self) -> None:
        """Prepare return sequences grouped by symbol."""
        grouped = self.returns_df.groupby('symbol')

        self.sequences = []
        self.symbols = []

        for symbol, group in grouped:
            returns = group['return'].values
            if len(returns) >= self.seq_len:
                self.sequences.append(returns)
                self.symbols.append(symbol)

        print(f"{self.split} split: {len(self.sequences)} stocks with sufficient history")

    def _compute_weights(self) -> None:
        """Compute sampling weights proportional to sequence length."""
        lengths = np.array([len(seq) for seq in self.sequences])
        # Weight proportional to number of possible windows
        # For sequence of length n, can sample (n - seq_len + 1) windows
        weights = np.maximum(lengths - self.seq_len + 1, 0)
        weights = weights.astype(np.float64)
        weights = weights / weights.sum()
        self.weights = weights

    def __len__(self) -> int:
        """Return dataset size.

        For training, we use a large virtual size to enable
        repeated sampling from the same stocks.
        """
        if self.split == 'train':
            # Virtual size for epoch-based training
            return 100000
        # For validation, sample each stock once
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a random sequence sample.

        Returns:
            Tuple of (input_tokens, target_tokens) where:
            - input_tokens: shape (seq_len,)
            - target_tokens: shape (seq_len,) - shifted by 1 for next-token prediction
        """
        if self.split == 'train':
            # Sample stock with probability proportional to history length
            stock_idx = np.random.choice(len(self.sequences), p=self.weights)
        else:
            # For validation, cycle through stocks
            stock_idx = idx % len(self.sequences)

        # Get returns for this stock
        returns = self.sequences[stock_idx]

        # Sample random window of length seq_len
        max_start = len(returns) - self.seq_len
        start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0

        window = returns[start_idx:start_idx + self.seq_len]

        # Convert to tokens
        tokens = self.discretizer.returns_to_tokens(window)

        # Create input and target sequences
        # Input: tokens[:-1], Target: tokens[1:]
        # But we'll use the full sequence and let the model handle it
        input_tokens = torch.from_numpy(tokens).long()
        target_tokens = torch.from_numpy(tokens).long()

        return input_tokens, target_tokens


def create_dataloaders(
    returns_df: pd.DataFrame,
    discretizer: ReturnDiscretizer,
    batch_size: int = 64,
    seq_len: int = 256,
    train_start: str = "1926-01-01",
    train_end: str = "2000-12-31",
    val_start: str = "1991-01-01",
    val_end: str = "2000-12-31",
    num_workers: int = 0,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders.

    Args:
        returns_df: DataFrame with returns data
        discretizer: ReturnDiscretizer instance
        batch_size: Batch size
        seq_len: Sequence length
        train_start: Training start date
        train_end: Training end date
        val_start: Validation start date
        val_end: Validation end date
        num_workers: Number of dataloader workers

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = StockReturnsDataset(
        returns_df=returns_df,
        discretizer=discretizer,
        seq_len=seq_len,
        split='train',
        start_date=train_start,
        end_date=train_end,
    )

    val_dataset = StockReturnsDataset(
        returns_df=returns_df,
        discretizer=discretizer,
        seq_len=seq_len,
        split='val',
        start_date=val_start,
        end_date=val_end,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
