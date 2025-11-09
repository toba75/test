"""Inference module for StockGPT."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import autocast

from ..model.gpt import StockGPT
from ..tokens.discretizer import ReturnDiscretizer
from ..tokens.mapping import expected_return


class StockGPTPredictor:
    """Predictor for StockGPT model.

    Args:
        model: Trained StockGPT model
        discretizer: ReturnDiscretizer instance
        device: Device for inference
        use_amp: Whether to use automatic mixed precision
    """

    def __init__(
        self,
        model: StockGPT,
        discretizer: ReturnDiscretizer,
        device: torch.device | None = None,
        use_amp: bool = True,
    ) -> None:
        self.model = model
        self.discretizer = discretizer
        self.use_amp = use_amp

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict_next_return(
        self,
        history_tokens: Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict next return for a batch of sequences.

        Args:
            history_tokens: Historical tokens of shape (batch, seq_len)

        Returns:
            Tuple of (expected_returns, probabilities) where:
            - expected_returns: shape (batch,) - E[r] via midpoint averaging
            - probabilities: shape (batch, 402) - probability distribution over tokens
        """
        history_tokens = history_tokens.to(self.device)

        # Forward pass
        if self.use_amp:
            with autocast(dtype=torch.bfloat16):
                logits = self.model(history_tokens)
        else:
            logits = self.model(history_tokens)

        # Get logits for last position (next token prediction)
        next_logits = logits[:, -1, :]  # (batch, vocab_size)

        # Convert to probabilities
        probs = F.softmax(next_logits, dim=-1).cpu().numpy()

        # Calculate expected returns
        exp_returns = expected_return(probs, self.discretizer)

        return exp_returns, probs

    def predict_for_dataframe(
        self,
        returns_df: pd.DataFrame,
        seq_len: int = 256,
        batch_size: int = 128,
    ) -> pd.DataFrame:
        """Generate predictions for all stocks in a dataframe.

        Args:
            returns_df: DataFrame with columns [symbol, date, return]
            seq_len: Sequence length for context
            batch_size: Batch size for inference

        Returns:
            DataFrame with predictions [symbol, date, predicted_return]
        """
        # Group by symbol
        grouped = returns_df.groupby('symbol')

        all_predictions = []

        for symbol, group in grouped:
            group = group.sort_values('date').reset_index(drop=True)
            returns = group['return'].values
            dates = group['date'].values

            # Skip if not enough history
            if len(returns) < seq_len:
                continue

            # Generate predictions for each window
            predictions = []
            pred_dates = []

            for i in range(seq_len - 1, len(returns) - 1):
                # Get historical window
                window = returns[max(0, i - seq_len + 1):i + 1]

                # Pad if necessary
                if len(window) < seq_len:
                    padding = np.zeros(seq_len - len(window))
                    window = np.concatenate([padding, window])

                # Convert to tokens
                tokens = self.discretizer.returns_to_tokens(window)
                tokens_tensor = torch.from_numpy(tokens).long().unsqueeze(0)

                # Predict
                pred_return, _ = self.predict_next_return(tokens_tensor)

                predictions.append(pred_return[0])
                pred_dates.append(dates[i + 1])  # Prediction is for next day

            # Create dataframe for this symbol
            if predictions:
                symbol_df = pd.DataFrame({
                    'symbol': symbol,
                    'date': pred_dates,
                    'predicted_return': predictions,
                })
                all_predictions.append(symbol_df)

        # Combine all predictions
        if all_predictions:
            predictions_df = pd.concat(all_predictions, ignore_index=True)
            return predictions_df.sort_values(['date', 'symbol']).reset_index(drop=True)
        return pd.DataFrame(columns=['symbol', 'date', 'predicted_return'])


def load_model_for_inference(
    checkpoint_path: Path | str,
    device: torch.device | None = None,
) -> tuple[StockGPT, dict[str, Any]]:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, config)
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get configuration
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    # Create model
    from ..model.gpt import create_model
    model = create_model(
        vocab_size=model_config.get('vocab_size', 402),
        seq_len=model_config.get('seq_len', 256),
        d_model=model_config.get('d_model', 128),
        n_layers=model_config.get('n_layers', 4),
        n_heads=model_config.get('n_heads', 4),
        dropout=model_config.get('dropout', 0.2),
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Training step: {checkpoint.get('step', 'unknown')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'unknown')}")

    return model, config
