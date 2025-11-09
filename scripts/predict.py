#!/usr/bin/env python
"""Generate predictions with trained StockGPT model."""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stockgpt.infer.forecast import StockGPTPredictor, load_model_for_inference
from stockgpt.tokens.discretizer import ReturnDiscretizer
from stockgpt.utils.logging import setup_logger


def main() -> None:
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Generate StockGPT predictions")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='prepared/returns.parquet',
        help='Path to returns data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/predictions.parquet',
        help='Output path for predictions'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for inference'
    )

    args = parser.parse_args()

    logger = setup_logger()

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model, config = load_model_for_inference(args.checkpoint)

    # Create discretizer
    discretizer = ReturnDiscretizer()

    # Create predictor
    predictor = StockGPTPredictor(
        model=model,
        discretizer=discretizer,
        use_amp=True,
    )

    # Load data
    logger.info(f"Loading returns data from {args.data}")
    returns_df = pd.read_parquet(args.data)

    # Filter to test period
    test_start = config.get('data', {}).get('test_start', '2001-01-01')
    test_end = config.get('data', {}).get('test_end', '2023-12-31')
    returns_df = returns_df[
        (returns_df['date'] >= pd.to_datetime(test_start)) &
        (returns_df['date'] <= pd.to_datetime(test_end))
    ]

    logger.info(f"Generating predictions for {len(returns_df):,} observations")

    # Generate predictions
    predictions_df = predictor.predict_for_dataframe(
        returns_df=returns_df,
        seq_len=config.get('model', {}).get('seq_len', 256),
        batch_size=args.batch_size,
    )

    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(output_path, index=False)

    logger.info(f"Saved predictions to {output_path}")
    logger.info(f"Total predictions: {len(predictions_df):,}")


if __name__ == "__main__":
    main()
