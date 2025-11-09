#!/usr/bin/env python
"""Train StockGPT model."""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stockgpt.model.gpt import create_model
from stockgpt.tokens.discretizer import ReturnDiscretizer
from stockgpt.train.dataset import create_dataloaders
from stockgpt.train.loop import Trainer
from stockgpt.utils.config import load_config
from stockgpt.utils.logging import setup_logger
from stockgpt.utils.seed import set_seed


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train StockGPT")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='prepared/returns.parquet',
        help='Path to prepared returns data'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(Path(args.config))

    # Setup logging
    logger = setup_logger(log_file=Path(config['output']['log_dir']) / 'train.log')

    # Set seed for reproducibility
    seed = config['train']['seed']
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # Load data
    logger.info(f"Loading returns data from {args.data}")
    returns_df = pd.read_parquet(args.data)
    logger.info(f"Loaded {len(returns_df):,} return observations")

    # Create discretizer
    discretizer = ReturnDiscretizer()

    # Create dataloaders
    logger.info("Creating dataloaders")
    train_loader, val_loader = create_dataloaders(
        returns_df=returns_df,
        discretizer=discretizer,
        batch_size=config['train']['batch_size'],
        seq_len=config['model']['seq_len'],
        train_start=config['data']['train_start'],
        train_end=config['data']['train_end'],
        val_start=config['data']['val_start'],
        val_end=config['data']['val_end'],
    )

    # Create model
    logger.info("Creating model")
    model = create_model(
        vocab_size=config['model']['vocab_size'],
        seq_len=config['model']['seq_len'],
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        dropout=config['model']['dropout'],
    )

    # Create trainer
    logger.info("Initializing trainer")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['train'],
    )

    # Train
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training")
    trainer.train(checkpoint_dir)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
