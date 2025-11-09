#!/usr/bin/env python
"""Prepare CRSP data for training."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stockgpt.dataio.schemas import load_stock_data, prepare_returns_data
from stockgpt.utils.logging import setup_logger


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare stock returns data")
    parser.add_argument(
        '--data-path',
        type=str,
        default='~/stockGPT/data',
        help='Path to directory with parquet files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='prepared/returns.parquet',
        help='Output path for prepared returns'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='Optional list of symbols to process'
    )

    args = parser.parse_args()

    logger = setup_logger()

    # Load stock data
    logger.info(f"Loading stock data from {args.data_path}")
    stock_data = load_stock_data(args.data_path, symbols=args.symbols)
    logger.info(f"Loaded {len(stock_data)} stocks")

    # Prepare returns
    logger.info("Computing returns and preparing data")
    prepare_returns_data(stock_data, args.output)

    logger.info("Data preparation completed")


if __name__ == "__main__":
    main()
