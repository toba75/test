#!/usr/bin/env python
"""Backtest StockGPT predictions."""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stockgpt.eval.portfolio import backtest_long_short_deciles, create_backtest_report
from stockgpt.utils.logging import setup_logger


def main() -> None:
    """Main backtesting function."""
    parser = argparse.ArgumentParser(description="Backtest StockGPT")
    parser.add_argument(
        '--predictions',
        type=str,
        default='outputs/predictions.parquet',
        help='Path to predictions file'
    )
    parser.add_argument(
        '--returns',
        type=str,
        default='prepared/returns.parquet',
        help='Path to returns data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/backtest_report.txt',
        help='Output path for report'
    )
    parser.add_argument(
        '--top-pct',
        type=float,
        default=10.0,
        help='Top percentile for long portfolio'
    )
    parser.add_argument(
        '--bottom-pct',
        type=float,
        default=10.0,
        help='Bottom percentile for short portfolio'
    )

    args = parser.parse_args()

    logger = setup_logger()

    # Load data
    logger.info(f"Loading predictions from {args.predictions}")
    predictions_df = pd.read_parquet(args.predictions)

    logger.info(f"Loading returns from {args.returns}")
    returns_df = pd.read_parquet(args.returns)

    # Run backtest
    logger.info("Running backtest")
    results = backtest_long_short_deciles(
        predictions_df=predictions_df,
        returns_df=returns_df,
        top_pct=args.top_pct,
        bottom_pct=args.bottom_pct,
    )

    # Create report
    report = create_backtest_report(results)

    # Print to console
    print("\n" + report)

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)

    logger.info(f"Saved report to {output_path}")

    # Save daily returns
    returns_path = output_path.parent / 'daily_returns.parquet'
    results['daily_returns'].to_parquet(returns_path, index=False)
    logger.info(f"Saved daily returns to {returns_path}")


if __name__ == "__main__":
    main()
