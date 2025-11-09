#!/usr/bin/env python
"""Perform Fama-MacBeth regression analysis."""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stockgpt.eval.fmb import create_fmb_report, fama_macbeth_regression
from stockgpt.utils.logging import setup_logger


def main() -> None:
    """Main Fama-MacBeth function."""
    parser = argparse.ArgumentParser(description="Fama-MacBeth regression")
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
        default='outputs/fmb_report.txt',
        help='Output path for report'
    )
    parser.add_argument(
        '--lags',
        type=int,
        default=20,
        help='Lags for Newey-West adjustment'
    )

    args = parser.parse_args()

    logger = setup_logger()

    # Load data
    logger.info(f"Loading predictions from {args.predictions}")
    predictions_df = pd.read_parquet(args.predictions)

    logger.info(f"Loading returns from {args.returns}")
    returns_df = pd.read_parquet(args.returns)

    # Run Fama-MacBeth
    logger.info("Running Fama-MacBeth regression")
    results = fama_macbeth_regression(
        predictions_df=predictions_df,
        returns_df=returns_df,
        lags=args.lags,
    )

    # Create report
    report = create_fmb_report(results)

    # Print to console
    print("\n" + report)

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)

    logger.info(f"Saved report to {output_path}")

    # Save coefficient time series
    coef_path = output_path.parent / 'fmb_coefficients.parquet'
    results['coefficients_ts'].to_parquet(coef_path, index=False)
    logger.info(f"Saved coefficient time series to {coef_path}")


if __name__ == "__main__":
    main()
