"""Generate sample parquet files for testing."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_stock_data(
    symbol: str,
    start_date: str = "2015-01-01",
    end_date: str = "2023-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic stock price data.

    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        seed: Random seed

    Returns:
        DataFrame with OHLCV data
    """
    rng = np.random.default_rng(seed + hash(symbol) % 1000)

    # Generate date range (business days)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)

    # Generate random walk for prices
    returns = rng.normal(0.0005, 0.02, n_days)  # 0.05% mean, 2% std
    log_prices = np.cumsum(returns)
    adjclose = 100.0 * np.exp(log_prices)

    # Generate OHLC from adjclose
    intraday_vol = rng.uniform(0.005, 0.02, n_days)
    open_price = adjclose * (1 + rng.normal(0, intraday_vol))
    high = np.maximum(open_price, adjclose) * (1 + rng.uniform(0, 0.01, n_days))
    low = np.minimum(open_price, adjclose) * (1 - rng.uniform(0, 0.01, n_days))
    close = adjclose  # Simplified: close = adjclose

    # Generate volume
    volume = rng.integers(1_000_000, 10_000_000, n_days)

    return pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'adjclose': adjclose,
        'volume': volume,
    })



def create_sample_parquet_files(output_dir: Path | str, n_stocks: int = 10) -> None:
    """Create sample parquet files for testing.

    Args:
        output_dir: Directory to save parquet files
        n_stocks: Number of stock files to create
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate sample symbols
    symbols = [f"STOCK{i:03d}" for i in range(n_stocks)]

    for symbol in symbols:
        df = generate_sample_stock_data(symbol)
        output_path = output_dir / f"{symbol}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Created {output_path}")

    print(f"\nCreated {n_stocks} sample parquet files in {output_dir}")


if __name__ == "__main__":
    # Create sample files
    output_dir = Path(__file__).parent.parent.parent.parent / "parquet_samples"
    create_sample_parquet_files(output_dir, n_stocks=10)
