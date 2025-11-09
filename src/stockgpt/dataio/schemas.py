"""Data schemas and validation for StockGPT."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from typing import Any


@dataclass
class StockData:
    """Container for stock OHLCV data.
    
    Attributes:
        symbol: Stock ticker symbol
        date: Date column
        open: Opening price
        high: High price
        low: Low price
        close: Closing price (unadjusted)
        adjclose: Adjusted closing price
        volume: Trading volume
    """
    symbol: str
    df: pd.DataFrame
    
    def __post_init__(self) -> None:
        """Validate data schema."""
        required_cols = ['date', 'adjclose']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Sort by date
        self.df = self.df.sort_values('date').reset_index(drop=True)
    
    def compute_returns(self) -> pd.Series:
        """Compute daily returns from adjusted close prices.
        
        Returns:
            Series of daily returns (NaN for first day)
        """
        returns = self.df['adjclose'].pct_change()
        return returns
    
    @classmethod
    def from_parquet(cls, path: Path, symbol: str | None = None) -> "StockData":
        """Load stock data from parquet file.
        
        Args:
            path: Path to parquet file
            symbol: Optional symbol name (inferred from filename if not provided)
        
        Returns:
            StockData instance
        """
        if symbol is None:
            symbol = path.stem
        
        df = pd.read_parquet(path)
        return cls(symbol=symbol, df=df)


def load_stock_data(data_path: Path | str, symbols: list[str] | None = None) -> dict[str, StockData]:
    """Load stock data from directory of parquet files.
    
    Args:
        data_path: Path to directory containing parquet files
        symbols: Optional list of symbols to load. If None, loads all.
    
    Returns:
        Dictionary mapping symbol to StockData
    """
    data_path = Path(data_path).expanduser()
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # Find parquet files
    parquet_files = list(data_path.glob("*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_path}")
    
    # Filter by symbols if provided
    if symbols is not None:
        symbol_set = set(symbols)
        parquet_files = [f for f in parquet_files if f.stem in symbol_set]
    
    # Load data
    stock_data = {}
    for file_path in parquet_files:
        try:
            stock = StockData.from_parquet(file_path)
            stock_data[stock.symbol] = stock
        except Exception as e:
            print(f"Warning: Failed to load {file_path.name}: {e}")
    
    return stock_data


def prepare_returns_data(
    stock_data: dict[str, StockData],
    output_path: Path | str,
) -> pd.DataFrame:
    """Prepare returns data for training.
    
    Computes returns for all stocks and saves to a single parquet file.
    
    Args:
        stock_data: Dictionary of StockData instances
        output_path: Path to save prepared returns data
    
    Returns:
        DataFrame with columns: symbol, date, return
    """
    all_returns = []
    
    for symbol, stock in stock_data.items():
        returns = stock.compute_returns()
        df_returns = pd.DataFrame({
            'symbol': symbol,
            'date': stock.df['date'],
            'return': returns
        })
        # Drop first row (NaN return)
        df_returns = df_returns.dropna(subset=['return'])
        all_returns.append(df_returns)
    
    # Combine all returns
    returns_df = pd.concat(all_returns, ignore_index=True)
    returns_df = returns_df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # Save to parquet
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    returns_df.to_parquet(output_path, index=False)
    
    print(f"Saved returns data to {output_path}")
    print(f"Total returns: {len(returns_df):,}")
    print(f"Unique symbols: {returns_df['symbol'].nunique():,}")
    print(f"Date range: {returns_df['date'].min()} to {returns_df['date'].max()}")
    
    return returns_df
