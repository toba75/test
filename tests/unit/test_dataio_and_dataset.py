import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import torch

from stockgpt.dataio import schemas, generate_samples
from stockgpt.train.dataset import StockReturnsDataset, create_dataloaders
from stockgpt.tokens.discretizer import ReturnDiscretizer


def make_sample_df(symbol="TEST", n=300):
    dates = pd.bdate_range("2020-01-01", periods=n)
    adj = 100 * (1 + np.cumsum(np.random.normal(0, 0.001, n)))
    df = pd.DataFrame({
        'date': dates,
        'adjclose': adj,
        'open': adj * 0.99,
        'high': adj * 1.01,
        'low': adj * 0.98,
        'close': adj,
        'volume': np.random.randint(1000, 10000, n)
    })
    return df


def test_stockdata_and_prepare_returns(tmp_path):
    df = make_sample_df(n=10)
    symbol = "S1"
    stock = schemas.StockData(symbol=symbol, df=df)
    returns = stock.compute_returns()
    assert len(returns) == 10

    # prepare returns data with two stocks
    sd = {symbol: stock, 'S2': schemas.StockData(symbol='S2', df=make_sample_df(n=12))}
    out = tmp_path / "out" / "returns.parquet"
    res = schemas.prepare_returns_data(sd, out)
    assert 'symbol' in res.columns
    assert res['symbol'].nunique() == 2


def test_generate_sample_stock_data_basic():
    df = generate_samples.generate_sample_stock_data('ABC', start_date='2021-01-01', end_date='2021-01-10')
    assert 'adjclose' in df.columns
    assert len(df) > 0


def test_dataset_and_dataloader_basic():
    # create returns df for two symbols with >seq_len returns
    n = 260
    s1 = make_sample_df('S1', n=n)
    s2 = make_sample_df('S2', n=n)

    # create returns DataFrame
    def df_to_returns(df, symbol):
        r = df['adjclose'].pct_change()
        r = r.dropna().reset_index(drop=True)
        return pd.DataFrame({'symbol': symbol, 'date': df['date'].iloc[1:].values, 'return': r.values})

    r1 = df_to_returns(s1, 'S1')
    r2 = df_to_returns(s2, 'S2')
    returns_df = pd.concat([r1, r2], ignore_index=True)

    discretizer = ReturnDiscretizer()
    # use date ranges that include the generated data (2020)
    train_loader, val_loader = create_dataloaders(
        returns_df,
        discretizer,
        batch_size=2,
        seq_len=128,
        num_workers=0,
        train_start="2019-01-01",
        train_end="2021-12-31",
        val_start="2019-01-01",
        val_end="2021-12-31",
    )
    # iterate one batch
    for xb, yb in train_loader:
        assert xb.shape[0] <= 2
        break
