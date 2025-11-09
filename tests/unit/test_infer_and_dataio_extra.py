import os
import tempfile
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from stockgpt.tokens.mapping import expected_return, sample_return
from stockgpt.tokens.discretizer import ReturnDiscretizer
from stockgpt.infer.forecast import StockGPTPredictor, load_model_for_inference
from stockgpt.model.gpt import create_model
from stockgpt.dataio import generate_samples, schemas


def test_mapping_defaults_and_sampling():
    # Call expected_return with None discretizer
    probs = np.ones((2, ReturnDiscretizer().n_tokens)) / ReturnDiscretizer().n_tokens
    exp = expected_return(probs)
    assert exp.shape == (2,)

    # sampling with default rng
    samples = sample_return(probs)
    assert samples.shape == (2,)


def test_create_sample_parquet_files(tmp_path):
    outdir = tmp_path / "parquets"
    generate_samples.create_sample_parquet_files(outdir, n_stocks=2)
    files = list(outdir.glob("*.parquet"))
    assert len(files) == 2


def test_load_stock_data_errors(tmp_path):
    # non-existent path
    with pytest_raises(FileNotFoundError):
        schemas.load_stock_data(tmp_path / "does_not_exist")

    # empty dir
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest_raises(ValueError):
        schemas.load_stock_data(empty)


def pytest_raises(exc):
    class _Ctx:
        def __init__(self, exc):
            self.exc = exc
        def __enter__(self):
            return None
        def __exit__(self, exc_type, exc, tb):
            assert exc_type is self.exc
            return True
    return _Ctx(exc)


def test_stockgpt_predictor_basic():
    # create a very small model
    model = create_model(vocab_size=ReturnDiscretizer().n_tokens, seq_len=16, d_model=32, n_layers=1, n_heads=4)
    disc = ReturnDiscretizer()
    # use CPU and no AMP for simplicity
    predictor = StockGPTPredictor(model=model, discretizer=disc, device=torch.device('cpu'), use_amp=False)

    # tokens shape (batch, seq_len)
    tokens = torch.zeros((1, 16), dtype=torch.long)
    exp_returns, probs = predictor.predict_next_return(tokens)
    assert exp_returns.shape == (1,)
    assert probs.shape == (1, disc.n_tokens)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_predict_for_dataframe_returns():
    # create returns df for one symbol with length > seq_len
    seq_len = 8
    n = 20
    dates = pd.bdate_range("2020-01-01", periods=n)
    returns = pd.Series(np.random.normal(0, 0.001, n))
    df = pd.DataFrame({'symbol': ['X'] * n, 'date': dates, 'return': returns})

    model = create_model(vocab_size=ReturnDiscretizer().n_tokens, seq_len=seq_len, d_model=32, n_layers=1, n_heads=4)
    disc = ReturnDiscretizer()
    predictor = StockGPTPredictor(model=model, discretizer=disc, device=torch.device('cpu'), use_amp=False)

    preds = predictor.predict_for_dataframe(df, seq_len=seq_len, batch_size=4)
    # predictions may be empty if sliding windows insufficient, but should be a DataFrame
    assert isinstance(preds, pd.DataFrame)


def test_load_model_for_inference_missing(tmp_path):
    # should raise FileNotFoundError for missing checkpoint
    with pytest_raises(FileNotFoundError):
        load_model_for_inference(tmp_path / "no.ckpt")

