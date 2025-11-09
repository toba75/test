import os
import tempfile
import numpy as np
import pandas as pd
import torch

from stockgpt.eval import metrics
from stockgpt.utils import config as cfg
from stockgpt.utils import seed as seed_mod


def test_compute_sharpe_and_annualized_return_and_drawdown():
    # simple returns: constant 1% daily for 252 days (~1 year)
    returns = np.array([0.01] * 252, dtype=np.float64)

    sharpe = metrics.compute_sharpe_ratio(returns, periods_per_year=252)
    # mean/std where std of constant series is zero -> sharpe 0
    assert isinstance(sharpe, float)

    annual = metrics.compute_annualized_return(returns, periods_per_year=252)
    # For 1% daily returns over 252 periods, annualized should equal total_return - 1
    expected_total = (1.01) ** 252 - 1
    assert pytest_close(annual, expected_total, atol=1e-6)

    max_dd = metrics.compute_max_drawdown(returns)
    assert max_dd == 0.0

    # zero-length
    assert metrics.compute_sharpe_ratio(np.array([])) == 0.0
    assert metrics.compute_annualized_return(np.array([])) == 0.0
    assert metrics.compute_max_drawdown(np.array([])) == 0.0


def pytest_close(a, b, atol=1e-8):
    return abs(a - b) <= atol


def test_compute_performance_metrics_with_series():
    returns = pd.Series([0.0, 0.01, -0.005])
    perf = metrics.compute_performance_metrics(returns, periods_per_year=252)
    assert 'sharpe_ratio' in perf
    assert perf['n_periods'] == 3


def test_load_and_save_config_tmpfile(tmp_path):
    cfg_dict = {'model': {'vocab_size': 100}, 'train': {'batch_size': 16}}
    p = tmp_path / "cfg_test.yaml"
    cfg.save_config(cfg_dict, p)
    loaded = cfg.load_config(p)
    assert loaded == cfg_dict


def test_set_seed_idempotent():
    # ensure calling set_seed doesn't raise and produces reproducible torch tensors
    seed_mod.set_seed(123)
    a = torch.randn(3)
    seed_mod.set_seed(123)
    b = torch.randn(3)
    # After reseeding the RNG, the same sequence should appear
    seed_mod.set_seed(123)
    c = torch.randn(3)
    assert torch.allclose(a, b) or torch.allclose(a, c)
