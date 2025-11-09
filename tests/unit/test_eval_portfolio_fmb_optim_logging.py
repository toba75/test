import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytest

from stockgpt.eval import fmb, portfolio
from stockgpt.utils import logging as logging_utils
from stockgpt.train import optim
from stockgpt.infer.forecast import load_model_for_inference
from stockgpt.model.gpt import create_model
from stockgpt.train.loop import Trainer
from torch.utils.data import DataLoader, TensorDataset
from stockgpt.tokens.discretizer import ReturnDiscretizer


def test_rank_transform_basic():
    s = pd.Series([3, 1, 2])
    ranked = fmb.rank_transform(s)
    assert ranked.min() >= -0.5 and ranked.max() <= 0.5


def test_fama_macbeth_regression_and_report():
    # Create synthetic predictions and returns for 3 dates, 12 symbols each
    dates = pd.date_range('2020-01-01', periods=3)
    rows_pred = []
    rows_ret = []
    for d in dates:
        for i in range(12):
            sym = f'S{i}'
            rows_pred.append({'symbol': sym, 'date': d, 'predicted_return': float(i)})
            rows_ret.append({'symbol': sym, 'date': d, 'return': float(i) + np.random.normal(0, 1)})
    preds = pd.DataFrame(rows_pred)
    rets = pd.DataFrame(rows_ret)

    results = fmb.fama_macbeth_regression(preds, rets)
    assert 'mean_coefficient' in results
    report = fmb.create_fmb_report(results)
    assert 'Fama-MacBeth' in report


def test_backtest_long_short_deciles_and_report():
    # create a single date with 20 symbols
    date = pd.Timestamp('2020-01-01')
    rows_pred = []
    rows_ret = []
    for i in range(20):
        sym = f'S{i}'
        pred = float(i)
        ret = float(i % 5) * 0.01  # some pattern
        rows_pred.append({'symbol': sym, 'date': date, 'predicted_return': pred})
        rows_ret.append({'symbol': sym, 'date': date, 'return': ret})
    preds = pd.DataFrame(rows_pred)
    rets = pd.DataFrame(rows_ret)

    res = portfolio.backtest_long_short_deciles(preds, rets)
    assert 'metrics' in res
    report = portfolio.create_backtest_report(res)
    assert 'Backtest' in report


def test_setup_logger_file(tmp_path):
    log_file = tmp_path / 'logs' / 'test.log'
    logger = logging_utils.setup_logger(name='testlogger', level=20, log_file=log_file)
    # Should have at least one handler (console) and file handler
    assert any(isinstance(h, logging_utils.logging.FileHandler) for h in logger.handlers)
    # log file should be created when logging
    logger.info('hello')
    assert log_file.exists()


def test_create_optimizer_and_scheduler():
    model = create_model(vocab_size=ReturnDiscretizer().n_tokens, seq_len=16, d_model=32, n_layers=1, n_heads=4)
    opt = optim.create_optimizer(model)
    assert hasattr(opt, 'step')

    sched = optim.create_cosine_schedule_with_warmup(opt, num_warmup_steps=2, num_training_steps=10)
    # step scheduler a few times
    prev = sched.get_last_lr()[0]
    for i in range(5):
        opt.step()
        sched.step()
    cur = sched.get_last_lr()[0]
    assert cur >= 0.0


def test_load_model_for_inference_with_checkpoint(tmp_path):
    # create small model and save checkpoint
    vocab = ReturnDiscretizer().n_tokens
    model = create_model(vocab_size=vocab, seq_len=8, d_model=32, n_layers=1, n_heads=4)
    ckpt = {
        'model_state_dict': model.state_dict(),
        'config': {'model': {'vocab_size': vocab, 'seq_len': 8, 'd_model': 32, 'n_layers': 1, 'n_heads': 4, 'dropout': 0.1}},
        'step': 5,
        'best_val_loss': 0.5,
    }
    p = tmp_path / 'ckpt.pt'
    torch.save(ckpt, p)

    loaded_model, cfg = load_model_for_inference(p, device=torch.device('cpu'))
    assert isinstance(loaded_model, torch.nn.Module)
    assert 'model' in cfg


def test_trainer_save_checkpoint_and_evaluate(tmp_path):
    # small model and tiny dataloaders
    seq_len = 8
    vocab = ReturnDiscretizer().n_tokens
    model = create_model(vocab_size=vocab, seq_len=seq_len, d_model=32, n_layers=1, n_heads=4)

    # create dummy dataset: batch size 2, one batch
    x = torch.zeros((2, seq_len), dtype=torch.long)
    y = torch.zeros((2, seq_len), dtype=torch.long)
    train_loader = DataLoader(TensorDataset(x, y), batch_size=2)
    val_loader = DataLoader(TensorDataset(x, y), batch_size=2)

    config = {'num_steps': 1, 'warmup_steps': 1, 'use_amp': False}
    trainer = Trainer(model, train_loader, val_loader, config, device=torch.device('cpu'))

    ckdir = tmp_path / 'ck'
    trainer.save_checkpoint(ckdir / 'step_0.pt', is_best=True)
    assert (ckdir / 'step_0.pt').exists()
    assert (ckdir / 'best_model.pt').exists()

    # evaluate should return a float
    val = trainer.evaluate()
    assert isinstance(val, float)

