# StockGPT Quickstart Guide

This guide will help you get started with StockGPT in just a few minutes using the provided sample data.

## Prerequisites

- Python 3.10 or higher
- pip package manager

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stockgpt
```

2. Install dependencies:
```bash
make install
# or
pip install -e ".[dev]"
```

## Quick Test with Sample Data

The repository includes 10 sample stock files in `parquet_samples/` for testing.

### 1. Prepare Data

```bash
python scripts/prepare_crsp.py \
    --data-path parquet_samples \
    --output prepared/returns.parquet
```

Expected output:
```
Saved returns data to prepared/returns.parquet
Total returns: 23,460
Unique symbols: 10
Date range: 2015-01-02 to 2023-12-29
```

### 2. Train Model (Mini Version)

For a quick test, modify `configs/train.yaml`:
```yaml
train:
  batch_size: 8
  num_steps: 100  # Reduced from 10,000 for quick test
  ...
```

Then train:
```bash
python scripts/train_stockgpt.py \
    --config configs/train.yaml \
    --data prepared/returns.parquet
```

**Note:** Full training requires ~10,000 steps and may take several hours on CPU, or 30-60 minutes on GPU.

### 3. Generate Predictions

```bash
python scripts/predict.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --data prepared/returns.parquet \
    --output outputs/predictions.parquet
```

### 4. Backtest Results

```bash
python scripts/backtest_daily.py \
    --predictions outputs/predictions.parquet \
    --returns prepared/returns.parquet \
    --output outputs/backtest_report.txt
```

Expected output format:
```
============================================================
StockGPT Backtest Report
============================================================

Performance Metrics:
  Total Return: XX.XX%
  Annualized Return: XX.XX%
  Sharpe Ratio: X.XX
  Volatility: XX.XX%
  Max Drawdown: XX.XX%

Trading Days: X,XXX
============================================================
```

### 5. Fama-MacBeth Analysis

```bash
python scripts/eval_fmb.py \
    --predictions outputs/predictions.parquet \
    --returns prepared/returns.parquet \
    --output outputs/fmb_report.txt
```

## Using Real Data

### Data Format

Your data should be in parquet format with one file per symbol:

**Required columns:**
- `date`: Date (datetime)
- `adjclose`: Adjusted closing price (float)

**Optional columns:**
- `open`, `high`, `low`, `close`: OHLC prices
- `volume`: Trading volume

### Directory Structure

Place your parquet files in:
- **Unix/Linux/Mac**: `~/stockGPT/data/`
- **Windows**: `%USERPROFILE%\stockGPT\data\`

Or specify a custom path:
```bash
python scripts/prepare_crsp.py --data-path /path/to/your/data
```

## Configuration

### Training Configuration

Edit `configs/train.yaml` to customize:

```yaml
# Model size
model:
  vocab_size: 402
  seq_len: 256
  d_model: 128
  n_layers: 4
  n_heads: 4
  dropout: 0.2

# Training hyperparameters
train:
  batch_size: 64        # Reduce if OOM
  num_steps: 10000      # Full training
  learning_rate: 0.0003
  use_amp: true         # Mixed precision (bf16)

# Data splits
data:
  train_start: "1926-01-01"
  train_end: "2000-12-31"
  val_start: "1991-01-01"
  val_end: "2000-12-31"
  test_start: "2001-01-01"
  test_end: "2023-12-31"
```

### Portfolio Configuration

Edit `configs/backtest.yaml`:

```yaml
portfolio:
  top_pct: 10           # Long top 10%
  bottom_pct: 10        # Short bottom 10%
  min_price: 3.0        # Minimum stock price
  min_adv_real: 5000000 # Minimum daily volume ($5M)
```

## Expected Performance

From the paper (reference benchmarks):
- **Annualized Return**: ~119%
- **Sharpe Ratio**: ~6.5

**Note:** Actual results depend on:
- Data quality and coverage
- Model training convergence
- Filter specifications
- Market regime

## Common Issues

### Out of Memory (OOM)

Reduce batch size in `configs/train.yaml`:
```yaml
train:
  batch_size: 32  # or 16, 8
```

### Slow Training

Enable mixed precision (should be enabled by default):
```yaml
train:
  use_amp: true
```

Use GPU if available (detected automatically).

### No GPU Available

Training on CPU is slower but works. Consider:
- Using a smaller model (reduce `d_model`, `n_layers`)
- Training for fewer steps initially
- Using cloud GPU (Google Colab, AWS, etc.)

## Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Specific test file
pytest tests/unit/test_model.py -v
```

## Code Quality

```bash
# Linting
make lint

# Type checking
make typecheck

# Format code
make format

# All quality checks
make quality
```

## Documentation

- **Methodology**: `docs/README-methodo.md`
- **Website**: Open `www/minisite/index.html` in browser
- **Source Code**: Well-documented in `src/stockgpt/`

## Next Steps

1. ✅ Run quickstart with sample data
2. ✅ Verify all tests pass
3. 📊 Prepare your own data
4. 🚀 Train on full dataset
5. 📈 Analyze results
6. 🔬 Experiment with hyperparameters

## Getting Help

- Check `docs/README-methodo.md` for detailed methodology
- Review test files in `tests/` for usage examples
- Read docstrings in source code
- See AGENT.md for implementation specifications

## Citation

If you use this implementation, please cite:

```
Dat Mai (2025). "StockGPT: A GenAI Model for Stock Prediction and Trading"
```

## License

MIT License - See LICENSE file for details.
