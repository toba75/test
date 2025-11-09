# StockGPT: A GenAI Model for Stock Prediction and Trading

This project implements a transformer-based autoregressive model trained on daily stock returns, following the methodology described in "StockGPT: A GenAI Model for Stock Prediction and Trading" (Dat Mai, 2025).

## Overview

StockGPT is a GPT-style decoder that predicts future stock returns by learning patterns from historical return sequences. The model uses a discrete tokenization scheme (50 basis points bins) and outputs a distribution over 402 possible return tokens.

## Key Features

- **Tokenization**: 50 bps bins, 402 tokens, closed-right intervals, ±100% cap
- **Architecture**: GPT decoder with seq=256, d_model=128, 4 layers, 4 heads, ~0.93M parameters
- **Training**: Cross-entropy loss, batch=64, 10k steps, mixed precision (bf16)
- **Inference**: Expected return via midpoint-weighted averaging
- **Backtesting**: Daily rebalanced long/short decile portfolios (top 10% / bottom 10%)
- **Evaluation**: Fama-MacBeth regressions with Newey-West standard errors

## Installation

```bash
# Install dependencies
make install

# Or manually
pip install -e ".[dev]"
```

## Project Structure

```
stockgpt/
├── src/stockgpt/          # Main package
│   ├── dataio/            # Data schemas and I/O
│   ├── tokens/            # Tokenization (discretizer, mapping)
│   ├── model/             # GPT architecture
│   ├── train/             # Training pipeline
│   ├── infer/             # Inference and forecasting
│   ├── eval/              # Evaluation (portfolio, Fama-MacBeth)
│   └── utils/             # Utilities (seed, logging, config)
├── configs/               # YAML configuration files
├── scripts/               # CLI scripts
├── tests/                 # Unit and integration tests
├── docs/                  # Documentation and reference paper
├── www/minisite/          # Documentation website
└── parquet_samples/       # Sample data for testing
```

## Usage

### Data Preparation

```bash
make prepare
```

### Training

```bash
make train
```

### Prediction

```bash
make predict
```

### Backtesting

```bash
make backtest
```

### Fama-MacBeth Evaluation

```bash
make fmb
```

## Data Configuration

The project expects OHLCV data in parquet format (1 file per symbol) at:
- **Windows**: `%USERPROFILE%\stockGPT\data`
- **Unix**: `~/stockGPT/data`

Sample parquet files for testing are provided in `parquet_samples/`.

## Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run quality checks
make quality
```

## Documentation

For detailed methodology, see:
- `docs/README-methodo.md` - Complete methodology description
- `docs/ssrn-4787199.pdf` - Reference paper (source of truth)
- `www/minisite/` - Full documentation website

## Reference

This implementation follows the protocol described in:

> Dat Mai (2025). "StockGPT: A GenAI Model for Stock Prediction and Trading". 
> Available at: `docs/ssrn-4787199.pdf`

## License

MIT License
