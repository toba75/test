# StockGPT Methodology

## Overview

StockGPT is a transformer-based autoregressive model for predicting stock returns. This document describes the complete methodology following the protocol outlined in "StockGPT: A GenAI Model for Stock Prediction and Trading" (Dat Mai, 2025).

**Reference**: `docs/ssrn-4787199.pdf`

## 1. Return Tokenization

### Discretization Scheme

Stock returns are discretized into 402 tokens using the following protocol:

- **Bin Width**: 50 basis points (bps)
- **Total Tokens**: 402 (indices 0-401)
- **Range**: ±10,000 bps (±100%)
- **Interval Type**: Closed-right intervals

### Token Mapping

```
Token 0:       (-∞, -10000] bps (extreme negative)
Token 1-200:   (-10000, 0] bps in 50 bps increments
Token 200:     (-50, 0] bps (contains 0%)
Token 201-400: (0, 10000] bps in 50 bps increments  
Token 401:     (10000, +∞) bps (extreme positive)
```

### Example

From the paper:
- Input returns: -2.4%, 0%, 0%, 5%, 4.8%
- Tokens: [196, 200, 200, 210, 210]

## 2. Model Architecture

### GPT Decoder Configuration

- **Sequence Length**: 256 tokens
- **Embedding Dimension**: 128
- **Number of Layers**: 4
- **Attention Heads**: 4
- **Dropout**: 0.2
- **Feed-Forward**: 4× hidden size (512)
- **Total Parameters**: ~930,000

### Architecture Details

```
Input: Token sequence (batch, 256)
  ↓
Token Embedding (vocab=402, d=128)
  +
Positional Embedding (seq=256, d=128)
  ↓
Dropout (p=0.2)
  ↓
Transformer Block × 4:
  - Pre-LayerNorm
  - Multi-Head Attention (4 heads, causal mask)
  - Residual Connection
  - Pre-LayerNorm
  - Feed-Forward (d→512→d, GELU)
  - Residual Connection
  ↓
Final LayerNorm
  ↓
Output Projection (d→402)
  ↓
Softmax → Probability distribution over tokens
```

## 3. Training

### Data Splits

- **Training**: 1926-2000
- **Validation**: 1991-2000 (for hyperparameter tuning)
- **Test**: 2001-2023

### Training Configuration

- **Loss Function**: Cross-entropy (classification over 402 tokens)
- **Batch Size**: 64
- **Training Steps**: 10,000
- **Optimizer**: AdamW
  - Learning rate: 3×10⁻⁴
  - Weight decay: 0.01
  - β₁=0.9, β₂=0.999
- **Learning Rate Schedule**: Cosine with warmup
  - Warmup steps: 500
  - Minimum LR ratio: 0.1
- **Gradient Clipping**: 1.0
- **Mixed Precision**: bfloat16 (AMP)

### Sampling Strategy

Sequences are sampled with probability proportional to the available history length per stock. This ensures that stocks with longer histories contribute more to training.

### Expected Training Behavior

- Cross-entropy loss should decrease substantially
- Typical final validation loss: ~2.5 (order of magnitude)
- Training should stabilize around 5,000-10,000 steps

## 4. Inference

### Prediction Protocol

1. Given historical return sequence of length 256
2. Forward pass through model
3. Get probability distribution over 402 tokens for next return
4. Calculate expected return: **E[r] = Σ pᵢ · midpoint(tokenᵢ)**

### Midpoint Calculation

For each token, the midpoint is the center of its interval:
- Token 200: midpoint of (-50, 0] = -25 bps = -0.0025
- Token 201: midpoint of (0, 50] = 25 bps = 0.0025
- Token 210: midpoint of (450, 500] = 475 bps = 0.0475

## 5. Portfolio Construction

### Daily Long-Short Strategy

**At close of day t:**

1. Rank all stocks by predicted E[r_{t+1}]
2. Apply filters:
   - Exclude stocks with insufficient liquidity (see deviation below)
   - Optional: minimum price filter ($3-5)
3. Select:
   - **Long**: Top 10% of stocks (equal-weighted)
   - **Short**: Bottom 10% of stocks (equal-weighted)
4. Portfolio return: r_portfolio = r_long - r_short (net zero)

### Deviation from Paper: Liquidity Filter

**Paper Protocol**: Filter by market capitalization (≥ 10th percentile)

**Our Implementation** (due to lack of market cap data):
- Construct point-in-time liquidity score using:
  - ADV_60: 60-day rolling median dollar volume
  - ILLIQ: Monthly Amihud illiquidity measure
  - CS_spread: Cross-sectional bid-ask spread (if available)
- Deflate ADV by CPI (constant 2025 dollars)
- Exclude bottom 10% by liquidity score
- Exclude stocks below minimum ADV threshold (e.g., $5M)

### Expected Performance

Paper reports (equal-weighted, market cap filter):
- **Annualized Return**: ~119%
- **Sharpe Ratio**: ~6.5

Note: These are reference benchmarks. Actual performance depends on:
- Data quality
- Transaction costs (not modeled)
- Model training convergence
- Filter specifications

## 6. Evaluation

### Fama-MacBeth Regression

**Cross-sectional regression (daily):**

```
rᵢ,ₜ₊₁ = αₜ + βₜ · xᵢ,ₜ + εᵢ,ₜ
```

Where:
- rᵢ,ₜ₊₁: Realized return for stock i at t+1
- xᵢ,ₜ: Rank-transformed prediction in [-0.5, 0.5]

**Time-series aggregation:**
- Mean coefficient: β̄ = mean(βₜ)
- Standard errors: Newey-West with 20 lags
- Annualized: β̄_annual = β̄ × 252

**Expected Results:**
- Positive and significant β̄
- t-statistic >> 2
- Example from paper: ~158% annualized
- Mean R²: 0.5-1%

### Performance Metrics

- **Sharpe Ratio**: Annualized
- **Alpha**: CAPM/Fama-French 5-factor (if factor data available)
- **Maximum Drawdown**
- **Total/Annualized Return**

## 7. Reproducibility

### Seeds Fixed

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

### MLflow Tracking

- Model checkpoints versioned
- Hyperparameters logged
- Training metrics tracked
- Artifacts saved (configs, predictions)

## 8. Limitations

1. **Transaction Costs**: Not modeled; real-world implementation requires careful cost analysis
2. **Retraining**: Model trained once on historical data; production systems should retrain periodically
3. **Liquidity Filter**: Using volume-based proxy instead of market cap
4. **Slippage**: Daily rebalancing assumes perfect execution at close
5. **Survivorship Bias**: Depends on data source

## 9. Extensions

Potential enhancements not in base implementation:
- Walk-forward analysis (multiple train/test splits)
- Multi-horizon predictions (2-day, 5-day, etc.)
- Factor-augmented models
- Transaction cost optimization
- Value-weighted portfolios

## References

1. Dat Mai (2025). "StockGPT: A GenAI Model for Stock Prediction and Trading". Available at: `docs/ssrn-4787199.pdf`
2. Vaswani et al. (2017). "Attention is All You Need"
3. Radford et al. (2019). "Language Models are Unsupervised Multitask Learners"

---

For implementation details, see the source code in `src/stockgpt/` and scripts in `scripts/`.
