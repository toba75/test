"""Performance metrics for evaluation."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def compute_sharpe_ratio(
    returns: NDArray[np.float64],
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sharpe ratio.

    Args:
        returns: Array of period returns
        periods_per_year: Number of periods per year (252 for daily)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    if std_return == 0:
        return 0.0

    sharpe = mean_return / std_return * np.sqrt(periods_per_year)

    return float(sharpe)


def compute_annualized_return(
    returns: NDArray[np.float64],
    periods_per_year: int = 252,
) -> float:
    """Compute annualized return.

    Args:
        returns: Array of period returns
        periods_per_year: Number of periods per year

    Returns:
        Annualized return as decimal
    """
    if len(returns) == 0:
        return 0.0

    total_return = np.prod(1 + returns)
    n_periods = len(returns)
    n_years = n_periods / periods_per_year

    annualized = total_return ** (1 / n_years) - 1

    return float(annualized)


def compute_max_drawdown(
    returns: NDArray[np.float64],
) -> float:
    """Compute maximum drawdown.

    Args:
        returns: Array of period returns

    Returns:
        Maximum drawdown as positive decimal
    """
    if len(returns) == 0:
        return 0.0

    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max

    max_dd = float(np.min(drawdowns))

    return abs(max_dd)


def compute_performance_metrics(
    returns: NDArray[np.float64] | pd.Series,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute comprehensive performance metrics.

    Args:
        returns: Array or Series of period returns
        periods_per_year: Number of periods per year

    Returns:
        Dictionary of performance metrics
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    return {
        'total_return': float(np.prod(1 + returns) - 1),
        'annualized_return': compute_annualized_return(returns, periods_per_year),
        'sharpe_ratio': compute_sharpe_ratio(returns, periods_per_year),
        'volatility': float(np.std(returns, ddof=1) * np.sqrt(periods_per_year)),
        'max_drawdown': compute_max_drawdown(returns),
        'n_periods': len(returns),
    }

