"""Portfolio backtesting for StockGPT."""

from typing import Any

import numpy as np
import pandas as pd

from .metrics import compute_performance_metrics


def backtest_long_short_deciles(
    predictions_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    top_pct: float = 10.0,
    bottom_pct: float = 10.0,
    min_stocks: int = 5,
) -> dict[str, Any]:
    """Backtest long/short decile portfolio.

    Args:
        predictions_df: DataFrame with [symbol, date, predicted_return]
        returns_df: DataFrame with [symbol, date, return]
        top_pct: Percentage for long portfolio (default 10%)
        bottom_pct: Percentage for short portfolio (default 10%)
        min_stocks: Minimum stocks required per side

    Returns:
        Dictionary with backtest results
    """
    # Merge predictions with actual returns
    merged = pd.merge(
        predictions_df,
        returns_df,
        on=['symbol', 'date'],
        how='inner',
        suffixes=('_pred', '_actual')
    )

    if len(merged) == 0:
        raise ValueError("No matching data between predictions and returns")

    # Group by date and compute portfolio returns
    daily_returns = []
    dates = []

    for date, group in merged.groupby('date'):
        n_stocks = len(group)

        if n_stocks < 2 * min_stocks:
            continue

        # Sort by predicted return
        sorted_group = group.sort_values('predicted_return', ascending=False)

        # Select top and bottom
        n_long = max(min_stocks, int(n_stocks * top_pct / 100))
        n_short = max(min_stocks, int(n_stocks * bottom_pct / 100))

        long_stocks = sorted_group.head(n_long)
        short_stocks = sorted_group.tail(n_short)

        # Equal-weighted returns
        long_return = long_stocks['return'].mean()
        short_return = short_stocks['return'].mean()

        # Long-short portfolio (net zero)
        portfolio_return = long_return - short_return

        daily_returns.append(portfolio_return)
        dates.append(date)

    # Compute metrics
    returns_array = np.array(daily_returns)
    metrics = compute_performance_metrics(returns_array, periods_per_year=252)

    return {
        'metrics': metrics,
        'daily_returns': pd.DataFrame({
            'date': dates,
            'return': daily_returns,
        }),
        'n_days': len(daily_returns),
    }



def create_backtest_report(results: dict[str, Any]) -> str:
    """Create text report from backtest results.

    Args:
        results: Results from backtest_long_short_deciles

    Returns:
        Formatted report string
    """
    metrics = results['metrics']

    report = [
        "=" * 60,
        "StockGPT Backtest Report",
        "=" * 60,
        "",
        "Performance Metrics:",
        f"  Total Return: {metrics['total_return']:.2%}",
        f"  Annualized Return: {metrics['annualized_return']:.2%}",
        f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
        f"  Volatility: {metrics['volatility']:.2%}",
        f"  Max Drawdown: {metrics['max_drawdown']:.2%}",
        "",
        f"Trading Days: {results['n_days']:,}",
        "=" * 60,
    ]

    return "\n".join(report)
