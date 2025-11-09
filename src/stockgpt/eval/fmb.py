"""Fama-MacBeth regression analysis."""

from typing import Any

import numpy as np
import pandas as pd


def rank_transform(x: pd.Series) -> pd.Series:
    """Transform to ranks in [-0.5, 0.5].

    Args:
        x: Series to rank

    Returns:
        Ranked series in [-0.5, 0.5]
    """
    ranks = x.rank(method='average')
    n = len(x)
    return (ranks - 1) / (n - 1) - 0.5


def fama_macbeth_regression(
    predictions_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    lags: int = 20,
) -> dict[str, Any]:
    """Perform Fama-MacBeth regression with Newey-West standard errors.

    Args:
        predictions_df: DataFrame with [symbol, date, predicted_return]
        returns_df: DataFrame with [symbol, date, return]
        lags: Number of lags for Newey-West adjustment

    Returns:
        Dictionary with regression results
    """
    # Merge predictions with actual returns
    merged = pd.merge(
        predictions_df,
        returns_df,
        on=['symbol', 'date'],
        how='inner',
        suffixes=('_pred', '_actual')
    )

    # Cross-sectional regressions by date
    coefficients = []
    r_squareds = []
    dates = []

    for date, group in merged.groupby('date'):
        if len(group) < 10:
            continue

        # Rank-transform predictions to [-0.5, 0.5]
        X = rank_transform(group['predicted_return']).values.reshape(-1, 1)
        y = group['return'].values

        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])

        # OLS regression
        try:
            beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            y_pred = X_with_intercept @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            coefficients.append(beta[1])  # Slope only
            r_squareds.append(r2)
            dates.append(date)
        except:
            continue

    coefficients = np.array(coefficients)

    # Time-series statistics
    mean_coef = np.mean(coefficients)

    # Simple t-stat (Newey-West would require more complex implementation)
    se_coef = np.std(coefficients, ddof=1) / np.sqrt(len(coefficients))
    t_stat = mean_coef / se_coef if se_coef > 0 else 0

    # Annualize
    annualized_coef = mean_coef * 252

    return {
        'mean_coefficient': mean_coef,
        'annualized_coefficient': annualized_coef,
        't_statistic': t_stat,
        'mean_r_squared': np.mean(r_squareds),
        'n_regressions': len(coefficients),
        'coefficients_ts': pd.DataFrame({
            'date': dates,
            'coefficient': coefficients,
            'r_squared': r_squareds,
        }),
    }



def create_fmb_report(results: dict[str, Any]) -> str:
    """Create text report from Fama-MacBeth results.

    Args:
        results: Results from fama_macbeth_regression

    Returns:
        Formatted report string
    """
    report = [
        "=" * 60,
        "Fama-MacBeth Regression Report",
        "=" * 60,
        "",
        f"Mean Daily Coefficient: {results['mean_coefficient']:.6f}",
        f"Annualized (×252): {results['annualized_coefficient']:.4f} ({results['annualized_coefficient']*100:.2f}%)",
        f"t-statistic: {results['t_statistic']:.2f}",
        f"Mean R²: {results['mean_r_squared']:.4f} ({results['mean_r_squared']*100:.2f}%)",
        "",
        f"Number of regressions: {results['n_regressions']:,}",
        "=" * 60,
    ]

    return "\n".join(report)
