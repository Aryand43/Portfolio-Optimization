from src.data.fetch_data import get_cached_historical_data
from src.calculations.metrics import (
    calculate_sharpe_ratio,
    calculate_var,
    calculate_cvar,
    calculate_max_drawdown,
    calculate_omega_ratio
)
import pandas as pd

def analyze_real_data_metrics(tickers, start_date, end_date):
    """
    Fetch historical data and calculate portfolio metrics.
    """
    # Fetch historical data
    data = get_cached_historical_data("stocks", tickers, start_date, end_date)
    
    # Extract 'Close' prices from MultiIndex DataFrame
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data.xs('Close', axis=1, level=1)
    else:
        close_prices = data  # Single ticker scenario

    # Calculate returns
    returns = close_prices.pct_change().dropna()

    # Calculate metrics for each ticker
    metrics = {}
    for ticker in tickers:
        ticker_returns = returns[ticker]
        metrics[ticker] = {
            "Sharpe Ratio": round(calculate_sharpe_ratio(ticker_returns), 2),
            "VaR (95%)": round(calculate_var(ticker_returns), 4),
            "CVaR (95%)": round(calculate_cvar(ticker_returns), 4),
            "Max Drawdown": round(calculate_max_drawdown(close_prices[ticker]), 4),
            "Omega Ratio": round(calculate_omega_ratio(ticker_returns, threshold=0.01), 2)
        }
    
    return pd.DataFrame(metrics).T
