import numpy as np
import pandas as pd
from src.calculations.metrics import (
    calculate_sharpe_ratio,
    calculate_var,
    calculate_cvar,
    calculate_max_drawdown,
    calculate_omega_ratio
)

def monte_carlo_with_metrics(returns, num_simulations=1000, time_horizon=252, initial_portfolio=10000, threshold=0.01):
    """
    Perform Monte Carlo simulations and calculate metrics.

    Parameters:
        returns (pd.DataFrame): Asset returns.
        num_simulations (int): Number of simulations to run.
        time_horizon (int): Number of days to simulate.
        initial_portfolio (float): Starting portfolio value.
        threshold (float): Minimum acceptable return for Omega Ratio.

    Returns:
        metrics_df (pd.DataFrame): Metrics calculated across simulations.
        portfolio_values (pd.DataFrame): Simulated portfolio values over time with date index.
    """
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    metrics = {
        "Sharpe Ratio": [],
        "VaR (95%)": [],
        "CVaR (95%)": [],
        "Max Drawdown": [],
        "Omega Ratio": []
    }

    # Generate a date range for the x-axis (business days)
    start_date = "2023-01-01"  # Start of the simulation
    date_index = pd.date_range(start=start_date, periods=time_horizon, freq="B")

    # Preallocate a NumPy array for all simulations
    all_portfolio_values = np.zeros((time_horizon, num_simulations))

    for sim in range(num_simulations):
        # Simulate daily returns
        daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, time_horizon)
        portfolio_returns = pd.Series(daily_returns.mean(axis=1))
        cumulative_returns = (1 + portfolio_returns).cumprod()
        all_portfolio_values[:, sim] = initial_portfolio * cumulative_returns

        # Calculate portfolio metrics
        metrics["Sharpe Ratio"].append(calculate_sharpe_ratio(portfolio_returns))
        metrics["VaR (95%)"].append(calculate_var(portfolio_returns, confidence_level=0.95))
        metrics["CVaR (95%)"].append(calculate_cvar(portfolio_returns, confidence_level=0.95))
        metrics["Max Drawdown"].append(calculate_max_drawdown(all_portfolio_values[:, sim]))
        metrics["Omega Ratio"].append(calculate_omega_ratio(portfolio_returns, threshold=threshold))

    # Create a single DataFrame after all simulations are complete
    portfolio_values = pd.DataFrame(all_portfolio_values, index=date_index, columns=[f"Simulation {i+1}" for i in range(num_simulations)])

    metrics_df = pd.DataFrame(metrics)
    return metrics_df, portfolio_values

