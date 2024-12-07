import numpy as np
import pandas as pd
from src.calculations.metrics import (
    calculate_sharpe_ratio,
    calculate_var,
    calculate_cvar,
    calculate_max_drawdown,
    calculate_omega_ratio
)

def monte_carlo_simulation(returns, num_simulations=1000, time_horizon=252, initial_portfolio=10000):
    """
    Perform Monte Carlo simulation for portfolio performance.

    Parameters:
        returns (pd.DataFrame): Asset returns.
        num_simulations (int): Number of simulations to run.
        time_horizon (int): Number of days to simulate.
        initial_portfolio (float): Starting portfolio value.

    Returns:
        portfolio_values (pd.DataFrame): Simulated portfolio values over time.
    """
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    portfolio_values = np.zeros((time_horizon, num_simulations))

    for sim in range(num_simulations):
        daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, time_horizon)
        cumulative_returns = np.cumprod(1 + daily_returns, axis=0)
        portfolio_values[:, sim] = initial_portfolio * cumulative_returns[:, -1]

    return pd.DataFrame(portfolio_values)

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

    for sim in range(num_simulations):
        daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, time_horizon)
        portfolio_returns = pd.Series(daily_returns.mean(axis=1))
        portfolio_values = initial_portfolio * (1 + portfolio_returns).cumprod()

        # Calculate metrics for each simulation
        metrics["Sharpe Ratio"].append(calculate_sharpe_ratio(portfolio_returns))
        metrics["VaR (95%)"].append(calculate_var(portfolio_returns, confidence_level=0.95))
        metrics["CVaR (95%)"].append(calculate_cvar(portfolio_returns, confidence_level=0.95))
        metrics["Max Drawdown"].append(calculate_max_drawdown(portfolio_values))
        metrics["Omega Ratio"].append(calculate_omega_ratio(portfolio_returns, threshold=threshold))

    metrics_df = pd.DataFrame(metrics)
    return metrics_df
