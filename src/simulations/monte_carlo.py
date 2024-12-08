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
    Handles covariance matrix failures with fallback to univariate simulations.
    """
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Handle invalid covariance matrix
    if np.isnan(cov_matrix.values).any() or cov_matrix.shape[0] == 0:
        print("Covariance matrix invalid. Replacing with diagonal matrix.")
        cov_matrix = np.diag(returns.var())

    metrics = {
        "Sharpe Ratio": [],
        "VaR (95%)": [],
        "CVaR (95%)": [],
        "Max Drawdown": [],
        "Omega Ratio": []
    }

    portfolio_values = np.zeros((time_horizon, num_simulations))

    for sim in range(num_simulations):
        try:
            # Attempt multivariate simulation
            daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, time_horizon)
        except np.linalg.LinAlgError:
            # Fallback to univariate simulations if covariance fails
            print(f"Simulation {sim+1} failed: Covariance invalid, using univariate returns.")
            daily_returns = np.random.normal(mean_returns.mean(), np.sqrt(returns.var().mean()), time_horizon)

        # Convert returns to portfolio values
        portfolio_returns = pd.Series(daily_returns.mean(axis=1) if daily_returns.ndim > 1 else daily_returns)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        portfolio_values[:, sim] = initial_portfolio * cumulative_returns

        # Calculate metrics
        try:
            metrics["Sharpe Ratio"].append(calculate_sharpe_ratio(portfolio_returns))
            metrics["VaR (95%)"].append(calculate_var(portfolio_returns, confidence_level=0.95))
            metrics["CVaR (95%)"].append(calculate_cvar(portfolio_returns, confidence_level=0.95))
            metrics["Max Drawdown"].append(calculate_max_drawdown(portfolio_values[:, sim]))
            metrics["Omega Ratio"].append(calculate_omega_ratio(portfolio_returns, threshold=threshold))
        except Exception as e:
            print(f"Metrics calculation failed for Simulation {sim+1}: {e}")
            continue

    metrics_df = pd.DataFrame(metrics)
    return metrics_df, pd.DataFrame(portfolio_values)
