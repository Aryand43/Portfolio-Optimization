import numpy as np
import pandas as pd

def monte_carlo_simulation(returns, num_simulations=1000, time_horizon=252, initial_portfolio=10000):
    """
    Perform Monte Carlo simulation for portfolio performance.

    Parameters:
        returns (pd.DataFrame): Asset returns.
        num_simulations (int): Number of simulations to run.
        time_horizon (int): Number of days to simulate.
        initial_portfolio (float): Starting portfolio value.

    Returns:
        simulations (pd.DataFrame): Simulated portfolio values over time.
    """
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    portfolio_values = np.zeros((time_horizon, num_simulations))

    for sim in range(num_simulations):
        # Generate daily returns for time_horizon days
        daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, time_horizon)
        # Combine returns across assets into a single portfolio return
        portfolio_returns = daily_returns.mean(axis=1)
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        portfolio_values[:, sim] = initial_portfolio * cumulative_returns

    return pd.DataFrame(portfolio_values)

