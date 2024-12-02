"""
optimizer.py

Purpose:
- Find optimal portfolio weights to maximize Sharpe Ratio (highest risk-adjusted return), incorporating transaction costs.

Key Enhancements:
- Transaction Cost Penalty:
  - Adjusts portfolio weights to account for the cost of rebalancing.
  - Formula: Penalty = Sum(abs(new_weights - old_weights) * transaction_costs).

Core Components:
- Sharpe Ratio function (calculate_sharpe_ratio)
- Portfolio optimization logic (optimize_portfolio_with_costs)
- Advanced constraints (bounds, sum-to-1 requirement)
"""

import numpy as np
from scipy.optimize import minimize

# Calculate Sharpe Ratio
def calculate_sharpe_ratio(weights, annualized_returns, covariance_matrix, risk_free_rate=0.02):
    """
    Calculate the Sharpe Ratio for a given portfolio.

    Args:
        weights (np.array): Portfolio weights.
        annualized_returns (np.array): Expected annual returns of assets.
        covariance_matrix (np.array): Covariance matrix of asset returns.
        risk_free_rate (float): Risk-free rate (default 2%).

    Returns:
        float: Negative Sharpe Ratio (for minimization).
    """
    portfolio_return = np.dot(weights, annualized_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio  # Negative because we minimize in scipy

# Objective function with transaction cost penalty
def calculate_objective_with_costs(weights, annualized_returns, covariance_matrix, initial_weights, transaction_costs, risk_free_rate=0.02):
    """
    Calculate the objective function with transaction cost penalty.

    Args:
        weights (np.array): Portfolio weights.
        annualized_returns (np.array): Expected annual returns of assets.
        covariance_matrix (np.array): Covariance matrix of asset returns.
        initial_weights (np.array): Initial portfolio weights before rebalancing.
        transaction_costs (np.array): Per-asset transaction cost rates.
        risk_free_rate (float): Risk-free rate (default 2%).

    Returns:
        float: Combined Sharpe Ratio and transaction cost penalty.
    """
    # Sharpe Ratio (negative for minimization)
    sharpe_ratio = calculate_sharpe_ratio(weights, annualized_returns, covariance_matrix, risk_free_rate)
    # Transaction Cost Penalty
    transaction_cost_penalty = np.sum(np.abs(weights - initial_weights) * transaction_costs)
    return sharpe_ratio + transaction_cost_penalty  # Combine Sharpe Ratio and transaction costs

# Optimization function with transaction costs
def optimize_portfolio_with_costs(annualized_returns, covariance_matrix, initial_weights=None, transaction_costs=None, risk_free_rate=0.02):
    """
    Optimize portfolio weights with transaction cost constraints.

    Args:
        annualized_returns (np.array): Expected annual returns of assets.
        covariance_matrix (np.array): Covariance matrix of asset returns.
        initial_weights (np.array): Initial portfolio weights (optional, defaults to equal weights).
        transaction_costs (np.array): Per-asset transaction cost rates (optional, defaults to zeros).
        risk_free_rate (float): Risk-free rate (default 2%).

    Returns:
        np.array: Optimal portfolio weights.
    """
    num_assets = len(annualized_returns)
    initial_weights = np.ones(num_assets) / num_assets if initial_weights is None else initial_weights
    transaction_costs = np.zeros(num_assets) if transaction_costs is None else transaction_costs
    bounds = [(0.05, 0.3) for _ in range(num_assets)]  # Min 5%, Max 30% per asset
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]  # Weights sum to 1

    # Optimize portfolio weights
    result = minimize(
        calculate_objective_with_costs,
        initial_weights,
        args=(annualized_returns, covariance_matrix, initial_weights, transaction_costs, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    optimal_weights = result.x

    return optimal_weights  # Return optimal weights only
