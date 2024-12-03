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
import scipy.optimize as optimize 
from scipy.optimize import minimize

def calculate_sharpe_ratio(weights, annualized_returns, covariance_matrix, risk_free_rate=0.02):
    portfolio_return = np.dot(weights, annualized_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio  # Negative for minimization

def calculate_objective_with_costs(weights, annualized_returns, covariance_matrix, initial_weights, transaction_costs, risk_free_rate=0.02):
    # Ensure weights remain within bounds
    weights = np.clip(weights, 0.05, 0.3)

    # Calculate Sharpe Ratio
    sharpe_ratio = calculate_sharpe_ratio(weights, annualized_returns, covariance_matrix, risk_free_rate)

    # Transaction Cost Penalty
    transaction_cost_penalty = np.clip(np.sum(np.abs(weights - initial_weights) * transaction_costs), 0, 1)
    return sharpe_ratio + transaction_cost_penalty

def validate_inputs(annualized_returns, covariance_matrix, initial_weights, transaction_costs):
    if np.any(np.isnan(annualized_returns)):
        raise ValueError("Annualized returns contain NaN values.")
    if np.any(np.isnan(covariance_matrix)):
        raise ValueError("Covariance matrix contains NaN values.")
    if initial_weights is not None and np.any(np.isnan(initial_weights)):
        raise ValueError("Initial weights contain NaN values.")
    if transaction_costs is not None and np.any(np.isnan(transaction_costs)):
        raise ValueError("Transaction costs contain NaN values.")
    if len(annualized_returns) != len(transaction_costs):
        raise ValueError("Mismatch between annualized_returns and transaction_costs dimensions.")

def optimize_portfolio_with_costs(
    annualized_returns,
    covariance_matrix,
    initial_weights=None,
    transaction_costs=None,
    risk_free_rate=0.02,
    bounds=None,
):
    """
    Optimize portfolio weights with transaction cost constraints.

    Args:
        annualized_returns (np.array): Expected annual returns of assets.
        covariance_matrix (np.array): Covariance matrix of asset returns.
        initial_weights (np.array): Initial portfolio weights.
        transaction_costs (np.array): Per-asset transaction cost rates.
        risk_free_rate (float): Risk-free rate.
        bounds (list of tuples): Bounds for each asset's weight.

    Returns:
        dict: Contains optimal weights, Sharpe ratio, transaction cost penalty, and optimization status.
    """
    num_assets = len(annualized_returns)

    # Default to equal initial weights
    if initial_weights is None:
        initial_weights = np.ones(num_assets) / num_assets

    # Default to zero transaction costs
    if transaction_costs is None:
        transaction_costs = np.zeros(num_assets)

    # Validate inputs
    validate_inputs(annualized_returns, covariance_matrix, initial_weights, transaction_costs)

    # Default bounds: 5% to 30% allocation per asset
    if bounds is None:
        bounds = [(0.05, 0.3) for _ in range(num_assets)]

    # Constraint: Weights must sum to 1
    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]

    # Objective function with transaction costs
    def objective(weights):
        portfolio_return = np.dot(weights, annualized_returns)
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(covariance_matrix, weights))
        )
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        cost_penalty = np.sum(np.abs(weights - initial_weights) * transaction_costs)
        return -sharpe_ratio + cost_penalty  # Minimize negative Sharpe Ratio + cost penalty

    # Run the optimization
    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    optimal_weights = np.clip(result.x, [b[0] for b in bounds], [b[1] for b in bounds])
    final_sharpe_ratio = -objective(optimal_weights)  # Remove negative sign
    final_cost_penalty = np.sum(np.abs(optimal_weights - initial_weights) * transaction_costs)

    return {
        "weights": optimal_weights,
        "sharpe_ratio": final_sharpe_ratio,
        "transaction_cost_penalty": final_cost_penalty,
        "success": result.success,
        "message": result.message,
    }
