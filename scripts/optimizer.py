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

def calculate_sharpe_ratio(weights, annualized_returns, covariance_matrix, risk_free_rate=0.02):
    portfolio_return = np.dot(weights, annualized_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio  # Negative for minimization

def calculate_objective_with_costs(weights, annualized_returns, covariance_matrix, initial_weights, transaction_costs, risk_free_rate=0.02):
    # Debugging: Ensure correct input types
    print(f"Type of weights: {type(weights)}")
    print(f"Type of initial_weights: {type(initial_weights)}")
    print(f"Type of transaction_costs: {type(transaction_costs)}")

    # Calculate Sharpe Ratio
    sharpe_ratio = calculate_sharpe_ratio(weights, annualized_returns, covariance_matrix, risk_free_rate)

    # Transaction Cost Penalty
    transaction_cost_penalty = np.sum(np.abs(weights - initial_weights) * transaction_costs)
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

def optimize_portfolio_with_costs(annualized_returns, covariance_matrix, initial_weights=None, transaction_costs=None, risk_free_rate=0.02, bounds=None):
    # Validate inputs
    validate_inputs(annualized_returns, covariance_matrix, initial_weights, transaction_costs)

    # Convert inputs to np.array
    annualized_returns = np.array(annualized_returns)
    covariance_matrix = np.array(covariance_matrix)
    initial_weights = np.ones(len(annualized_returns)) / len(annualized_returns) if initial_weights is None else np.array(initial_weights)
    transaction_costs = np.zeros(len(annualized_returns)) if transaction_costs is None else np.array(transaction_costs)

    bounds = bounds or [(0.05, 0.3) for _ in range(len(annualized_returns))]
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

    # Optimization
    result = minimize(
        calculate_objective_with_costs,
        initial_weights,
        args=(annualized_returns, covariance_matrix, initial_weights, transaction_costs, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        print(f"Optimization failed: {result.message}")

    optimal_weights = result.x
    final_sharpe_ratio = -calculate_sharpe_ratio(optimal_weights, annualized_returns, covariance_matrix, risk_free_rate)
    final_cost_penalty = np.sum(np.abs(optimal_weights - initial_weights) * transaction_costs)

    return {
        "weights": optimal_weights,
        "sharpe_ratio": final_sharpe_ratio,
        "transaction_cost_penalty": final_cost_penalty,
        "success": result.success,
        "message": result.message,
    }
