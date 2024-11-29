"""
optimizer.py

Purpose:
- Find optimal portfolio weights to maximize Sharpe Ratio (highest risk-adjusted return).

Key Steps:
1. Calculate Sharpe Ratio:
   - Formula: (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
   - Portfolio Return: Weighted sum of returns.
   - Portfolio Volatility: Risk derived from covariance matrix.
2. Optimize Weights:
   - Use scipy.optimize.minimize to maximize Sharpe Ratio.
   - Constraints: Weights sum to 1, no short-selling (0 ≤ weight ≤ 1).
3. Output:
   - Optimal asset allocation (weights).

Core Components:
- Sharpe Ratio function (calculate_sharpe_ratio)
- Optimization logic (optimize_portfolio)

"""

import numpy as np
from scipy.optimize import minimize

#Calculate Sharpe Ratio
def calculate_sharpe_ratio(weights, annualized_returns, covariance_matrix, risk_free_rate=0.02):
    portfolio_return = np.dot(weights, annualized_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio  #Negative because we minimize in scipy

#Optimization function
def optimize_portfolio(annualized_returns, covariance_matrix, risk_free_rate=0.02):
    num_assets = len(annualized_returns)
    initial_weights = np.ones(num_assets) / num_assets  #Start with equal weights
    bounds = [(0, 1) for _ in range(num_assets)]  #No short-selling
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]  #Weights must sum to 1

    result = minimize(
        calculate_sharpe_ratio,
        initial_weights,
        args=(annualized_returns, covariance_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x  #Optimal weights
