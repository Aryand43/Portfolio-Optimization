"""
This module contains core financial calculations for portfolio optimization.

1. calculate_returns:
   - Computes log returns of a given price series.
   - Input: A DataFrame of asset prices.
   - Output: A DataFrame of log returns.

2. calculate_covariance:
   - Calculates the covariance matrix of asset returns.
   - Input: A DataFrame of asset returns.
   - Output: Covariance matrix as a DataFrame.

3. portfolio_performance:
   - Evaluates the expected return and risk (volatility) of a portfolio.
   - Inputs:
       - weights: Array of portfolio weights.
       - mean_returns: Series of average asset returns.
       - cov_matrix: Covariance matrix of returns.
   - Output: A tuple (expected return, risk).

4. calculate_sharpe_ratio:
   - Computes the Sharpe Ratio for a portfolio.
   - Inputs:
       - portfolio_return: Expected portfolio return.
       - portfolio_volatility: Portfolio risk (volatility).
       - risk_free_rate: Default 0.02 (2%).
   - Output: Sharpe Ratio.

5. calculate_omega_ratio:
   - Computes the Omega Ratio for a portfolio.
   - Inputs:
       - returns: DataFrame of portfolio returns.
       - threshold: Threshold return (e.g., risk-free rate).
   - Output: Omega Ratio.

6. calculate_var:
   - Computes Value at Risk (VaR) at a specified confidence level.
   - Inputs:
       - returns: Series of portfolio returns.
       - confidence_level: The confidence level for VaR calculation (e.g., 0.95).
   - Output: Value at Risk.

7. calculate_cvar:
   - Computes Conditional Value at Risk (CVaR) for the portfolio.
   - Inputs:
       - returns: Series of portfolio returns.
       - confidence_level: The confidence level for CVaR calculation.
   - Output: Conditional Value at Risk.

8. calculate_max_drawdown:
   - Computes the Maximum Drawdown of the portfolio.
   - Inputs:
       - portfolio_values: Series of portfolio values over time.
   - Output: Maximum Drawdown.
"""

import numpy as np
import pandas as pd

# Existing functions (calculate_returns, calculate_covariance, portfolio_performance)

def calculate_returns(prices):
    returns = np.log(prices / prices.shift(1))
    return returns.dropna()

def calculate_covariance(returns):
    return returns.cov()

def portfolio_performance(weights, mean_returns, cov_matrix):
    weights = np.array(weights)
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    """
    Calculate the Sharpe Ratio for a portfolio.
    """
    excess_returns = returns.mean() - risk_free_rate
    volatility = returns.std()
    return excess_returns / volatility if volatility != 0 else 0

def calculate_omega_ratio(returns, threshold=0.0):
    """
    Calculate the Omega Ratio of the portfolio.

    Parameters:
        returns: pd.Series - Asset returns.
        threshold: float - Minimum acceptable return (default is 0%).

    Returns:
        Omega Ratio as a float.
    """
    excess_gains = returns[returns > threshold] - threshold
    excess_losses = threshold - returns[returns <= threshold]

    if excess_losses.sum() == 0:
        return float('inf')  # If no losses, Omega Ratio is infinite.

    return excess_gains.sum() / excess_losses.sum()


def calculate_var(returns, confidence_level=0.95):
    """
    Calculate the Value at Risk (VaR) for the portfolio.
    """
    return np.percentile(returns, (1 - confidence_level) * 100)

def calculate_cvar(returns, confidence_level=0.95):
    """
    Calculate the Conditional Value at Risk (CVaR) for the portfolio.
    """
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def calculate_max_drawdown(portfolio_values):
    """
    Calculate the Maximum Drawdown (MDD) for a portfolio.

    Parameters:
        portfolio_values (array-like): Portfolio values over time.

    Returns:
        float: Maximum Drawdown value.
    """
    if isinstance(portfolio_values, np.ndarray):  # Handle NumPy arrays
        portfolio_values = pd.Series(portfolio_values)

    cumulative_max = portfolio_values.cummax()
    drawdown = (portfolio_values - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    return max_drawdown