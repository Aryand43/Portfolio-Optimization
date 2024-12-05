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
"""

import numpy as np
import pandas as pd

def calculate_returns(prices):
    """
    Calculate log returns of a price series.
    """
    returns = np.log(prices / prices.shift(1))
    return returns.dropna()

def calculate_covariance(returns):
    """
    Calculate the covariance matrix of returns.
    """
    return returns.cov()

def portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Calculate the expected portfolio return and risk.
    """
    weights = np.array(weights)  # Ensure weights are a NumPy array
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility
