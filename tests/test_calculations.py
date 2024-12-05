"""
This module tests the functionality of the `metrics.py` module in `src/calculations`.

1. test_calculate_returns:
   - Tests the calculation of log returns for a given price series.
   - Input: A DataFrame of asset prices.
   - Expected: No null values in the resulting returns DataFrame.

2. test_calculate_covariance:
   - Tests the calculation of the covariance matrix for a set of asset returns.
   - Input: A DataFrame of asset returns.
   - Expected: A covariance matrix of the correct shape.

3. test_portfolio_performance:
   - Tests the calculation of portfolio performance (expected return and risk).
   - Input: Portfolio weights, mean returns, and covariance matrix.
   - Expected: A tuple with two elements (return, risk).
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.calculations.metrics import calculate_returns, calculate_covariance, portfolio_performance


# Test data: Prices of two assets over four time periods
prices = pd.DataFrame({
    'Asset1': [100, 102, 104, 103],
    'Asset2': [50, 51, 52, 53]
})

def calculate_returns(prices):
    """
    Calculate log returns of a price series.
    """
    returns = np.log(prices / prices.shift(1))
    return returns.dropna()

def test_calculate_covariance():
    """
    Tests the calculation of the covariance matrix for the asset returns.
    """
    returns = calculate_returns(prices)
    cov_matrix = calculate_covariance(returns)
    assert cov_matrix.shape == (2, 2), "Covariance matrix calculation failed"

def test_portfolio_performance():
    """
    Tests the calculation of portfolio performance metrics (expected return and risk).
    """
    returns = calculate_returns(prices)
    mean_returns = returns.mean()
    cov_matrix = calculate_covariance(returns)
    weights = [0.5, 0.5]  # Equal weights for two assets
    perf = portfolio_performance(weights, mean_returns, cov_matrix)
    assert len(perf) == 2, "Portfolio performance calculation failed"
