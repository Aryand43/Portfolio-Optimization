"""
metrics.py

Purpose:
- Contains functions to calculate key portfolio metrics for analysis, optimization, and risk management.

Key Functions:
1. calculate_daily_returns(data):
   - Computes daily percentage changes in stock prices.

2. calculate_annualized_returns(daily_returns):
   - Converts daily returns to annualized returns (assumes 252 trading days).

3. calculate_covariance_matrix(daily_returns):
   - Computes the annualized covariance matrix to measure risk correlations.

4. calculate_portfolio_return(weights, annualized_returns):
   - Calculates the overall portfolio return based on weights.

5. calculate_portfolio_risk(weights, covariance_matrix):
   - Calculates portfolio volatility (risk) using the covariance matrix.

6. calculate_var(returns, confidence_level=0.95):
   - Computes the Value at Risk (VaR), which is the threshold loss at a given confidence level.

7. calculate_cvar(returns, confidence_level=0.95):
   - Computes the Conditional Value at Risk (CVaR), which measures the average loss beyond the VaR threshold.

8. calculate_max_drawdown(returns):
   - Computes the Maximum Drawdown (MDD), which measures the largest peak-to-trough drop in cumulative returns.

Usage:
- These functions serve as building blocks for portfolio optimization, risk-return analysis, and advanced risk metrics.
"""

import pandas as pd
import numpy as np

#Function to calculate daily returns
def calculate_daily_returns(data):
    return data.pct_change().dropna()

#Function to calculate expected annualized returns
def calculate_annualized_returns(daily_returns):
    mean_daily_returns = daily_returns.mean()
    return mean_daily_returns * 252  # 252 trading days in a year

#Function to calculate annualized covariance matrix
def calculate_covariance_matrix(daily_returns):
    return daily_returns.cov() * 252  # Annualize covariance

#Function to calculate portfolio return
def calculate_portfolio_return(weights, annualized_returns):
    return np.dot(weights, annualized_returns)

#Function to calculate portfolio risk
def calculate_portfolio_risk(weights, covariance_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

#Function to calculate Value at Risk (VaR)
def calculate_var(returns, confidence_level=0.95):
    """
    Calculate the Value at Risk (VaR) for a given confidence level.

    Args:
        returns (numpy.ndarray): Array of portfolio returns.
        confidence_level (float): Confidence level for VaR (default 95%).

    Returns:
        float: The VaR value.
    """
    sorted_returns = np.sort(-returns)
    var_index = int((1 - confidence_level) * len(sorted_returns))
    var = sorted_returns[var_index]
    return var

#Function to calculate Conditional Value at Risk (CVaR)
def calculate_cvar(returns, confidence_level=0.95):
    """
    Calculate the Conditional Value at Risk (CVaR).

    Args:
        returns (numpy.ndarray): Array of portfolio returns.
        confidence_level (float): Confidence level for CVaR (default 95%).

    Returns:
        float: The CVaR value.
    """
    sorted_returns = np.sort(-returns)
    var_index = int((1 - confidence_level) * len(sorted_returns))
    cvar = np.mean(sorted_returns[:var_index])  # Average of returns beyond VaR
    return cvar

#Function to calculate Maximum Drawdown (MDD)
def calculate_max_drawdown(returns):
    """
    Calculate the Maximum Drawdown (MDD) of a portfolio.

    Args:
        returns (numpy.ndarray): Array of portfolio returns.

    Returns:
        float: The Maximum Drawdown value, or 0 if invalid data is provided.
    """
    # Handle edge cases
    if len(returns) == 0 or np.all(np.isnan(returns)):
        return 0  # Gracefully handle empty or invalid data

    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + returns) - 1

    # Safeguard against invalid cumulative returns
    if np.all(cumulative_returns == 0) or np.all(np.isnan(cumulative_returns)):
        return 0

    # Calculate the running maximum
    running_max = np.maximum.accumulate(cumulative_returns)

    # Safeguard against division by zero
    running_max[running_max == 0] = np.nan

    # Calculate drawdown as a percentage
    drawdown = (cumulative_returns / running_max) - 1

    # Sanity check: Ensure drawdown does not exceed -100%
    drawdown = np.clip(drawdown, -1, 0)

    # Return the worst drawdown (most negative), excluding invalid values
    return np.nanmin(drawdown) if not np.isnan(drawdown).all() else 0
