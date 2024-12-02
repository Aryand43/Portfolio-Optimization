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
    if data.isnull().any().any():
        st.warning("NaN values detected in the input data. Cleaning the data by forward-filling.")
        data = data.fillna(method="ffill")  # Forward-fill remaining NaN values
    return data.pct_change().dropna()

#Function to calculate expected annualized returns
def calculate_annualized_returns(daily_returns):
    mean_daily_returns = daily_returns.mean()
    return mean_daily_returns * 252  # 252 trading days in a year

#Function to calculate annualized covariance matrix
def calculate_covariance_matrix(daily_returns, trading_days=252):
    return daily_returns.cov() * trading_days


#Function to calculate portfolio return
def calculate_portfolio_return(weights, annualized_returns):
    return np.dot(weights, annualized_returns)

#Function to calculate portfolio risk
def calculate_portfolio_risk(weights, covariance_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

#Function to calculate Value at Risk (VaR)
def calculate_var(returns, confidence_level=0.95):
    sorted_returns = np.sort(returns)  # No need for negative sorting
    var_index = int((1 - confidence_level) * len(sorted_returns))
    return abs(sorted_returns[var_index])  # Return positive value

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
    if len(returns) == 0 or np.all(np.isnan(returns)):
        return 0  # Gracefully handle invalid data

    cumulative_returns = np.cumprod(1 + returns) - 1
    if np.all(cumulative_returns == 0):
        return 0  # Flat returns, no drawdown
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns / running_max) - 1
    return np.nanmin(drawdown) if not np.isnan(drawdown).all() else 0

def stress_test_portfolio(portfolio_weights, asset_returns, stress_scenario, mode="multiplicative"):
    """
    Simulate portfolio performance under stress scenarios.

    Args:
        portfolio_weights (np.array): Portfolio weights.
        asset_returns (pd.DataFrame): Historical asset returns.
        stress_scenario (dict): Stress scenario with asset-specific shocks.
        mode (str): "multiplicative" (default) or "additive" for shock application.

    Returns:
        dict: Portfolio metrics under stress.
    """
    stressed_returns = asset_returns.copy()
    for asset, shock in stress_scenario.items():
        if asset in stressed_returns.columns:
            if mode == "multiplicative":
                stressed_returns[asset] *= (1 + shock)
            elif mode == "additive":
                stressed_returns[asset] += shock

    portfolio_stressed_returns = stressed_returns.dot(portfolio_weights)
    return {
        "stressed_return": portfolio_stressed_returns.mean(),
        "stressed_risk": portfolio_stressed_returns.std(),
        "stressed_mdd": calculate_max_drawdown(portfolio_stressed_returns),
    }

