"""
metrics.py

Purpose:
- Contains functions to calculate key portfolio metrics for analysis and optimization.

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

Usage:
- These functions are building blocks for portfolio optimization and risk-return analysis.
"""

import pandas as pd
import numpy as np

#Function to calculate daily returns
def calculate_daily_returns(data):
    return data.pct_change().dropna()

#Function to calculate expected annualized returns
def calculate_annualized_returns(daily_returns):
    mean_daily_returns = daily_returns.mean()
    return mean_daily_returns * 252  #252 trading days in a year

#Function to calculate annualized covariance matrix
def calculate_covariance_matrix(daily_returns):
    return daily_returns.cov() * 252  # Annualize covariance

#Function to calculate portfolio return
def calculate_portfolio_return(weights, annualized_returns):
    return np.dot(weights, annualized_returns)

#Function to calculate portfolio risk
def calculate_portfolio_risk(weights, covariance_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
