import pandas as pd
import numpy as np
from scripts.metrics import (
    calculate_daily_returns,
    calculate_annualized_returns,
    calculate_covariance_matrix,
    calculate_portfolio_return,
    calculate_portfolio_risk,
)
from scripts.optimizer import optimize_portfolio
from scripts.visualizations import plot_efficient_frontier, plot_allocation

#Step 1: Load historical stock data
print("Loading data...")
data = pd.read_csv('data/stock_prices.csv', index_col=0, parse_dates=True)

#Step 2: Calculate portfolio metrics
print("Calculating portfolio metrics...")
daily_returns = calculate_daily_returns(data)
annualized_returns = calculate_annualized_returns(daily_returns)
covariance_matrix = calculate_covariance_matrix(daily_returns)

#Step 3: Optimize portfolio to maximize Sharpe Ratio
print("Optimizing portfolio...")
optimal_weights = optimize_portfolio(annualized_returns, covariance_matrix)

#Step 4: Calculate optimized portfolio metrics
optimized_return = calculate_portfolio_return(optimal_weights, annualized_returns)
optimized_risk = calculate_portfolio_risk(optimal_weights, covariance_matrix)

#Step 5: Print optimization results
print("\nOptimized Portfolio Metrics:")
print(f"  Optimal Weights: {optimal_weights}")
print(f"  Annualized Return: {optimized_return:.2%}")
print(f"  Portfolio Risk (Volatility): {optimized_risk:.2%}")

#Step 6: Visualize results
print("\nGenerating Efficient Frontier...")
plot_efficient_frontier(annualized_returns, covariance_matrix)

print("Generating Portfolio Allocation Pie Chart...")
asset_names = data.columns.tolist()  # Get asset names from the dataset
plot_allocation(optimal_weights, asset_names)
