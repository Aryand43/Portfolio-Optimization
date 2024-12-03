"""
visualizations.py

Purpose:
- Generate key visualizations for portfolio analysis:
  1. Efficient Frontier: Shows the trade-off between risk (volatility) and return.
  2. Portfolio Allocation: Displays the optimal asset weights as a pie chart.
  3. Correlation Heatmap: Visualizes the correlation between different assets.

Key Functions:
1. plot_efficient_frontier(returns, covariance_matrix, risk_free_rate=0.02):
   - Simulates random portfolios.
   - Calculates their risk, return, and Sharpe Ratio.
   - Plots risk vs. return, with Sharpe Ratio as color.

2. plot_allocation(weights, asset_names):
   - Creates a pie chart of the optimal portfolio weights.

3. plot_correlation_heatmap(data, ax):
   - Creates a heatmap to visualize correlations between assets.

Usage:
- Call these functions after optimizing the portfolio to interpret results visually.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to plot the Efficient Frontier
def plot_efficient_frontier(returns, covariance_matrix, risk_free_rate=0.02):
    num_portfolios = 10000
    num_assets = len(returns)

    # Generate random portfolio weights
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)  # Normalize weights to sum to 1
        portfolio_return = np.dot(weights, returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = sharpe_ratio
        weights_record.append(weights)

    # Plot
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis', marker='o')
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

# Function to plot Portfolio Allocation
def plot_allocation(weights, asset_names, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(weights, labels=asset_names, autopct='%1.1f%%', startangle=90)
    ax.set_title('Optimal Portfolio Allocation')
    plt.show()

# Function to plot Correlation Heatmap
def plot_correlation_heatmap(data, ax):
    """
    Plot correlation heatmap for a given dataset.

    Args:
        data (pd.DataFrame): Historical price data of assets.
        ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib axis to plot on.
    """
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title('Asset Correlation Heatmap')
