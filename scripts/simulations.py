"""
simulations.py

Purpose:
- Perform Monte Carlo simulations to validate portfolio performance under random conditions, while also incorporating advanced risk modeling.

Key Steps:
1. Simulate Random Portfolios:
   - Generate random weights for portfolio allocations.
   - Normalize weights so they sum to 1 (fully invested portfolio).

2. Calculate Portfolio Metrics:
   - Portfolio Return: Weighted sum of annualized returns.
   - Portfolio Volatility: Risk derived from the covariance matrix.
   - Sharpe Ratio: (Portfolio Return - Risk-Free Rate) / Portfolio Volatility.

3. Collect Results:
   - Store volatility, return, and Sharpe Ratio for each simulated portfolio.
   - Analyze performance across thousands of portfolios.

4. Output:
   - A dataset of portfolio risk, return, and Sharpe Ratios.
   - Insights into the best and worst portfolio performance metrics.

Enhancements:
- Introduced a probabilistic shock element to simulate stressed market conditions.
- Allow users to specify different risk-free rates to observe performance under varied economic environments.

Usage:
- Use simulation results to compare the optimized portfolio's performance with random portfolios.
"""
import numpy as np

def monte_carlo_simulation(returns, covariance_matrix, num_simulations=10000, risk_free_rate=0.02, shock_factor=None):
    num_assets = len(returns)
    results = np.zeros((3, num_simulations))
    
    for i in range(num_simulations):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)  # Normalize to sum to 1

        # Apply a probabilistic shock if specified
        if shock_factor is not None:
            shocked_returns = returns * (1 + np.random.uniform(-shock_factor, shock_factor, num_assets))
        else:
            shocked_returns = returns

        portfolio_return = np.dot(weights, shocked_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = sharpe_ratio

    return results
