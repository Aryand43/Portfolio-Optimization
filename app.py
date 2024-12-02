"""
app.py

Purpose:
- Create an interactive Streamlit app for portfolio optimization and analysis.

Key Features:
1. **User Inputs**:
   - Select stock tickers and date range for historical data.
   - Configure options to run optimization and simulations dynamically.

2. **Data Fetching and Processing**:
   - Fetch live stock prices using `yfinance`.
   - Calculate key portfolio metrics (returns, risk, covariance).

3. **Portfolio Optimization**:
   - Use `scipy.optimize` to maximize Sharpe Ratio.
   - Display optimized portfolio weights, return, and risk.
   - Calculate VaR (Value at Risk), CVaR (Conditional Value at Risk), and Maximum Drawdown (MDD) for the optimized portfolio.

4. **Monte Carlo Simulations**:
   - Validate portfolio performance with random allocations.
   - Display best and worst Sharpe Ratios.
   - Calculate VaR, CVaR, and MDD for simulated portfolios.

5. **Visualizations**:
   - Efficient Frontier: Visualize risk-return trade-offs.
   - Portfolio Allocation: Show optimized weights in a pie chart.

Usage:
- Run the app using `streamlit run app.py` and interact via the web interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scripts.metrics import (
    calculate_daily_returns,
    calculate_annualized_returns,
    calculate_covariance_matrix,
    calculate_portfolio_return,
    calculate_portfolio_risk,
    calculate_var,
    calculate_cvar,
    calculate_max_drawdown,
)
from scripts.optimizer import optimize_portfolio
from scripts.visualizations import plot_efficient_frontier, plot_allocation
from scripts.simulations import monte_carlo_simulation

# Streamlit UI
st.title("Portfolio Optimization Tool with Real-Time Data")

# User Inputs
st.sidebar.header("Portfolio Configuration")
tickers = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "AAPL,MSFT,GOOG,AMZN,META")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

# Fetch Live Data
@st.cache_data
def fetch_stock_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)

        if data.empty:
            st.error("No data available for the given range. Please adjust the date range.")
            return None
        
        # Ensure full date range coverage
        data = data['Adj Close']
        all_dates = pd.date_range(start=start_date, end=end_date)
        data = data.reindex(all_dates, method='pad')  # Fill missing dates with the last available data

        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

stock_data = fetch_stock_data(tickers, start_date, end_date)
if stock_data is not None:
    st.write("### Stock Data (Full Range)")
    st.dataframe(stock_data)

    # Calculate Metrics
    daily_returns = calculate_daily_returns(stock_data)
    annualized_returns = calculate_annualized_returns(daily_returns)
    covariance_matrix = calculate_covariance_matrix(daily_returns)

    # Portfolio Optimization
    if st.sidebar.button("Run Optimization"):
        st.write("Optimizing portfolio...")
        optimal_weights = optimize_portfolio(
            annualized_returns, covariance_matrix
        )
        optimized_return = calculate_portfolio_return(optimal_weights, annualized_returns)
        optimized_risk = calculate_portfolio_risk(optimal_weights, covariance_matrix)

        # Calculate VaR, CVaR, and MDD
        portfolio_returns = daily_returns.dot(optimal_weights)
        var_95 = calculate_var(portfolio_returns, confidence_level=0.95)
        cvar_95 = calculate_cvar(portfolio_returns, confidence_level=0.95)
        mdd = calculate_max_drawdown(portfolio_returns)

        # Display Results
        st.write("### Optimized Portfolio Metrics")
        st.write(f"**Annualized Return:** {optimized_return:.2%}")
        st.write(f"**Portfolio Risk (Volatility):** {optimized_risk:.2%}")
        st.write(f"**Maximum Drawdown (MDD):** {mdd:.2%}")
        st.write(f"**Value at Risk (VaR) at 95% Confidence Level:** {var_95:.2%}")
        st.write(f"**Conditional Value at Risk (CVaR) at 95% Confidence Level:** {cvar_95:.2%}")
        st.write("**Optimal Weights:**")
        st.write({tickers.split(",")[i]: round(optimal_weights[i], 4) for i in range(len(optimal_weights))})

        # Visualizations
        st.write("### Efficient Frontier")
        plot_efficient_frontier(annualized_returns, covariance_matrix)

        st.write("### Portfolio Allocation")
        plot_allocation(optimal_weights, tickers.split(","))

    # Monte Carlo Simulations
    if st.sidebar.button("Run Monte Carlo Simulations"):
        st.write("Running Monte Carlo simulations...")
        simulation_results = monte_carlo_simulation(annualized_returns, covariance_matrix)
        best_sharpe = np.max(simulation_results[2, :])
        worst_sharpe = np.min(simulation_results[2, :])

        # Calculate VaR, CVaR, and MDD for Simulated Portfolios
        simulated_var_95 = calculate_var(simulation_results[0, :], confidence_level=0.95)
        simulated_cvar_95 = calculate_cvar(simulation_results[0, :], confidence_level=0.95)
        simulated_max_drawdown = calculate_max_drawdown(simulation_results[0, :])

        # Display Monte Carlo Results
        st.write("### Monte Carlo Simulation Results")
        st.write(f"**Best Sharpe Ratio:** {best_sharpe:.2f}")
        st.write(f"**Worst Sharpe Ratio:** {worst_sharpe:.2f}")
        st.write(f"**Simulated Maximum Drawdown (MDD):** {simulated_max_drawdown:.2%}")
        st.write(f"**Simulated Value at Risk (VaR) at 95% Confidence Level:** {simulated_var_95:.2%}")
        st.write(f"**Simulated Conditional Value at Risk (CVaR) at 95% Confidence Level:** {simulated_cvar_95:.2%}")

else:
    st.warning("Failed to fetch stock data. Please check your inputs.")
