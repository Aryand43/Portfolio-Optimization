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

4. **Monte Carlo Simulations**:
   - Validate portfolio performance with random allocations.
   - Display best and worst Sharpe Ratios.

5. **Visualizations**:
   - Efficient Frontier: Visualize risk-return trade-offs.
   - Portfolio Allocation: Show optimized weights in a pie chart.

How It Works:
- The app dynamically processes user inputs, runs optimization or simulations, and displays results interactively.
- Visualization and results are updated based on user actions.

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
)
from scripts.optimizer import optimize_portfolio
from scripts.visualizations import plot_efficient_frontier, plot_allocation
from scripts.simulations import monte_carlo_simulation

#Streamlit UI
st.title("Portfolio Optimization Tool with Real-Time Data")

#User Inputs
st.sidebar.header("Portfolio Configuration")
tickers = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "AAPL,MSFT,GOOG,AMZN,META")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

#Fetch Live Data
@st.cache_data
def fetch_stock_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        return data['Adj Close']
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

stock_data = fetch_stock_data(tickers, start_date, end_date)
if stock_data is not None:
    st.write("### Stock Data (First Few Rows)")
    st.write(stock_data.head())

    #Calculate Metrics
    daily_returns = calculate_daily_returns(stock_data)
    annualized_returns = calculate_annualized_returns(daily_returns)
    covariance_matrix = calculate_covariance_matrix(daily_returns)

    #Portfolio Optimization
    if st.sidebar.button("Run Optimization"):
        st.write("Optimizing portfolio...")
        optimal_weights = optimize_portfolio(annualized_returns, covariance_matrix)
        optimized_return = calculate_portfolio_return(optimal_weights, annualized_returns)
        optimized_risk = calculate_portfolio_risk(optimal_weights, covariance_matrix)

        #Display Results
        st.write("### Optimized Portfolio Metrics")
        st.write(f"**Annualized Return:** {optimized_return:.2%}")
        st.write(f"**Portfolio Risk (Volatility):** {optimized_risk:.2%}")
        st.write("**Optimal Weights:**")
        st.write({tickers.split(",")[i]: round(optimal_weights[i], 4) for i in range(len(optimal_weights))})

        #Visualizations
        st.write("### Efficient Frontier")
        plot_efficient_frontier(annualized_returns, covariance_matrix)

        st.write("### Portfolio Allocation")
        plot_allocation(optimal_weights, tickers.split(","))

    #Monte Carlo Simulations
    if st.sidebar.button("Run Monte Carlo Simulations"):
        st.write("Running Monte Carlo simulations...")
        simulation_results = monte_carlo_simulation(annualized_returns, covariance_matrix)
        best_sharpe = np.max(simulation_results[2, :])
        worst_sharpe = np.min(simulation_results[2, :])

        st.write("### Monte Carlo Simulation Results")
        st.write(f"**Best Sharpe Ratio:** {best_sharpe:.2f}")
        st.write(f"**Worst Sharpe Ratio:** {worst_sharpe:.2f}")

else:
    st.warning("Failed to fetch stock data. Please check your inputs.")
