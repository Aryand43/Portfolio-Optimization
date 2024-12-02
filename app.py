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
   - Use `scipy.optimize` to maximize Sharpe Ratio, incorporating transaction costs.
   - Display optimized portfolio weights, return, and risk.
   - Calculate VaR (Value at Risk), CVaR (Conditional Value at Risk), and Maximum Drawdown (MDD) for the optimized portfolio.

4. **Monte Carlo Simulations**:
   - Validate portfolio performance with random allocations.
   - Display best and worst Sharpe Ratios.
   - Calculate VaR, CVaR, and MDD for simulated portfolios.

5. **Stress Testing**:
   - Evaluate portfolio performance under predefined or custom stress scenarios.
   - Display metrics such as stressed return, risk, and Maximum Drawdown (MDD).

6. **Visualizations**:
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
    stress_test_portfolio,
)
from scripts.optimizer import optimize_portfolio_with_costs
from scripts.visualizations import plot_efficient_frontier, plot_allocation
from scripts.simulations import monte_carlo_simulation


def validate_tickers(tickers_input):
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
    tickers = [ticker for ticker in tickers if ticker]
    if len(tickers) < 2:
        st.error("Please enter at least two stock tickers.")
        return None
    return tickers


@st.cache_data(ttl=3600)
def fetch_stock_data(tickers, start_date, end_date):
    try:
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            return None
        data = yf.download(tickers, start=start_date, end=end_date)
        if data.empty:
            st.error("No data available for the given tickers and date range.")
            return None
        data = data["Adj Close"]
        if data.isnull().all().any():
            st.error("Insufficient data for one or more selected tickers.")
            return None
        all_dates = pd.date_range(start=start_date, end=end_date)
        data = data.reindex(all_dates, method="pad")

        # Drop rows/columns with too many NaNs
        data = data.dropna(how="all")  # Drop rows where all values are NaN
        data = data.dropna(axis=1, how="any")  # Drop columns (tickers) with any NaN
        if data.empty:
            st.error("Filtered data contains no valid entries. Check your inputs.")
            return None

        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

def main():
    st.title("Portfolio Optimization Tool with Real-Time Data")
    st.sidebar.header("Portfolio Configuration")

    # Sidebar Inputs
    tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "AAPL,MSFT,GOOG,AMZN,META")
    tickers = validate_tickers(tickers_input)
    if not tickers:
        return

    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))
    transaction_cost = st.sidebar.slider("Transaction Cost (%)", min_value=0.0, max_value=5.0, step=0.1) / 100

    # Fetch Data
    stock_data = fetch_stock_data(tickers, start_date, end_date)
    if stock_data is None:
        return

    st.write("### Stock Data")
    st.dataframe(stock_data)

    # Calculate Metrics
    daily_returns = calculate_daily_returns(stock_data)
    annualized_returns = calculate_annualized_returns(daily_returns)
    covariance_matrix = calculate_covariance_matrix(daily_returns)

    # Tabs for Organized Sections
    tab1, tab2, tab3 = st.tabs(["Portfolio Optimization", "Stress Testing", "Simulations"])

    # Tab 1: Portfolio Optimization
    with tab1:
        st.header("Portfolio Optimization")
        st.write("Optimize your portfolio by balancing risk and return.")

        if st.button("Run Optimization"):
            with st.spinner("Optimizing portfolio..."):
                try:
                    initial_weights = np.ones(len(annualized_returns)) / len(annualized_returns)
                    transaction_costs = [transaction_cost] * len(annualized_returns)
                    optimal_weights = optimize_portfolio_with_costs(
                        annualized_returns, covariance_matrix, initial_weights, transaction_costs
                    )
                    # Display Results
                    optimized_return = calculate_portfolio_return(optimal_weights, annualized_returns)
                    optimized_risk = calculate_portfolio_risk(optimal_weights, covariance_matrix)
                    portfolio_returns = daily_returns.dot(optimal_weights)
                    var_95 = calculate_var(portfolio_returns, confidence_level=0.95)
                    cvar_95 = calculate_cvar(portfolio_returns, confidence_level=0.95)
                    mdd = calculate_max_drawdown(portfolio_returns)

                    st.write(f"**Annualized Return:** {optimized_return:.2%}")
                    st.write(f"**Portfolio Risk:** {optimized_risk:.2%}")
                    st.write(f"**MDD:** {mdd:.2%}")
                    st.write(f"**VaR (95%):** {var_95:.2%}")
                    st.write(f"**CVaR (95%):** {cvar_95:.2%}")
                    plot_allocation(optimal_weights, tickers)
                except Exception as e:
                    st.error(f"Error during optimization: {e}")

    # Tab 2: Stress Testing
    with tab2:
        st.header("Stress Testing")
        st.write("Evaluate portfolio performance under market stress scenarios.")
        enable_stress_test = st.checkbox("Enable Stress Testing")
        if enable_stress_test:
            scenario_options = {"Market Crash (-30%)": {"AAPL": -0.3}, "Inflation Spike": {"Bonds": -0.2}}
            selected_scenario = st.selectbox("Choose Scenario", options=scenario_options.keys())
            custom_shocks = st.text_input("Custom Scenarios (e.g., AAPL:-0.2)")
            # Add stress testing logic here

    # Tab 3: Simulations
    with tab3:
        st.header("Monte Carlo Simulations")
        if st.button("Run Simulations"):
            with st.spinner("Running Monte Carlo simulations..."):
                # Add simulation logic here
                pass


if __name__ == "__main__":
    main()
