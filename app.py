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
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None


def main():
    st.title("Portfolio Optimization Tool with Real-Time Data")
    st.sidebar.header("Portfolio Configuration")

    tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "AAPL,MSFT,GOOG,AMZN,META")
    tickers = validate_tickers(tickers_input)
    if not tickers:
        return

    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))
    transaction_cost = st.sidebar.slider("Transaction Cost (%)", min_value=0.0, max_value=5.0, step=0.1) / 100

    stock_data = fetch_stock_data(tickers, start_date, end_date)
    if stock_data is None:
        return

    st.write("### Stock Data (Full Range)")
    st.dataframe(stock_data)

    daily_returns = calculate_daily_returns(stock_data)
    annualized_returns = calculate_annualized_returns(daily_returns)
    covariance_matrix = calculate_covariance_matrix(daily_returns)

    if st.sidebar.button("Run Optimization"):
        try:
            st.write("Optimizing portfolio...")
            initial_weights = np.ones(len(annualized_returns)) / len(annualized_returns)
            transaction_costs = [transaction_cost] * len(annualized_returns)
            optimal_weights = optimize_portfolio_with_costs(
                annualized_returns, covariance_matrix, initial_weights, transaction_costs
            )
            optimized_return = calculate_portfolio_return(optimal_weights, annualized_returns)
            optimized_risk = calculate_portfolio_risk(optimal_weights, covariance_matrix)
            portfolio_returns = daily_returns.dot(optimal_weights)
            var_95 = calculate_var(portfolio_returns, confidence_level=0.95)
            cvar_95 = calculate_cvar(portfolio_returns, confidence_level=0.95)
            mdd = calculate_max_drawdown(portfolio_returns)

            st.write("### Optimized Portfolio Metrics")
            st.write(f"**Annualized Return:** {optimized_return:.2%}")
            st.write(f"**Portfolio Risk (Volatility):** {optimized_risk:.2%}")
            st.write(f"**Maximum Drawdown (MDD):** {mdd:.2%}")
            st.write(f"**Value at Risk (VaR) at 95% Confidence Level:** {var_95:.2%}")
            st.write(f"**Conditional Value at Risk (CVaR) at 95% Confidence Level:** {cvar_95:.2%}")
            weights_dict = {tickers[i]: round(optimal_weights[i], 4) for i in range(len(optimal_weights))}
            st.write("**Optimal Weights:**")
            st.write(weights_dict)

            st.write("### Efficient Frontier")
            plot_efficient_frontier(annualized_returns, covariance_matrix)

            st.write("### Portfolio Allocation")
            plot_allocation(optimal_weights, tickers)

        except Exception as e:
            st.error(f"Error during portfolio optimization: {e}")

    st.sidebar.header("Stress Testing")
    enable_stress_test = st.sidebar.checkbox("Enable Stress Testing")
    if enable_stress_test:
        scenario_options = {
            "Market Crash (-30%)": {"AAPL": -0.3, "MSFT": -0.3, "GOOG": -0.3},
            "Inflation Spike": {"AAPL": -0.05, "MSFT": -0.05, "Bonds": -0.2, "Gold": 0.1},
        }
        selected_scenario = st.sidebar.selectbox("Choose Scenario", options=list(scenario_options.keys()))
        custom_shocks = st.sidebar.text_input("Custom Shocks (e.g., AAPL:-0.2,MSFT:-0.1)", value="")
        custom_shock_dict = {}
        if custom_shocks:
            try:
                custom_shock_dict = {
                    asset.strip(): float(shock.strip())
                    for asset, shock in [pair.split(":") for pair in custom_shocks.split(",")]
                }
            except ValueError:
                st.sidebar.error("Invalid custom shock format. Use 'AAPL:-0.2,MSFT:-0.1'.")
                custom_shock_dict = {}

        if st.sidebar.button("Run Stress Test"):
            try:
                if 'optimal_weights' not in locals():
                    st.error("Please run the portfolio optimization before performing stress testing.")
                    return
                stress_scenario = custom_shock_dict if custom_shocks else scenario_options[selected_scenario]
                stress_scenario = {k: v for k, v in stress_scenario.items() if k in daily_returns.columns}
                if not stress_scenario:
                    st.error("No valid assets in the stress scenario match the portfolio. Please adjust your input.")
                    return
                stress_results = stress_test_portfolio(optimal_weights, daily_returns, stress_scenario)
                st.write("### Stress Test Results")
                st.write(f"**Stressed Return:** {stress_results['stressed_return']:.2%}")
                st.write(f"**Stressed Risk (Volatility):** {stress_results['stressed_risk']:.2%}")
                st.write(f"**Maximum Drawdown (MDD):** {stress_results['stressed_mdd']:.2%}")
            except Exception as e:
                st.error(f"Error during stress testing: {e}")

    if st.sidebar.button("Run Monte Carlo Simulations"):
        try:
            st.write("Running Monte Carlo simulations...")
            simulation_results = monte_carlo_simulation(annualized_returns, covariance_matrix)
            best_sharpe = np.max(simulation_results[2, :])
            worst_sharpe = np.min(simulation_results[2, :])
            simulated_var_95 = calculate_var(simulation_results[0, :], confidence_level=0.95)
            simulated_cvar_95 = calculate_cvar(simulation_results[0, :], confidence_level=0.95)
            simulated_max_drawdown = calculate_max_drawdown(simulation_results[0, :])
            st.write("### Monte Carlo Simulation Results")
            st.write(f"**Best Sharpe Ratio:** {best_sharpe:.2f}")
            st.write(f"**Worst Sharpe Ratio:** {worst_sharpe:.2f}")
            st.write(f"**Simulated Maximum Drawdown (MDD):** {simulated_max_drawdown:.2%}")
            st.write(f"**Simulated Value at Risk (VaR) at 95% Confidence Level:** {simulated_var_95:.2%}")
            st.write(f"**Simulated Conditional Value at Risk (CVaR) at 95% Confidence Level:** {simulated_cvar_95:.2%}")
        except Exception as e:
            st.error(f"Error during Monte Carlo simulation: {e}")


if __name__ == "__main__":
    main()
