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
from scripts.optimizer import optimize_portfolio_with_costs
from scripts.visualizations import plot_efficient_frontier, plot_allocation, plot_correlation_heatmap
from scripts.simulations import monte_carlo_simulation
from scripts.ml_pipeline import prepare_features, train_ml_model, predict_future
import matplotlib.pyplot as plt

# Streamlit UI
st.title("Portfolio Optimization Tool with ML Integration")

# User Inputs
tickers_input = st.text_input("Enter Stock Tickers (comma-separated)", "AAPL,MSFT,GOOG,AMZN,META")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))
transaction_cost = st.slider("Transaction Cost (%)", min_value=0.0, max_value=5.0, step=0.1) / 100

if "stock_data" not in st.session_state:
    st.session_state.stock_data = None

# Fetch and Process Data
if st.button("Fetch Data"):
    stock_data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"].dropna()
    if stock_data.empty:
        st.error("No data available for the given tickers and date range.")
    elif len(stock_data) < 10:  # Ensure enough data points for ML
        st.error("Not enough data for analysis. Please select a larger date range.")
    else:
        st.session_state.stock_data = stock_data

if st.session_state.stock_data is not None:
    stock_data = st.session_state.stock_data
    st.write("### Stock Data")
    st.dataframe(stock_data)

    # Calculate Metrics
    daily_returns = calculate_daily_returns(stock_data)
    annualized_returns = calculate_annualized_returns(daily_returns)
    covariance_matrix = calculate_covariance_matrix(daily_returns)

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    plot_correlation_heatmap(stock_data, ax)
    st.pyplot(fig)

    # Portfolio Optimization
    if st.button("Run Optimization"):
        initial_weights = np.ones(len(annualized_returns)) / len(annualized_returns)
        transaction_costs = [transaction_cost] * len(annualized_returns)
        optimization_result = optimize_portfolio_with_costs(annualized_returns, covariance_matrix, initial_weights, transaction_costs)
        optimal_weights = optimization_result["weights"]
        optimized_return = calculate_portfolio_return(optimal_weights, annualized_returns)
        optimized_risk = calculate_portfolio_risk(optimal_weights, covariance_matrix)
        portfolio_returns = daily_returns.dot(optimal_weights)
        var_95 = calculate_var(portfolio_returns, confidence_level=0.95)
        cvar_95 = calculate_cvar(portfolio_returns, confidence_level=0.95)
        mdd = calculate_max_drawdown(portfolio_returns)

        # Display Optimization Results
        st.write(f"**Annualized Return:** {optimized_return:.2%}")
        st.write(f"**Portfolio Risk:** {optimized_risk:.2%}")
        st.write(f"**Maximum Drawdown (MDD):** {mdd:.2%}")
        st.write(f"**VaR (95%):** {var_95:.2%}")
        st.write(f"**CVaR (95%):** {cvar_95:.2%}")

        # Plot Allocation
        fig, ax = plt.subplots()
        plot_allocation(optimal_weights, tickers, ax=ax)
        st.pyplot(fig)

        # Plot Efficient Frontier
        st.write("### Efficient Frontier")
        fig, ax = plt.subplots()
        plot_efficient_frontier(annualized_returns, covariance_matrix, ax=ax)
        st.pyplot(fig)

    # Monte Carlo Simulations
    if st.button("Run Monte Carlo Simulation"):
        st.write("### Monte Carlo Simulation Results")
        simulation_results = monte_carlo_simulation(annualized_returns, covariance_matrix)
        best_sharpe = np.max(simulation_results[2, :])
        worst_sharpe = np.min(simulation_results[2, :])
        st.write(f"**Best Sharpe Ratio:** {best_sharpe:.2f}")
        st.write(f"**Worst Sharpe Ratio:** {worst_sharpe:.2f}")

# Predictive Analytics
if st.button("Run ML Predictions"):
    if st.session_state.stock_data is not None:
        daily_returns = calculate_daily_returns(st.session_state.stock_data)
        try:
            features, target = prepare_features(daily_returns)

            if len(features) == 0 or len(target) == 0:
                st.error("Not enough data to run ML predictions. Please select a larger date range.")
            else:
                model, predictions = train_ml_model(features, target)
                latest_data = features.iloc[[-1]]  # Use the latest available data for future prediction
                future_predictions = predict_future(model, latest_data, days=5)

                # Display Predictions
                st.write("### Predicted Future Returns")
                st.write([f"Day {i + 1}: {pred:.2%}" for i, pred in enumerate(future_predictions)])
        except ValueError as e:
            st.error(str(e))
    else:
        st.error("Please fetch data first.")
