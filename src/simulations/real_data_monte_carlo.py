from src.data.fetch_data import (
    get_cached_historical_data,
    get_historical_data_for_asset_class,
    get_realtime_data_for_asset_class,
)
from src.simulations.monte_carlo import monte_carlo_with_metrics
from src.visualizations.plot_monte_carlo_with_metrics import plot_monte_carlo_with_metrics
from src.visualizations.plot_asset_performance import plot_asset_performance
import os
import pandas as pd

def run_monte_carlo_with_visualization(tickers, start_date, end_date, num_simulations=1000, time_horizon=252):
    """
    Fetch real-world data, run Monte Carlo simulations, calculate metrics, and visualize results.
    """
    print("Fetching historical data...")
    data = get_cached_historical_data("stocks", tickers, start_date, end_date)

    try:
        close_prices = data.xs('Close', axis=1, level=1)
        returns = close_prices.pct_change().dropna()
    except Exception as e:
        print(f"Error processing data: {e}")
        return

    print("Running Monte Carlo simulations with metrics...")
    try:
        metrics_df, portfolio_values = monte_carlo_with_metrics(returns, num_simulations, time_horizon)
    except Exception as e:
        print(f"Error during Monte Carlo metrics calculation: {e}")
        return

    print("Visualizing results...")
    save_dir = "src/visualizations/outputs"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "monte_carlo_with_metrics.png")

    try:
        plot_monte_carlo_with_metrics(portfolio_values, metrics_df, save_path=save_path)
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    run_monte_carlo_with_visualization(["AAPL", "MSFT"], "2022-01-01", "2022-12-31")

'''
def run_real_time_monte_carlo(asset_type, symbols, num_simulations=1000, time_horizon=252, initial_portfolio=10000):
    """
    Fetch real-time prices, run Monte Carlo simulations, and visualize results.
    """
    print("Fetching real-time data...")
    real_time_data = get_realtime_data_for_asset_class(asset_type, symbols)

    # Convert real-time prices into a DataFrame
    prices = {symbol: info["price"] for symbol, info in real_time_data.items() if info["price"] is not None}
    prices_df = pd.DataFrame([prices])

    if prices_df.empty:
        print("No valid real-time data to process.")
        return

    # Calculate returns for Monte Carlo simulation
    returns = prices_df.pct_change().dropna()

    print("Running Monte Carlo simulations with metrics...")
    try:
        metrics_df, portfolio_values = monte_carlo_with_metrics(returns, num_simulations, time_horizon, initial_portfolio)
    except Exception as e:
        print(f"Error during Monte Carlo metrics calculation: {e}")
        return

    print("Visualizing results...")
    save_dir = "src/visualizations/outputs"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "real_time_monte_carlo_with_metrics.png")

    try:
        plot_monte_carlo_with_metrics(portfolio_values, metrics_df, save_path=save_path)
    except Exception as e:
        print(f"Error during visualization: {e}")
'''

def compare_asset_classes_performance(start_date, end_date):
    """
    Fetch and compare performance of multiple asset classes.
    """
    asset_classes = {
        "stocks": ["AAPL", "MSFT"],
        "indices": ["^GSPC"],  # S&P 500 Index
        "forex": ["EURUSD=X"],  # Forex pair
        "crypto": ["BTC-USD"],  # Bitcoin
    }

    data = {}
    for asset_class, symbols in asset_classes.items():
        print(f"Fetching data for {asset_class}...")
        data[asset_class] = get_historical_data_for_asset_class(asset_class, symbols, start_date, end_date)

    # Define save path
    save_dir = "src/visualizations/outputs"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "asset_performance_comparison.png")

    print("Visualizing asset-specific performance...")
    try:
        plot_asset_performance(data, list(asset_classes.keys()), save_path=save_path)
    except Exception as e:
        print(f"Error during asset performance visualization: {e}")


if __name__ == "__main__":
    # Run Monte Carlo simulations for stocks
    run_monte_carlo_with_visualization(["AAPL", "MSFT"], "2022-01-01", "2022-12-31")

    # Compare performance across multiple asset classes
    compare_asset_classes_performance("2022-01-01", "2022-12-31")

    # Run real-time Monte Carlo simulations
    run_real_time_monte_carlo("stocks", ["AAPL", "MSFT"])
