import os
from src.data.fetch_data import get_cached_historical_data
from src.simulations.monte_carlo import monte_carlo_with_metrics, monte_carlo_simulation
from src.visualizations.plot_monte_carlo import plot_monte_carlo_paths

def run_monte_carlo_full(tickers, start_date, end_date, num_simulations=1000, time_horizon=252):
    """
    Fetch real-world data, calculate metrics, and visualize Monte Carlo portfolio paths.
    """
    print("Fetching historical data...")
    data = get_cached_historical_data("stocks", tickers, start_date, end_date)

    # Extract close prices and calculate returns
    try:
        close_prices = data.xs('Close', axis=1, level=1)
        returns = close_prices.pct_change().dropna()
    except Exception as e:
        print(f"Error processing data: {e}")
        return

    # Run Monte Carlo simulation
    print("Running Monte Carlo simulations...")
    portfolio_values = monte_carlo_simulation(returns, num_simulations, time_horizon)

    # Display Metrics
    try:
        metrics = monte_carlo_with_metrics(returns, num_simulations, time_horizon)
        print("Monte Carlo Metrics:")
        print(metrics.describe())
    except Exception as e:
        print(f"Error during Monte Carlo metrics calculation: {e}")

    # Visualize Portfolio Paths
    print("Visualizing Monte Carlo portfolio paths...")
    plot_monte_carlo_paths(portfolio_values)

def run_monte_carlo_visualization(tickers, start_date, end_date, num_simulations=1000, time_horizon=252):
    """
    Fetch real-world data, run Monte Carlo simulations, and visualize portfolio paths.
    """
    print("Fetching historical data...")
    data = get_cached_historical_data("stocks", tickers, start_date, end_date)

    try:
        close_prices = data.xs('Close', axis=1, level=1)
        returns = close_prices.pct_change().dropna()
    except Exception as e:
        print(f"Error processing data: {e}")
        return

    print("Running Monte Carlo simulations...")
    portfolio_values = monte_carlo_simulation(returns, num_simulations, time_horizon)

    # Define save path
    save_dir = "src/visualizations/outputs"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "monte_carlo_simulation.png")

    print("Visualizing Monte Carlo paths...")
    try:
        plot_monte_carlo_paths(portfolio_values, save_dir=save_dir, file_name="monte_carlo_simulation.png")
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    run_monte_carlo_full(["AAPL", "MSFT"], "2022-01-01", "2022-12-31")
    run_monte_carlo_visualization(["AAPL", "MSFT"], "2022-01-01", "2022-12-31")
