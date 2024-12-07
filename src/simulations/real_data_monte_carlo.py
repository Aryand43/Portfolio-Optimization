from src.data.fetch_data import get_cached_historical_data
from src.simulations.monte_carlo import monte_carlo_with_metrics
from src.visualizations.plot_monte_carlo_with_metrics import plot_monte_carlo_with_metrics
import os

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
