from src.data.fetch_data import get_cached_historical_data
from src.simulations.monte_carlo import monte_carlo_with_metrics

def run_monte_carlo_with_metrics(tickers, start_date, end_date, num_simulations=1000, time_horizon=252):
    """
    Fetch real-world data and calculate metrics using Monte Carlo simulations.
    """
    print("Fetching historical data...")
    data = get_cached_historical_data("stocks", tickers, start_date, end_date)
    print("Fetched data:")
    print(data.head())

    # Extract close prices and calculate returns
    try:
        close_prices = data.xs('Close', axis=1, level=1)
        print("Close Prices:")
        print(close_prices.head())
        
        returns = close_prices.pct_change().dropna()
        print("Calculated Returns:")
        print(returns.head())
    except Exception as e:
        print(f"Error processing data: {e}")
        return

    # Run Monte Carlo simulation with metrics
    try:
        metrics = monte_carlo_with_metrics(returns, num_simulations, time_horizon)
        print("Monte Carlo Metrics:")
        print(metrics.describe())
    except Exception as e:
        print(f"Error during Monte Carlo simulation: {e}")

if __name__ == "__main__":
    run_monte_carlo_with_metrics(["AAPL", "MSFT"], "2022-01-01", "2022-12-31")
