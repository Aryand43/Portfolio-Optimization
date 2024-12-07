from src.data.fetch_data import get_cached_historical_data
from src.simulations.monte_carlo import monte_carlo_simulation

def run_real_data_monte_carlo(tickers, start_date, end_date, num_simulations=1000, time_horizon=252):
    """
    Fetch real-world data and run Monte Carlo simulations.
    """
    data = get_cached_historical_data("stocks", tickers, start_date, end_date)
    close_prices = data.xs('Close', axis=1, level=1)
    returns = close_prices.pct_change().dropna()
    
    simulations = monte_carlo_simulation(returns, num_simulations, time_horizon)
    return simulations
