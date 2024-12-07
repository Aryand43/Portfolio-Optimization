from src.simulations.monte_carlo import monte_carlo_with_metrics
import pandas as pd

def test_monte_carlo_with_metrics():
    """
    Test Monte Carlo simulation with integrated metrics.
    """
    data = pd.DataFrame({
        "Asset1": [0.01, 0.02, -0.01, 0.03],
        "Asset2": [0.02, 0.01, 0.0, -0.02]
    })
    metrics = monte_carlo_with_metrics(data, num_simulations=10, time_horizon=10)

    assert not metrics.empty, "Metrics DataFrame is empty"
    assert "Sharpe Ratio" in metrics.columns, "Sharpe Ratio missing from results"
    assert "VaR (95%)" in metrics.columns, "VaR missing from results"
    assert "CVaR (95%)" in metrics.columns, "CVaR missing from results"
    assert "Max Drawdown" in metrics.columns, "Max Drawdown missing from results"
    assert "Omega Ratio" in metrics.columns, "Omega Ratio missing from results"
