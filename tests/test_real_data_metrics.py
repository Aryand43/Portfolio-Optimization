import pandas as pd
from src.simulations.real_data_metrics import analyze_real_data_metrics
from src.calculations.metrics import calculate_omega_ratio


def test_analyze_real_data_metrics():
    """
    Test fetching real-world data and calculating metrics.
    """
    tickers = ["AAPL", "MSFT"]
    metrics = analyze_real_data_metrics(tickers, "2022-01-01", "2022-12-31")
    assert not metrics.empty, "Failed to calculate metrics for real-world data"
    assert "Sharpe Ratio" in metrics.columns, "Sharpe Ratio missing from output"
    assert "VaR (95%)" in metrics.columns, "VaR missing from output"

def test_calculate_omega_ratio():
    """
    Test the calculation of the Omega Ratio with real-world data.
    """
    returns = pd.Series([0.05, 0.07, 0.03, -0.02, 0.04])
    omega_ratio = calculate_omega_ratio(returns, threshold=0.02)
    assert round(omega_ratio, 2) == 2.75, "Omega Ratio calculation failed"


