"""
This module tests the data fetching functionality from fetch_data.py.
"""

from src.data.fetch_data import get_historical_data, get_realtime_data

def test_get_historical_data():
    """
    Test fetching historical data.
    """
    data = get_historical_data("AAPL", "2022-01-01", "2022-12-31")
    assert not data.empty, "Failed to fetch historical data"
    assert "Close" in data.columns, "Data missing expected columns"

def test_get_realtime_data():
    """
    Test fetching real-time data.
    """
    data = get_realtime_data("AAPL")
    assert "price" in data, "Failed to fetch real-time price"
    assert "timestamp" in data, "Failed to fetch timestamp"
