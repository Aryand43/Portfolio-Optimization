"""
This module tests the data fetching functionality from fetch_data.py.
"""

from src.data.fetch_data import (
    get_historical_data,
    get_realtime_data,
    get_batch_historical_data,
    get_batch_realtime_data,
    get_historical_data_for_asset_class,  
    get_realtime_data_for_asset_class,   
    get_cached_historical_data,
)

def test_get_historical_data():
    """
    Test fetching historical data for a single stock.
    """
    data = get_historical_data("AAPL", "2022-01-01", "2022-12-31")
    assert not data.empty, "Failed to fetch historical data"
    assert "Close" in data.columns, "Data missing expected 'Close' column"

def test_get_realtime_data():
    """
    Test fetching real-time data for a single stock.
    """
    data = get_realtime_data("AAPL")
    assert "price" in data, "Failed to fetch real-time price"
    assert "timestamp" in data, "Failed to fetch timestamp"

def test_get_batch_historical_data():
    """
    Test fetching historical data for multiple stocks.
    """
    tickers = ["AAPL", "MSFT"]
    data = get_batch_historical_data(tickers, "2022-01-01", "2022-12-31")
    assert not data.empty, "Failed to fetch batch historical data"
    assert "AAPL" in data.columns.levels[0], "Batch data missing expected ticker"

def test_get_batch_realtime_data():
    """
    Test fetching real-time data for multiple stocks.
    """
    tickers = ["AAPL", "MSFT"]
    data = get_batch_realtime_data(tickers)
    assert all(ticker in data for ticker in tickers), "Failed to fetch batch real-time data"
    assert "price" in data["AAPL"], "Real-time data missing 'price' key"

def test_historical_data_for_crypto():
    """
    Test fetching historical data for crypto assets.
    """
    data = get_historical_data_for_asset_class("crypto", ["BTC-USD"], "2022-01-01", "2022-12-31")
    assert not data.empty, "Failed to fetch historical data for crypto"
    assert ("BTC-USD", "Close") in data.columns, "Crypto data missing expected 'Close' column for BTC-USD"

def test_realtime_data_for_forex():
    """
    Test fetching real-time data for forex pairs.
    """
    data = get_realtime_data_for_asset_class("forex", ["EURUSD=X"])
    assert "price" in data["EURUSD=X"], "Failed to fetch real-time data for forex"

def test_historical_data_for_indices():
    """
    Test fetching historical data for indices.
    """
    data = get_historical_data_for_asset_class("indices", ["^GSPC"], "2022-01-01", "2022-12-31")
    assert not data.empty, "Failed to fetch historical data for indices"
    assert ("^GSPC", "Close") in data.columns, "Indices data missing expected 'Close' column for ^GSPC"


def test_realtime_data_for_indices():
    """
    Test fetching real-time data for indices.
    """
    data = get_realtime_data_for_asset_class("indices", ["^GSPC"])
    assert "price" in data["^GSPC"], "Failed to fetch real-time data for indices"

def test_cached_historical_data():
    """
    Test fetching cached historical data for efficiency.
    """
    data = get_cached_historical_data("stocks", ["AAPL", "MSFT"], "2022-01-01", "2022-12-31")
    assert not data.empty, "Failed to fetch cached historical data"
    assert ("AAPL", "Close") in data.columns, "Cached data missing 'Close' column for AAPL"
    assert ("MSFT", "Close") in data.columns, "Cached data missing 'Close' column for MSFT"