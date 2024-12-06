"""
This module fetches real-world financial data using the yfinance library.

Functions:
1. get_historical_data:
   - Fetch historical price data for a given ticker.
   - Input: Ticker symbol, start date, and end date.
   - Output: DataFrame of historical OHLCV data.

2. get_realtime_data:
   - Fetch the latest price data for a given ticker.
   - Input: Ticker symbol.
   - Output: Latest price and timestamp.
"""
from joblib import Memory
import yfinance as yf

def get_historical_data(ticker, start_date, end_date):
    """
    Fetch historical data for a given ticker.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def get_realtime_data(ticker):
    """
    Fetch the latest price for a given ticker.
    """
    stock = yf.Ticker(ticker)
    latest_data = stock.history(period="1d")
    return {
        "price": latest_data["Close"].iloc[-1],
        "timestamp": latest_data.index[-1]
    }

def get_batch_historical_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    return data

def get_batch_realtime_data(tickers):
    results = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        latest_data = stock.history(period="1d")
        results[ticker] = {
            "price": latest_data["Close"].iloc[-1],
            "timestamp": latest_data.index[-1]
        }
    return results

def get_historical_data_for_asset_class(asset_type, symbols, start_date, end_date):
    """
    Fetch historical data for a specific asset class (stocks, indices, forex, crypto).
    """
    if asset_type == "stocks":
        data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker')
    elif asset_type == "forex":
        # Forex tickers in yfinance usually use pair symbols like "EURUSD=X"
        data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker')
    elif asset_type == "crypto":
        # Crypto tickers in yfinance usually use symbols like "BTC-USD"
        data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker')
    elif asset_type == "indices":
        # Index tickers like "^GSPC" for S&P 500
        data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker')
    else:
        raise ValueError("Invalid asset type. Choose from: stocks, indices, forex, crypto")
    return data

def get_realtime_data_for_asset_class(asset_type, symbols):
    """
    Fetch real-time data for a specific asset class (stocks, indices, forex, crypto).
    """
    results = {}
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        latest_data = stock.history(period="1d")
        results[symbol] = {
            "price": latest_data["Close"].iloc[-1],
            "timestamp": latest_data.index[-1]
        }
    return results

# Setup cache directory
memory = Memory("./data_cache", verbose=0)

@memory.cache
def get_cached_historical_data(asset_type, symbols, start_date, end_date):
    return get_historical_data_for_asset_class(asset_type, symbols, start_date, end_date)
