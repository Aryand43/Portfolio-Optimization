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
