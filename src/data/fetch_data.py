"""
This module fetches real-world financial data using the yfinance library.

Supported Asset Classes:
1. Stocks
2. Indices
3. Forex (e.g., "EURUSD=X")
4. Cryptocurrencies (e.g., "BTC-USD")

Functions:
1. get_historical_data: Fetch historical OHLCV data for a single symbol.
2. get_realtime_data: Fetch the latest price for a single symbol.
3. get_batch_historical_data: Fetch historical data for multiple tickers.
4. get_batch_realtime_data: Fetch real-time data for multiple tickers.
5. get_historical_data_for_asset_class: Fetch historical data by asset class.
6. get_realtime_data_for_asset_class: Fetch real-time data by asset class.
7. get_cached_historical_data: Cached historical data retrieval for efficiency.
"""

from joblib import Memory
import yfinance as yf

# Setup cache directory for joblib caching
memory = Memory("./data_cache", verbose=0)

def get_historical_data(ticker, start_date, end_date):
    """
    Fetch historical OHLCV data for a single symbol.
    """
    return yf.download(ticker, start=start_date, end=end_date)

def get_realtime_data(ticker):
    """
    Fetch the latest closing price and timestamp for a single symbol.
    """
    stock = yf.Ticker(ticker)
    latest_data = stock.history(period="1d")
    return {
        "price": latest_data["Close"].iloc[-1],
        "timestamp": latest_data.index[-1]
    }

def get_batch_historical_data(tickers, start_date, end_date):
    """
    Fetch historical OHLCV data for multiple symbols.
    """
    return yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

def get_batch_realtime_data(tickers):
    """
    Fetch the latest prices for multiple symbols.
    """
    results = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            latest_data = stock.history(period="1d")
            results[ticker] = {
                "price": latest_data["Close"].iloc[-1],
                "timestamp": latest_data.index[-1]
            }
        except Exception as e:
            results[ticker] = {"error": str(e)}
    return results

def get_historical_data_for_asset_class(asset_type, symbols, start_date, end_date):
    """
    Fetch historical OHLCV data for a specific asset class.

    Parameters:
        asset_type (str): "stocks", "indices", "forex", or "crypto".
        symbols (list): List of ticker symbols.
        start_date (str): Start date for historical data (YYYY-MM-DD).
        end_date (str): End date for historical data (YYYY-MM-DD).

    Returns:
        pd.DataFrame: Historical OHLCV data.
    """
    if asset_type not in {"stocks", "indices", "forex", "crypto"}:
        raise ValueError("Invalid asset type. Choose from: stocks, indices, forex, crypto.")
    
    try:
        return yf.download(symbols, start=start_date, end=end_date, group_by="ticker")
    except Exception as e:
        print(f"Error fetching data for {asset_type}: {e}")
        return None

def get_realtime_data_for_asset_class(asset_type, symbols):
    """
    Fetch real-time data for a specific asset class (stocks, indices, forex, crypto).
    Returns:
        dict: Real-time prices and timestamps.
    """
    results = {}
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            latest_data = stock.history(period="1d")
            if not latest_data.empty:
                results[symbol] = {
                    "price": latest_data["Close"].iloc[-1],
                    "timestamp": latest_data.index[-1]
                }
            else:
                print(f"No real-time data for {symbol}")
                results[symbol] = {"price": None, "timestamp": None}
        except Exception as e:
            print(f"Failed to fetch real-time data for {symbol}: {e}")
            results[symbol] = {"price": None, "timestamp": None}
    return results

@memory.cache
def get_cached_historical_data(asset_type, symbols, start_date, end_date):
    """
    Cached version of get_historical_data_for_asset_class for efficiency.
    """
    return get_historical_data_for_asset_class(asset_type, symbols, start_date, end_date)
