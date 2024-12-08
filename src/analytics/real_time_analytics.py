import time
from src.data.fetch_data import get_realtime_data_for_asset_class

def run_real_time_analytics(asset_type, symbols, interval=10):
    """
    Fetch and display real-time prices and basic metrics for given assets.
    Parameters:
        asset_type (str): Type of asset (stocks, indices, forex, crypto).
        symbols (list): List of asset symbols.
        interval (int): Time interval for fetching real-time data (seconds).
    """
    print(f"Fetching real-time data for {symbols} ({asset_type})...")
    print("-" * 60)
    try:
        while True:
            data = get_realtime_data_for_asset_class(asset_type, symbols)
            for symbol, info in data.items():
                if info["price"] is not None:
                    print(f"{symbol}: ${info['price']:.2f} (as of {info['timestamp']})")
                else:
                    print(f"{symbol}: Failed to fetch data.")
            print("-" * 60)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Real-time analytics stopped.")

if __name__ == "__main__":
    # Example: Run real-time analytics for stocks and forex
    run_real_time_analytics("stocks", ["AAPL", "MSFT"], interval=15)
    # Uncomment for other assets:
    # run_real_time_analytics("forex", ["EURUSD=X", "GBPUSD=X"], interval=15)
    # run_real_time_analytics("crypto", ["BTC-USD", "ETH-USD"], interval=15)
