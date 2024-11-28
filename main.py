import yfinance as yf
import pandas as pd

#List of Stock Tickers (e.g S&P 500 tech stocks)
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']

#Fetch Historical Data
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    data.to_csv('data/stock_prices.csv')
    return data

#Main Function
if __name__ == "__main__":
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    print("Fetching data...")
    data = fetch_data(tickers, start_date, end_date)
    print("Data fetched successfully!")
