import matplotlib.pyplot as plt
import pandas as pd

def plot_asset_performance(data, asset_classes, save_path=None):
    """
    Plot normalized performance of various asset classes over time.

    Parameters:
        data (dict): Dictionary with asset class names as keys and DataFrames as values.
        asset_classes (list): List of asset class names.
        save_path (str): Path to save the plot (optional).
    """
    plt.figure(figsize=(12, 6))

    for asset_class in asset_classes:
        df = data[asset_class]
        
        # Extract 'Close' prices and normalize them to start at 100
        close_prices = df.xs('Close', axis=1, level=1).copy()
        normalized_prices = (close_prices / close_prices.iloc[0]) * 100

        # Plot each asset's normalized performance
        for ticker in normalized_prices.columns:
            plt.plot(normalized_prices.index, normalized_prices[ticker], label=f"{ticker} ({asset_class})")

    plt.title("Asset-Specific Performance Comparison (Normalized)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price (Base = 100)")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()
