# src/visualizations/plot_monte_carlo_with_metrics.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_monte_carlo_with_metrics(portfolio_values, metrics, save_path=None):
    """
    Plot Monte Carlo simulated portfolio paths with calculated metrics.

    Parameters:
        portfolio_values (pd.DataFrame): Simulated portfolio values over time.
        metrics (pd.DataFrame): Metrics calculated for the Monte Carlo simulations.
        save_path (str): Path to save the plot (optional).
    """
    plt.figure(figsize=(12, 6))

    # Plot all portfolio paths
    plt.plot(portfolio_values, alpha=0.1, color="blue", label="Simulated Paths")

    # Overlay metrics as annotations or thresholds
    sharpe_ratio_mean = metrics["Sharpe Ratio"].mean()
    omega_ratio_mean = metrics["Omega Ratio"].mean()

    plt.axhline(y=portfolio_values.iloc[0, 0], color="red", linestyle="--", label="Initial Portfolio Value")
    plt.axhline(y=portfolio_values.mean().mean(), color="purple", linestyle="-.", label="Mean Portfolio Value")

    plt.title("Monte Carlo Simulated Portfolio Paths with Risk Metrics")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.grid(True)

    # Add metrics in the legend
    plt.figtext(0.15, 0.01, f"Sharpe Ratio: {sharpe_ratio_mean:.2f} | Omega Ratio: {omega_ratio_mean:.2f}", 
                fontsize=10, ha="left", bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()
