# src/visualizations/plot_monte_carlo_with_metrics.py

import matplotlib.pyplot as plt
import pandas as pd

def plot_monte_carlo_with_metrics(portfolio_values, metrics_df, save_path=None):
    """
    Plot Monte Carlo simulated portfolio paths and overlay risk metrics.

    Parameters:
        portfolio_values (pd.DataFrame): Simulated portfolio values over time.
        metrics_df (pd.DataFrame): Calculated metrics (Sharpe, VaR, CVaR, etc.).
        save_path (str): Path to save the figure (optional).
    """
    plt.figure(figsize=(12, 6))

    # Plot all simulated paths without labels to avoid clutter
    plt.plot(portfolio_values, alpha=0.1, color="blue", linewidth=0.5)

    # Overlay mean metrics summary
    sharpe_ratio = metrics_df["Sharpe Ratio"].mean()
    var_95 = metrics_df["VaR (95%)"].mean()
    cvar_95 = metrics_df["CVaR (95%)"].mean()
    max_drawdown = metrics_df["Max Drawdown"].mean()
    omega_ratio = metrics_df["Omega Ratio"].mean()

    metrics_text = (f"Mean Sharpe Ratio: {sharpe_ratio:.2f}\n"
                    f"Mean VaR (95%): {var_95:.2f}\n"
                    f"Mean CVaR (95%): {cvar_95:.2f}\n"
                    f"Mean Max Drawdown: {max_drawdown:.2f}\n"
                    f"Mean Omega Ratio: {omega_ratio:.2f}")

    # Add a text box with the metrics
    plt.text(0.02, 0.85, metrics_text, transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.title("Monte Carlo Simulated Portfolio Paths with Risk Metrics")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.grid(True)

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()