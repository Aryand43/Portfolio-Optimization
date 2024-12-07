import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_monte_carlo_paths(portfolio_values, save_dir=None, file_name="monte_carlo_simulation.png", show=True):
    """
    Plot Monte Carlo simulated portfolio paths, save the figure, and display it.

    Parameters:
        portfolio_values (pd.DataFrame): Simulated portfolio values over time.
        save_dir (str): Directory path to save the figure (optional).
        file_name (str): Name of the file to save the plot as (default: 'monte_carlo_simulation.png').
        show (bool): Whether to display the plot after saving (default: True).
    """
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, alpha=0.1, color="blue")
    plt.title("Monte Carlo Simulated Portfolio Paths")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.grid(True)

    # Save the plot if save directory is specified
    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
            save_path = os.path.join(save_dir, file_name)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot successfully saved to: {save_path}")
        except Exception as e:
            print(f"Error saving the plot: {e}")

    # Show the plot if enabled
    if show:
        plt.show()
    else:
        print("Plot display is disabled.")
