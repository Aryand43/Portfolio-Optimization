import tkinter as tk
from tkinter import ttk, messagebox
from src.data.fetch_data import get_historical_data
from src.simulations.monte_carlo import monte_carlo_with_metrics
from src.visualizations.plot_monte_carlo_with_metrics import plot_monte_carlo_with_metrics
from src.calculations.metrics import calculate_sharpe_ratio, calculate_max_drawdown, calculate_var, calculate_cvar, calculate_omega_ratio
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PortfolioGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Portfolio Optimization GUI")

        # Configure dynamic resizing
        for i in range(9):  # Configure rows to resize dynamically
            root.grid_rowconfigure(i, weight=1)
        root.grid_columnconfigure(0, weight=1)  # Left column
        root.grid_columnconfigure(1, weight=1)  # Right column

        # Input Fields
        ttk.Label(root, text="Enter Asset Ticker (e.g., AAPL, BTC-USD):").grid(row=0, column=0, sticky="w")
        self.asset_entry = ttk.Entry(root, width=50)
        self.asset_entry.grid(row=0, column=1, sticky="ew")

        ttk.Label(root, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, sticky="w")
        self.start_date_entry = ttk.Entry(root, width=50)
        self.start_date_entry.grid(row=1, column=1, sticky="ew")

        ttk.Label(root, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, sticky="w")
        self.end_date_entry = ttk.Entry(root, width=50)
        self.end_date_entry.grid(row=2, column=1, sticky="ew")

        # Buttons
        self.fetch_button = ttk.Button(root, text="Fetch Historical Data", command=self.fetch_data)
        self.fetch_button.grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")

        self.run_monte_carlo_button = ttk.Button(root, text="Run Monte Carlo Simulation", command=self.run_monte_carlo)
        self.run_monte_carlo_button.grid(row=4, column=0, columnspan=2, pady=5, sticky="ew")

        self.metrics_button = ttk.Button(root, text="Calculate Metrics", command=self.calculate_metrics)
        self.metrics_button.grid(row=5, column=0, columnspan=2, pady=5, sticky="ew")

        # Output Text
        self.output_text = tk.Text(root, height=8, width=70)
        self.output_text.grid(row=6, column=0, columnspan=2, sticky="nsew")

        # Table for Metrics
        self.tree = ttk.Treeview(root, columns=("Metric", "Value"), show="headings", height=5)
        self.tree.heading("Metric", text="Metric")
        self.tree.heading("Value", text="Value")
        self.tree.grid(row=7, column=0, columnspan=2, sticky="nsew")

        # Visualization Frame
        self.figure_frame = ttk.Frame(root)
        self.figure_frame.grid(row=8, column=0, columnspan=2, sticky="nsew")

    def fetch_data(self):
        try:
            ticker = self.asset_entry.get()
            start_date = self.start_date_entry.get()
            end_date = self.end_date_entry.get()

            data = get_historical_data(ticker, start_date, end_date)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Fetched Data ({start_date} to {end_date}):\n{data}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch data: {e}")

    def run_monte_carlo(self):
        try:
            self.output_text.insert(tk.END, "Running Monte Carlo simulation...\n")

            # Dummy returns for Monte Carlo
            test_returns = pd.DataFrame({
                "Asset1": [0.01, 0.02, -0.01, 0.03],
                "Asset2": [0.02, -0.01, 0.01, 0.04]
            })

            # Call Monte Carlo function
            metrics_df, portfolio_values = monte_carlo_with_metrics(test_returns)

            # Ensure 'Sharpe Ratio' exists
            if 'Sharpe Ratio' not in metrics_df.columns:
                raise KeyError("'Sharpe Ratio' is missing from metrics_df")

            # Pass both outputs to the visualization
            fig = plot_monte_carlo_with_metrics(portfolio_values, metrics_df)

            # Embed the plot
            canvas = FigureCanvasTkAgg(fig, self.figure_frame)
            canvas.get_tk_widget().pack(expand=True, fill="both")
            canvas.draw()

            # Allow user to close graph
            def close_figure():
                plt.close(fig)
                canvas.get_tk_widget().destroy()
            ttk.Button(self.figure_frame, text="Close Chart", command=close_figure).pack()

            self.output_text.insert(tk.END, "Monte Carlo Simulation Completed Successfully!\n")

        except KeyError as e:
            messagebox.showerror("Error", f"Monte Carlo Simulation failed: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected Error: {e}")

    def calculate_metrics(self):
        try:
            # Dummy returns for testing
            test_returns = pd.Series([0.01, 0.02, -0.01, 0.03])

            # Calculate metrics
            metrics = {
                "Sharpe Ratio": calculate_sharpe_ratio(test_returns),
                "Max Drawdown": calculate_max_drawdown(test_returns.cumsum()),
                "VaR (95%)": calculate_var(test_returns, confidence_level=0.95),
                "CVaR (95%)": calculate_cvar(test_returns, confidence_level=0.95),
                "Omega Ratio": calculate_omega_ratio(test_returns, threshold=0.01)
            }

            # Clear existing table rows
            for row in self.tree.get_children():
                self.tree.delete(row)

            # Insert metrics into the table
            for metric, value in metrics.items():
                self.tree.insert("", "end", values=(metric, round(value, 4)))

        except Exception as e:
            messagebox.showerror("Error", f"Metrics Calculation failed: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PortfolioGUI(root)
    root.mainloop()
