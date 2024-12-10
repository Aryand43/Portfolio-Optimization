# Portfolio Optimization Tool

## Overview

This project provides a **Portfolio Optimization Tool** with an intuitive GUI for financial data analysis, Monte Carlo simulations, and risk metric calculations. The tool is designed for quantitative finance professionals, investors, and students.

---

## Features

1. **Historical Data Fetching**
   - Fetch historical price data for stocks, indices, forex, and cryptocurrencies using `yfinance`.
   - Specify asset ticker, start date, and end date.

2. **Monte Carlo Simulation**
   - Perform Monte Carlo simulations to forecast portfolio performance.
   - Key metrics include:
     - Sharpe Ratio
     - Value at Risk (VaR)
     - Conditional Value at Risk (CVaR)
     - Maximum Drawdown (MDD)
     - Omega Ratio
   - Visualize simulated portfolio paths with a clean graph.

3. **Risk Metrics Calculation**
   - Calculate financial metrics like:
     - Sharpe Ratio
     - VaR (95%)
     - CVaR (95%)
     - Max Drawdown
     - Omega Ratio
   - Display results in a clean, tabular format.

4. **Dynamic GUI Interface**
   - Simple and user-friendly interface using **Tkinter**.
   - Resizable interface that adjusts with the window size.

---

## File Structure

```plaintext
Portfolio-Optimization/
│
├── app.py                     # Main GUI application
├── README.md                  # Project documentation
├── requirements.txt           # Dependencies
├── setup.py                   # Project setup (optional)
│
├── src/                       # Source code directory
│   ├── data/                  # Data fetching module
│   │   └── fetch_data.py
│   ├── simulations/           # Monte Carlo simulations
│   │   └── monte_carlo.py
│   ├── visualizations/        # Plotting functions
│   │   └── plot_monte_carlo_with_metrics.py
│   └── calculations/          # Financial metrics calculations
│       └── metrics.py
│
├── tests/                     # Unit tests
│   ├── test_fetch_data.py
│   ├── test_monte_carlo.py
│   └── test_calculations.py
│
└── data/                      # Data cache (if any)
