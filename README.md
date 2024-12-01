# Portfolio-Optimization
A Python-based portfolio optimization tool to maximize Sharpe Ratio using real market data.

## Features
- Calculate daily and annualized returns.
- Optimize portfolio weights to maximize Sharpe Ratio.
- Simulate portfolio performance using Monte Carlo simulations.
- Visualize the Efficient Frontier and portfolio allocations.
- Interactive Streamlit app for real-time analysis.

## How to Use
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Portfolio-Optimization.git
   cd Portfolio-Optimization
2. **Set up a virtual environment**:
   python -m venv venv
   source venv/bin/activate  #Windows: venv\Scripts\activate
3. **Install dependencies:**
   pip install -r requirements.txt
4. **Run the Streamlit app**
   streamlit run app.py

## Project Structure 
- **scripts/metrics.py**: Functions for portfolio calculations (returns, risk, covariance).
- **scripts/optimizer.py**: Optimization logic to maximize Sharpe Ratio.
- **scripts/simulations.py**: Monte Carlo simulation functions.
- **scripts/visualizations.py**: Plotting functions for visualizations.
- **app.py**: Streamlit app for interactivity.
- **main.py**: Script to run the project end-to-end.
   
## Next Steps
- **Add predictive modeling** (e.g., machine learning for stock price prediction).
- **Integrate real-time data fetching** (e.g., Alpha Vantage, IEX Cloud).
- **Deploy the app** on Streamlit Cloud or Heroku for public access.
