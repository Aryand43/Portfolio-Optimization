from src.simulations.monte_carlo import monte_carlo_simulation
import pandas as pd

def test_monte_carlo_simulation():
    """
    Test Monte Carlo simulation with synthetic returns.
    """
    data = pd.DataFrame({
        "Asset1": [0.01, 0.02, -0.01, 0.03],
        "Asset2": [0.02, 0.01, 0.0, -0.02]
    })
    simulations = monte_carlo_simulation(data)
    assert simulations.shape[0] > 0, "Simulation failed to generate portfolio paths"
    assert simulations.shape[1] == 1000, "Incorrect number of simulations"
