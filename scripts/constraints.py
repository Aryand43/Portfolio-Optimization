import numpy as np

def transaction_cost_penalty(weights, initial_weights, transaction_costs):
    """
    Calculate the penalty for transaction costs during rebalancing.
    
    Args:
        weights (np.array): Current portfolio weights.
        initial_weights (np.array): Portfolio weights before rebalancing.
        transaction_costs (np.array): Per-asset transaction cost rates.

    Returns:
        float: Total transaction cost penalty.
    """
    return np.sum(np.abs(weights - initial_weights) * transaction_costs)
