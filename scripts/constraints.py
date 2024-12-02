import numpy as np

def validate_inputs(weights, initial_weights, transaction_costs):
    """
    Validate inputs for the transaction cost penalty calculation.
    
    Args:
        weights (np.array): Current portfolio weights.
        initial_weights (np.array): Portfolio weights before rebalancing.
        transaction_costs (np.array): Per-asset transaction cost rates.
    
    Raises:
        ValueError: If any input is invalid (e.g., mismatched shapes, NaN values).
    """
    if len(weights) != len(initial_weights):
        raise ValueError("Weights and initial_weights must have the same length.")
    if len(weights) != len(transaction_costs):
        raise ValueError("Weights and transaction_costs must have the same length.")
    if np.any(np.isnan(weights)) or np.any(np.isnan(initial_weights)) or np.any(np.isnan(transaction_costs)):
        raise ValueError("Inputs must not contain NaN values.")

def transaction_cost_penalty(weights, initial_weights, transaction_costs, normalize=False):
    """
    Calculate the penalty for transaction costs during rebalancing.
    
    Args:
        weights (np.array): Current portfolio weights.
        initial_weights (np.array): Portfolio weights before rebalancing.
        transaction_costs (np.array): Per-asset transaction cost rates.
        normalize (bool): If True, return penalty as a fraction of the total portfolio weight (default False).

    Returns:
        float: Total transaction cost penalty (normalized if specified).
    """
    # Validate inputs
    validate_inputs(weights, initial_weights, transaction_costs)

    # Handle edge cases
    if np.all(weights == initial_weights) or np.all(transaction_costs == 0):
        return 0.0

    # Calculate penalty
    penalty = np.sum(np.abs(weights - initial_weights) * transaction_costs)
    if normalize:
        return penalty / np.sum(weights)
    return penalty
