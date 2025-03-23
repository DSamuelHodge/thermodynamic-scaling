"""
Information-theoretic analysis of neural network weights.
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import mutual_info_score
import warnings


def calculate_entropy(weights, temperature):
    """
    Calculate the entropy of the weight distribution at a given temperature.
    
    Parameters:
    -----------
    weights : array_like
        Weight matrix or flattened weights.
    temperature : float
        Temperature parameter.
    
    Returns:
    --------
    float
        Entropy value.
    """
    # Flatten weights if necessary
    if weights.ndim > 1:
        weights = weights.flatten()
    
    # Calculate Boltzmann probabilities
    beta = 1.0 / temperature
    energies = weights
    
    # Handle numerical stability by subtracting the maximum energy
    max_energy = np.max(energies)
    exp_terms = np.exp(-beta * (energies - max_energy))
    Z = np.sum(exp_terms)
    probs = exp_terms / Z
    
    # Calculate entropy
    entropy_val = -np.sum(probs * np.log(probs + 1e-10))
    
    return entropy_val


def fisher_information(weights, temperature, delta=1e-5):
    """
    Calculate the Fisher information with respect to temperature.
    
    Parameters:
    -----------
    weights : array_like
        Weight matrix or flattened weights.
    temperature : float
        Temperature parameter.
    delta : float, optional
        Small increment for numerical differentiation.
    
    Returns:
    --------
    float
        Fisher information value.
    """
    # Flatten weights if necessary
    if weights.ndim > 1:
        weights = weights.flatten()
    
    # Calculate entropy at T and T+delta
    entropy_T = calculate_entropy(weights, temperature)
    entropy_T_plus = calculate_entropy(weights, temperature + delta)
    
    # Numerical approximation of the derivative of entropy with respect to T
    dS_dT = (entropy_T_plus - entropy_T) / delta
    
    # Fisher information is proportional to the square of this derivative
    fisher_info = dS_dT**2
    
    return fisher_info


def kullback_leibler_divergence(weights1, weights2, temperature):
    """
    Calculate the Kullback-Leibler divergence between two weight distributions.
    
    Parameters:
    -----------
    weights1 : array_like
        First weight matrix or flattened weights.
    weights2 : array_like
        Second weight matrix or flattened weights.
    temperature : float
        Temperature parameter.
    
    Returns:
    --------
    float
        KL divergence value.
    """
    # Flatten weights if necessary
    if weights1.ndim > 1:
        weights1 = weights1.flatten()
    if weights2.ndim > 1:
        weights2 = weights2.flatten()
    
    # Ensure both weight arrays have the same size
    if len(weights1) != len(weights2):
        warnings.warn("Weight arrays have different sizes. Truncating to the smaller size.")
        min_len = min(len(weights1), len(weights2))
        weights1 = weights1[:min_len]
        weights2 = weights2[:min_len]
    
    # Calculate Boltzmann probabilities for both distributions
    beta = 1.0 / temperature
    
    # For weights1
    energies1 = weights1
    max_energy1 = np.max(energies1)
    exp_terms1 = np.exp(-beta * (energies1 - max_energy1))
    Z1 = np.sum(exp_terms1)
    probs1 = exp_terms1 / Z1
    
    # For weights2
    energies2 = weights2
    max_energy2 = np.max(energies2)
    exp_terms2 = np.exp(-beta * (energies2 - max_energy2))
    Z2 = np.sum(exp_terms2)
    probs2 = exp_terms2 / Z2
    
    # Calculate KL divergence
    kl_div = np.sum(probs1 * np.log((probs1 + 1e-10) / (probs2 + 1e-10)))
    
    return kl_div


def mutual_information_layer_pairs(model_weights, layer_indices, temperature):
    """
    Calculate mutual information between layer pairs at a given temperature.
    
    Parameters:
    -----------
    model_weights : dict
        Dictionary containing model weights.
    layer_indices : list
        List of layer indices to analyze.
    temperature : float
        Temperature parameter.
    
    Returns:
    --------
    dict
        Dictionary of mutual information values between layer pairs.
    """
    mi_values = {}
    
    for i in layer_indices:
        for j in layer_indices:
            if i >= j:
                continue  # Skip duplicate pairs and self-comparisons
            
            # Get weight matrices
            if f'layer_{i}' in model_weights and f'layer_{j}' in model_weights:
                weights_i = model_weights[f'layer_{i}'].get('self_attention.query', None)
                weights_j = model_weights[f'layer_{j}'].get('self_attention.query', None)
                
                if weights_i is None or weights_j is None:
                    continue
                
                # Flatten weights
                weights_i_flat = weights_i.flatten()
                weights_j_flat = weights_j.flatten()
                
                # Ensure both arrays have the same size
                min_len = min(len(weights_i_flat), len(weights_j_flat))
                weights_i_flat = weights_i_flat[:min_len]
                weights_j_flat = weights_j_flat[:min_len]
                
                # Calculate Boltzmann probabilities
                beta = 1.0 / temperature
                
                # For weights_i
                max_energy_i = np.max(weights_i_flat)
                exp_terms_i = np.exp(-beta * (weights_i_flat - max_energy_i))
                Z_i = np.sum(exp_terms_i)
                probs_i = exp_terms_i / Z_i
                
                # For weights_j
                max_energy_j = np.max(weights_j_flat)
                exp_terms_j = np.exp(-beta * (weights_j_flat - max_energy_j))
                Z_j = np.sum(exp_terms_j)
                probs_j = exp_terms_j / Z_j
                
                # Discretize probabilities for mutual information calculation
                bins = 100
                probs_i_binned = np.digitize(probs_i, np.linspace(0, np.max(probs_i), bins))
                probs_j_binned = np.digitize(probs_j, np.linspace(0, np.max(probs_j), bins))
                
                # Calculate mutual information
                mi = mutual_info_score(probs_i_binned, probs_j_binned)
                mi_values[f'{i}_{j}'] = mi
    
    return mi_values


def information_bottleneck_analysis(weights, temperature_range, n_temps=20):
    """
    Perform information bottleneck analysis across a temperature range.
    
    Parameters:
    -----------
    weights : array_like
        Weight matrix to analyze.
    temperature_range : tuple
        (min_temp, max_temp) range to analyze.
    n_temps : int, optional
        Number of temperature points to evaluate.
    
    Returns:
    --------
    dict
        Dictionary containing temperatures, entropy values, and Fisher information.
    """
    temperatures = np.linspace(temperature_range[0], temperature_range[1], n_temps)
    entropies = np.zeros(n_temps)
    fisher_infos = np.zeros(n_temps)
    
    for i, temp in enumerate(temperatures):
        entropies[i] = calculate_entropy(weights, temp)
        fisher_infos[i] = fisher_information(weights, temp)
    
    return {
        'temperatures': temperatures,
        'entropy': entropies,
        'fisher_information': fisher_infos
    }


def complexity_measures(weights, temperature):
    """
    Calculate various complexity measures based on information theory.
    
    Parameters:
    -----------
    weights : array_like
        Weight matrix to analyze.
    temperature : float
        Temperature parameter.
    
    Returns:
    --------
    dict
        Dictionary containing complexity measures.
    """
    # Flatten weights if necessary
    if weights.ndim > 1:
        weights = weights.flatten()
    
    # Calculate Boltzmann probabilities
    beta = 1.0 / temperature
    energies = weights
    max_energy = np.max(energies)
    exp_terms = np.exp(-beta * (energies - max_energy))
    Z = np.sum(exp_terms)
    probs = exp_terms / Z
    
    # Shannon entropy
    shannon_entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # Renyi entropy (order 2)
    renyi_entropy = -np.log(np.sum(probs**2))
    
    # Effective dimensions
    effective_dim = np.exp(shannon_entropy)
    
    # Participation ratio (similar to inverse Simpson index)
    participation_ratio = 1.0 / np.sum(probs**2)
    
    return {
        'shannon_entropy': shannon_entropy,
        'renyi_entropy': renyi_entropy,
        'effective_dimension': effective_dim,
        'participation_ratio': participation_ratio
    }