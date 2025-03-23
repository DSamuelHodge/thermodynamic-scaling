"""
Renormalization group analysis of neural network weights.
"""

import numpy as np
from scipy.linalg import svd
from scipy.spatial.distance import pdist, squareform
import warnings


def block_spin_renormalization(weight_matrix, block_size):
    """
    Perform simple block-spin like renormalization on weight matrix.
    
    Parameters:
    -----------
    weight_matrix : array_like
        2D weight matrix to renormalize.
    block_size : int
        Size of the blocks to average.
    
    Returns:
    --------
    array_like
        Renormalized weight matrix.
    """
    if weight_matrix.ndim != 2:
        raise ValueError("Weight matrix must be 2D")
    
    rows, cols = weight_matrix.shape
    
    # Calculate the new dimensions after blocking
    new_rows = rows // block_size
    new_cols = cols // block_size
    
    # If the matrix dimensions are not multiples of block_size, warn and truncate
    if rows % block_size != 0 or cols % block_size != 0:
        warnings.warn(f"Matrix dimensions ({rows}, {cols}) are not multiples of block_size {block_size}. "
                      f"Truncating to ({new_rows * block_size}, {new_cols * block_size}).")
        weight_matrix = weight_matrix[:new_rows * block_size, :new_cols * block_size]
    
    # Reshape and average blocks
    blocked_weights = weight_matrix.reshape(new_rows, block_size, new_cols, block_size)
    renormalized_weights = blocked_weights.mean(axis=(1, 3))
    
    return renormalized_weights


def renormalization_flow(weight_matrix, temps, n_steps=5, block_size=2):
    """
    Track how thermodynamic quantities change under RG flow.
    
    Parameters:
    -----------
    weight_matrix : array_like
        Original weight matrix.
    temps : array_like
        Array of temperatures to evaluate.
    n_steps : int, optional
        Number of renormalization steps.
    block_size : int, optional
        Size of the blocks for renormalization.
    
    Returns:
    --------
    dict
        Dictionary with RG flow results.
    """
    from ..thermodynamics import compute_thermal_properties, find_critical_point
    
    # Initialize storage for RG flow data
    flow_data = {
        'temperatures': temps,
        'matrices': [weight_matrix],
        'specific_heat': [],
        'critical_temps': [],
        'critical_exponents': []
    }
    
    current_matrix = weight_matrix
    
    # Compute properties for the original matrix
    thermal_props = compute_thermal_properties(current_matrix, temps)
    flow_data['specific_heat'].append(thermal_props['specific_heat'])
    
    # Find the critical temperature
    critical_temp, _ = find_critical_point(temps, thermal_props['specific_heat'])
    flow_data['critical_temps'].append(critical_temp)
    
    # Perform RG steps
    for step in range(n_steps):
        # Skip the last step if the matrix is too small
        min_dim = min(current_matrix.shape)
        if min_dim <= block_size:
            warnings.warn(f"Matrix too small for further renormalization at step {step}. Stopping.")
            break
        
        # Apply renormalization
        current_matrix = block_spin_renormalization(current_matrix, block_size)
        flow_data['matrices'].append(current_matrix)
        
        # Compute thermal properties
        thermal_props = compute_thermal_properties(current_matrix, temps)
        flow_data['specific_heat'].append(thermal_props['specific_heat'])
        
        # Find the critical temperature
        critical_temp, _ = find_critical_point(temps, thermal_props['specific_heat'])
        flow_data['critical_temps'].append(critical_temp)
    
    # Calculate critical exponents
    if len(flow_data['critical_temps']) > 1:
        exponents = []
        for i in range(1, len(flow_data['critical_temps'])):
            # Calculate the ratio of critical temperatures
            t_ratio = flow_data['critical_temps'][i] / flow_data['critical_temps'][i-1]
            # Estimate the critical exponent (assuming block_size scaling)
            nu_estimate = np.log(t_ratio) / np.log(block_size)
            exponents.append(nu_estimate)
        
        flow_data['critical_exponents'] = exponents
    
    return flow_data


def calculate_correlation_length(weight_matrix, temperature):
    """
    Estimate correlation length in the weight matrix at a given temperature.
    
    Parameters:
    -----------
    weight_matrix : array_like
        Weight matrix to analyze.
    temperature : float
        Temperature parameter.
    
    Returns:
    --------
    float
        Estimated correlation length.
    """
    # Ensure the matrix is square for simplicity
    if weight_matrix.shape[0] != weight_matrix.shape[1]:
        min_dim = min(weight_matrix.shape)
        weight_matrix = weight_matrix[:min_dim, :min_dim]
    
    # Calculate the correlation function
    n = weight_matrix.shape[0]
    
    # Calculate the distance matrix
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)
    distances = np.sqrt((X - X.T)**2 + (Y - Y.T)**2)
    
    # Calculate Boltzmann weights
    beta = 1.0 / temperature
    energies = weight_matrix
    max_energy = np.max(energies)
    boltzmann_weights = np.exp(-beta * (energies - max_energy))
    
    # Calculate correlation function
    mean_weight = np.mean(boltzmann_weights)
    centered_weights = boltzmann_weights - mean_weight
    correlation = np.zeros_like(distances)
    
    # Calculate correlations for each distance
    for d in np.unique(distances):
        mask = (distances == d)
        if np.sum(mask) > 0:
            correlation_at_d = np.mean(centered_weights[mask] * centered_weights.T[mask])
            correlation[mask] = correlation_at_d
    
    # Calculate average correlation as a function of distance
    unique_distances = np.sort(np.unique(distances))[1:]  # Skip d=0
    correlations = []
    
    for d in unique_distances:
        mask = (distances == d)
        correlations.append(np.mean(correlation[mask]))
    
    correlations = np.array(correlations)
    
    # Fit exponential decay to estimate correlation length
    from scipy.optimize import curve_fit
    
    def exp_decay(r, xi, a):
        return a * np.exp(-r / xi)
    
    try:
        popt, _ = curve_fit(exp_decay, unique_distances, np.abs(correlations), 
                          p0=[1.0, correlations[0]], bounds=([0, 0], [np.inf, np.inf]))
        correlation_length = popt[0]
    except:
        # If fitting fails, use a simple estimate based on the decay
        log_corr = np.log(np.abs(correlations) + 1e-10)
        slope, _ = np.polyfit(unique_distances[:min(5, len(unique_distances))], log_corr[:min(5, len(unique_distances))], 1)
        correlation_length = -1.0 / slope if slope < 0 else 1.0
    
    return correlation_length


def scaling_dimension_analysis(weight_matrices, temps):
    """
    Analyze how critical exponents scale with the network size.
    
    Parameters:
    -----------
    weight_matrices : dict
        Dictionary of weight matrices with different sizes.
    temps : array_like
        Array of temperatures to evaluate.
    
    Returns:
    --------
    dict
        Dictionary with scaling dimension results.
    """
    from ..thermodynamics import compute_thermal_properties, find_critical_point
    
    results = {
        'sizes': [],
        'critical_temps': [],
    }
    
    for name, matrix in weight_matrices.items():
        if matrix.ndim != 2:
            continue
        
        size = max(matrix.shape)
        results['sizes'].append(size)
        
        # Compute thermal properties
        thermal_props = compute_thermal_properties(matrix, temps)
        
        # Find critical temperature
        critical_temp, _ = find_critical_point(temps, thermal_props['specific_heat'])
        results['critical_temps'].append(critical_temp)
    
    # Sort by size
    sorted_indices = np.argsort(results['sizes'])
    results['sizes'] = np.array(results['sizes'])[sorted_indices]
    results['critical_temps'] = np.array(results['critical_temps'])[sorted_indices]
    
    # Fit power law to estimate the scaling dimension
    if len(results['sizes']) > 1:
        from scipy.optimize import curve_fit
        
        def power_law(L, Tc_inf, nu):
            return Tc_inf * (1 - L**(-1/nu))
        
        try:
            popt, pcov = curve_fit(power_law, results['sizes'], results['critical_temps'], 
                                 p0=[max(results['critical_temps']), 1.0], 
                                 bounds=([0, 0], [np.inf, np.inf]))
            
            results['Tc_infinite'] = popt[0]
            results['nu'] = popt[1]
            results['fit_uncertainty'] = np.sqrt(np.diag(pcov))
        except:
            # If fitting fails, use simple linear regression on log-log scale
            log_sizes = np.log(results['sizes'])
            log_Tc_shift = np.log(max(results['critical_temps']) - results['critical_temps'])
            
            slope, intercept = np.polyfit(log_sizes, log_Tc_shift, 1)
            
            results['nu'] = -1.0 / slope if slope < 0 else 1.0
            results['fit_uncertainty'] = [0.0, 0.0]  # Placeholder
            results['Tc_infinite'] = max(results['critical_temps'])
    
    return results


def fixed_point_analysis(weight_matrix, block_size=2, max_steps=10, convergence_threshold=1e-4):
    """
    Identify potential RG fixed points in the weight space.
    
    Parameters:
    -----------
    weight_matrix : array_like
        Weight matrix to analyze.
    block_size : int, optional
        Size of the blocks for renormalization.
    max_steps : int, optional
        Maximum number of RG steps.
    convergence_threshold : float, optional
        Threshold for determining convergence.
    
    Returns:
    --------
    dict
        Dictionary with fixed point analysis results.
    """
    results = {
        'converged': False,
        'steps_to_convergence': max_steps,
        'final_matrix': None,
        'flow_path': [],
        'similarity_metrics': []
    }
    
    current_matrix = weight_matrix.copy()
    results['flow_path'].append(current_matrix.copy())
    
    for step in range(max_steps):
        # Apply renormalization
        renorm_matrix = block_spin_renormalization(current_matrix, block_size)
        
        # Rescale to match the statistical properties of the previous matrix
        std_ratio = np.std(current_matrix) / np.std(renorm_matrix)
        renorm_matrix *= std_ratio
        
        # Calculate similarity between consecutive matrices
        # For this, we use the cosine similarity between flattened matrices
        current_flat = current_matrix.flatten() - np.mean(current_matrix)
        renorm_flat = renorm_matrix.flatten() - np.mean(renorm_matrix)
        
        similarity = np.dot(current_flat, renorm_flat) / (np.linalg.norm(current_flat) * np.linalg.norm(renorm_flat))
        results['similarity_metrics'].append(similarity)
        
        # Check for convergence (similarity close to 1)
        if abs(similarity - 1.0) < convergence_threshold:
            results['converged'] = True
            results['steps_to_convergence'] = step + 1
            break
        
        # Update current matrix and store in flow path
        current_matrix = renorm_matrix.copy()
        results['flow_path'].append(current_matrix.copy())
        
        # Break if matrix becomes too small
        if min(current_matrix.shape) <= block_size:
            break
    
    results['final_matrix'] = current_matrix.copy()
    
    return results