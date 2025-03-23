"""
Core thermodynamic calculations for analyzing neural network weights.
"""

import numpy as np
from scipy import stats
from scipy.signal import find_peaks
import warnings
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    warnings.warn("CuPy not available. GPU acceleration will not be used.")


def partition_function(energies, beta, use_gpu=False):
    """
    Compute partition function Z = sum(exp(-beta * E)) with numerical stability.
    
    Parameters:
    -----------
    energies : array_like
        Array of energy levels.
    beta : float
        Inverse temperature (1/T).
    use_gpu : bool, optional
        Whether to use GPU acceleration (requires CuPy).
    
    Returns:
    --------
    float
        Partition function value.
    """
    if use_gpu and HAS_GPU:
        # GPU implementation with CuPy
        energies_gpu = cp.asarray(energies)
        max_energy = cp.max(energies_gpu)
        return float(cp.sum(cp.exp(-beta * (energies_gpu - max_energy))) * cp.exp(-beta * max_energy))
    else:
        # CPU implementation with NumPy
        max_energy = np.max(energies)
        return float(np.sum(np.exp(-beta * (energies - max_energy))) * np.exp(-beta * max_energy))


def expectation_value(energies, beta, order=1, use_gpu=False):
    """
    Compute expectation value <E^order>_T using Boltzmann distribution.
    
    Parameters:
    -----------
    energies : array_like
        Array of energy levels.
    beta : float
        Inverse temperature (1/T).
    order : int, optional
        Order of the expectation value.
    use_gpu : bool, optional
        Whether to use GPU acceleration (requires CuPy).
    
    Returns:
    --------
    float
        Expectation value.
    """
    if use_gpu and HAS_GPU:
        energies_gpu = cp.asarray(energies)
        Z = partition_function(energies_gpu, beta, use_gpu=True)
        return float(cp.sum(energies_gpu**order * cp.exp(-beta * energies_gpu)) / Z)
    else:
        Z = partition_function(energies, beta, use_gpu=False)
        return float(np.sum(energies**order * np.exp(-beta * energies)) / Z)


def specific_heat(energies, temperature, k_B=1.0, use_gpu=False):
    """
    Compute specific heat C(T) = (<E^2>_T - <E>_T^2) / (k_B * T^2).
    
    Parameters:
    -----------
    energies : array_like
        Array of energy levels.
    temperature : float
        Temperature.
    k_B : float, optional
        Boltzmann constant (default: 1.0 for normalized units).
    use_gpu : bool, optional
        Whether to use GPU acceleration (requires CuPy).
    
    Returns:
    --------
    float
        Specific heat.
    """
    beta = 1.0 / (k_B * temperature)
    
    if use_gpu and HAS_GPU:
        energies_gpu = cp.asarray(energies)
        energy_expectation = expectation_value(energies_gpu, beta, order=1, use_gpu=True)
        energy_squared_expectation = expectation_value(energies_gpu, beta, order=2, use_gpu=True)
    else:
        energy_expectation = expectation_value(energies, beta, order=1, use_gpu=False)
        energy_squared_expectation = expectation_value(energies, beta, order=2, use_gpu=False)
    
    return (energy_squared_expectation - energy_expectation**2) / (k_B * temperature**2)


def entropy(energies, temperature, k_B=1.0, use_gpu=False):
    """
    Compute entropy S(T) = k_B * (log(Z) + beta * <E>).
    
    Parameters:
    -----------
    energies : array_like
        Array of energy levels.
    temperature : float
        Temperature.
    k_B : float, optional
        Boltzmann constant (default: 1.0 for normalized units).
    use_gpu : bool, optional
        Whether to use GPU acceleration (requires CuPy).
    
    Returns:
    --------
    float
        Entropy.
    """
    beta = 1.0 / (k_B * temperature)
    
    if use_gpu and HAS_GPU:
        energies_gpu = cp.asarray(energies)
        Z = partition_function(energies_gpu, beta, use_gpu=True)
        energy_expectation = expectation_value(energies_gpu, beta, order=1, use_gpu=True)
    else:
        Z = partition_function(energies, beta, use_gpu=False)
        energy_expectation = expectation_value(energies, beta, order=1, use_gpu=False)
    
    return k_B * (np.log(Z) + beta * energy_expectation)


def free_energy(energies, temperature, k_B=1.0, use_gpu=False):
    """
    Compute Helmholtz free energy F(T) = -k_B * T * log(Z).
    
    Parameters:
    -----------
    energies : array_like
        Array of energy levels.
    temperature : float
        Temperature.
    k_B : float, optional
        Boltzmann constant (default: 1.0 for normalized units).
    use_gpu : bool, optional
        Whether to use GPU acceleration (requires CuPy).
    
    Returns:
    --------
    float
        Free energy.
    """
    beta = 1.0 / (k_B * temperature)
    
    if use_gpu and HAS_GPU:
        energies_gpu = cp.asarray(energies)
        Z = partition_function(energies_gpu, beta, use_gpu=True)
    else:
        Z = partition_function(energies, beta, use_gpu=False)
    
    return -k_B * temperature * np.log(Z)


def compute_thermal_properties(weight_matrix, temps, k_B=1.0, use_gpu=False):
    """
    Compute thermal properties across a temperature range.
    
    Parameters:
    -----------
    weight_matrix : array_like
        Weight matrix to analyze.
    temps : array_like
        Array of temperatures to evaluate.
    k_B : float, optional
        Boltzmann constant (default: 1.0 for normalized units).
    use_gpu : bool, optional
        Whether to use GPU acceleration (requires CuPy).
    
    Returns:
    --------
    dict
        Dictionary containing temperatures and computed properties.
    """
    # Flatten the weight matrix to get the energy spectrum
    energies = weight_matrix.flatten()
    
    # Initialize arrays for results
    specific_heats = np.zeros_like(temps)
    energies_mean = np.zeros_like(temps)
    entropies = np.zeros_like(temps)
    free_energies = np.zeros_like(temps)
    
    # Compute properties for each temperature
    for i, T in enumerate(temps):
        beta = 1.0 / (k_B * T)
        
        # Specific heat
        specific_heats[i] = specific_heat(energies, T, k_B, use_gpu)
        
        # Mean energy
        energies_mean[i] = expectation_value(energies, beta=beta, order=1, use_gpu=use_gpu)
        
        # Entropy
        entropies[i] = entropy(energies, T, k_B, use_gpu)
        
        # Free energy
        free_energies[i] = free_energy(energies, T, k_B, use_gpu)
    
    # Return results as a dictionary
    return {
        'temperatures': temps,
        'specific_heat': specific_heats,
        'energy': energies_mean,
        'entropy': entropies,
        'free_energy': free_energies
    }


def find_critical_point(temps, specific_heats, smoothing=True, window_length=5):
    """
    Find the critical temperature and characteristics.
    
    Parameters:
    -----------
    temps : array_like
        Array of temperatures.
    specific_heats : array_like
        Array of specific heat values.
    smoothing : bool, optional
        Whether to apply smoothing to the specific heat curve.
    window_length : int, optional
        Window length for smoothing.
    
    Returns:
    --------
    float
        Critical temperature.
    float
        Peak height of the specific heat.
    """
    # Apply smoothing if requested
    if smoothing and len(specific_heats) >= window_length:
        from scipy.signal import savgol_filter
        specific_heats_smooth = savgol_filter(specific_heats, window_length, 2)
    else:
        specific_heats_smooth = specific_heats
    
    # Find peaks in the specific heat curve
    peaks, properties = find_peaks(specific_heats_smooth, height=0)
    
    if len(peaks) == 0:
        # If no clear peak, use the maximum value
        peak_idx = np.argmax(specific_heats_smooth)
    else:
        # Get the highest peak
        highest_peak_idx = np.argmax(properties['peak_heights'])
        peak_idx = peaks[highest_peak_idx]
    
    # Get the critical temperature and peak height
    critical_temp = temps[peak_idx]
    peak_height = specific_heats_smooth[peak_idx]
    
    return critical_temp, peak_height


def analyze_critical_scaling(temps, specific_heats, critical_temp, temp_range=0.2):
    """
    Analyze the critical scaling behavior near the critical point.
    
    Parameters:
    -----------
    temps : array_like
        Array of temperatures.
    specific_heats : array_like
        Array of specific heat values.
    critical_temp : float
        Critical temperature.
    temp_range : float, optional
        Relative range around the critical point to analyze.
    
    Returns:
    --------
    dict
        Dictionary containing scaling exponents and goodness of fit.
    """
    # Determine the range of temperatures to analyze
    min_temp = critical_temp * (1 - temp_range)
    max_temp = critical_temp * (1 + temp_range)
    
    # Find indices within the range
    indices = np.where((temps >= min_temp) & (temps <= max_temp))
    temps_range = temps[indices]
    specific_heats_range = specific_heats[indices]
    
    # Define the relative temperature (t = |T - Tc| / Tc)
    t_below = (critical_temp - temps_range[temps_range < critical_temp]) / critical_temp
    t_above = (temps_range[temps_range > critical_temp] - critical_temp) / critical_temp
    
    C_below = specific_heats_range[temps_range < critical_temp]
    C_above = specific_heats_range[temps_range > critical_temp]
    
    # Fit power law (C ~ t^(-alpha)) for both sides
    results = {}
    
    if len(t_below) > 2:
        # Log-linear fit for below Tc
        slope_below, intercept_below, r_value_below, p_value_below, std_err_below = stats.linregress(
            np.log(t_below), np.log(C_below)
        )
        alpha_below = -slope_below
        results['alpha_below'] = alpha_below
        results['r_squared_below'] = r_value_below ** 2
        results['std_err_below'] = std_err_below
    
    if len(t_above) > 2:
        # Log-linear fit for above Tc
        slope_above, intercept_above, r_value_above, p_value_above, std_err_above = stats.linregress(
            np.log(t_above), np.log(C_above)
        )
        alpha_above = -slope_above
        results['alpha_above'] = alpha_above
        results['r_squared_above'] = r_value_above ** 2
        results['std_err_above'] = std_err_above
    
    return results


def batch_analyze_matrices(matrices, temp_range=(0.1, 2.0), n_temps=50, k_B=1.0, use_gpu=False):
    """
    Analyze multiple weight matrices and collect the results.
    
    Parameters:
    -----------
    matrices : dict
        Dictionary of weight matrices to analyze.
    temp_range : tuple, optional
        Range of temperatures to evaluate.
    n_temps : int, optional
        Number of temperature points.
    k_B : float, optional
        Boltzmann constant (default: 1.0 for normalized units).
    use_gpu : bool, optional
        Whether to use GPU acceleration (requires CuPy).
    
    Returns:
    --------
    dict
        Dictionary containing all analysis results.
    """
    temps = np.linspace(temp_range[0], temp_range[1], n_temps)
    results = {}
    
    for name, matrix in matrices.items():
        # Skip if not a proper matrix
        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
            continue
        
        # Compute thermal properties
        thermal_props = compute_thermal_properties(matrix, temps, k_B, use_gpu)
        
        # Find critical point
        critical_temp, peak_height = find_critical_point(
            thermal_props['temperatures'], 
            thermal_props['specific_heat']
        )
        
        # Analyze critical scaling
        scaling_results = analyze_critical_scaling(
            thermal_props['temperatures'],
            thermal_props['specific_heat'],
            critical_temp
        )
        
        # Store results
        results[name] = {
            'thermal_properties': thermal_props,
            'critical_temperature': critical_temp,
            'peak_height': peak_height,
            'matrix_shape': matrix.shape,
            'scaling_results': scaling_results
        }
    
    return results


def eigenvalue_spectrum(weight_matrix):
    """
    Calculate the eigenvalue spectrum of a weight matrix.
    
    Parameters:
    -----------
    weight_matrix : array_like
        Weight matrix to analyze.
    
    Returns:
    --------
    array_like
        Array of eigenvalues.
    """
    # Handle non-square matrices by using the SVD
    if weight_matrix.shape[0] != weight_matrix.shape[1]:
        from scipy.linalg import svd
        # Singular values are the square roots of the eigenvalues of W*W^T
        singular_values = svd(weight_matrix, compute_uv=False)
        # Return squared singular values
        return singular_values**2
    else:
        # For square matrices, use direct eigenvalue decomposition
        from scipy.linalg import eigvalsh
        return eigvalsh(weight_matrix)