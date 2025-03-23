"""
Eigenvalue spectrum analysis for neural network weights.
"""

import numpy as np
from scipy.linalg import eigh, svd
from scipy.stats import gaussian_kde
import warnings


def eigenvalue_distribution(weight_matrix, bins=50):
    """
    Calculate the eigenvalue distribution of a weight matrix.
    
    Parameters:
    -----------
    weight_matrix : array_like
        Weight matrix to analyze.
    bins : int, optional
        Number of bins for histogram.
    
    Returns:
    --------
    tuple
        (eigenvalues, histogram_counts, bin_edges)
    """
    # Calculate eigenvalues differently based on matrix shape
    if weight_matrix.shape[0] == weight_matrix.shape[1]:
        # For square matrices, compute eigenvalues directly
        eigenvalues = eigh(weight_matrix, eigvals_only=True)
    else:
        # For non-square matrices, compute eigenvalues of W*W^T
        # These are the squares of the singular values
        gram_matrix = weight_matrix @ weight_matrix.T
        eigenvalues = eigh(gram_matrix, eigvals_only=True)
        # Take square root to get singular values
        eigenvalues = np.sqrt(np.abs(eigenvalues))
    
    # Calculate histogram
    hist, bin_edges = np.histogram(eigenvalues, bins=bins, density=True)
    
    return eigenvalues, hist, bin_edges


def spectral_density(eigenvalues, energy_range=None, bandwidth=0.1, n_points=1000):
    """
    Compute spectral density using KDE smoothing.
    
    Parameters:
    -----------
    eigenvalues : array_like
        Array of eigenvalues.
    energy_range : tuple, optional
        (min_energy, max_energy) range for evaluation. If None, determined from data.
    bandwidth : float, optional
        Bandwidth parameter for KDE.
    n_points : int, optional
        Number of points to evaluate the density.
    
    Returns:
    --------
    tuple
        (energy points, density values)
    """
    # Determine energy range if not provided
    if energy_range is None:
        min_energy = np.min(eigenvalues) - 2 * bandwidth * np.std(eigenvalues)
        max_energy = np.max(eigenvalues) + 2 * bandwidth * np.std(eigenvalues)
        energy_range = (min_energy, max_energy)
    
    # Create evaluation points
    energy_points = np.linspace(energy_range[0], energy_range[1], n_points)
    
    # Apply KDE smoothing
    kde = gaussian_kde(eigenvalues, bw_method=bandwidth)
    density = kde(energy_points)
    
    return energy_points, density


def marchenko_pastur_distance(eigenvalues, aspect_ratio):
    """
    Compute distance from empirical eigenvalue distribution to Marchenko-Pastur law.
    
    Parameters:
    -----------
    eigenvalues : array_like
        Array of eigenvalues.
    aspect_ratio : float
        Aspect ratio (m/n) of the matrix.
    
    Returns:
    --------
    float
        Distance metric from M-P law.
    """
    # Ensure eigenvalues are normalized
    n_eigenvalues = len(eigenvalues)
    normalized_eigenvalues = eigenvalues / np.mean(eigenvalues) * aspect_ratio
    
    # Theoretical bounds of the M-P distribution
    lambda_min = (1 - np.sqrt(1/aspect_ratio))**2
    lambda_max = (1 + np.sqrt(1/aspect_ratio))**2
    
    # Calculate empirical CDF
    sorted_eigenvalues = np.sort(normalized_eigenvalues)
    empirical_cdf = np.arange(1, n_eigenvalues + 1) / n_eigenvalues
    
    # Calculate theoretical CDF (simplified M-P law)
    # This is an approximation valid for aspect_ratio > 1
    def mp_cdf(x):
        if x <= lambda_min:
            return 0.0
        elif x >= lambda_max:
            return 1.0
        else:
            # Numerical integration of M-P density
            from scipy.integrate import quad
            
            # M-P density function
            def mp_density(t):
                return np.sqrt(np.maximum(0, (lambda_max - t) * (t - lambda_min))) / (2 * np.pi * t * aspect_ratio)
            
            result, _ = quad(mp_density, lambda_min, x)
            # Add the mass at zero if aspect_ratio < 1
            if aspect_ratio < 1:
                result += max(0, 1 - aspect_ratio)
            
            return result
    
    # Calculate theoretical CDF at empirical points
    theoretical_cdf = np.array([mp_cdf(x) for x in sorted_eigenvalues])
    
    # Calculate Kolmogorov-Smirnov distance
    ks_distance = np.max(np.abs(empirical_cdf - theoretical_cdf))
    
    return ks_distance


def correlate_eigenvalues_criticality(models_data):
    """
    Analyze correlation between eigenvalue statistics and critical temperatures.
    
    Parameters:
    -----------
    models_data : dict
        Dictionary containing model data with eigenvalues and critical temperatures.
    
    Returns:
    --------
    dict
        Dictionary with correlation analysis results.
    """
    # Extract data for correlation analysis
    model_names = []
    critical_temps = []
    mean_eigenvalues = []
    max_eigenvalues = []
    eigenvalue_gaps = []
    participation_ratios = []
    
    for model_name, data in models_data.items():
        if 'eigenvalues' not in data or 'critical_temperature' not in data:
            continue
        
        model_names.append(model_name)
        critical_temps.append(data['critical_temperature'])
        
        eigenvalues = data['eigenvalues']
        mean_eigenvalues.append(np.mean(eigenvalues))
        max_eigenvalues.append(np.max(eigenvalues))
        
        # Calculate eigenvalue gap (largest - second largest)
        sorted_eigenvalues = np.sort(eigenvalues)
        if len(sorted_eigenvalues) > 1:
            gap = sorted_eigenvalues[-1] - sorted_eigenvalues[-2]
            eigenvalue_gaps.append(gap)
        else:
            eigenvalue_gaps.append(0)
        
        # Calculate participation ratio
        pr = 1.0 / np.sum((eigenvalues / np.sum(eigenvalues))**2)
        participation_ratios.append(pr)
    
    # Calculate correlations
    from scipy.stats import pearsonr, spearmanr
    
    results = {}
    
    if len(critical_temps) > 1:
        metrics = {
            'mean_eigenvalue': mean_eigenvalues,
            'max_eigenvalue': max_eigenvalues,
            'eigenvalue_gap': eigenvalue_gaps,
            'participation_ratio': participation_ratios
        }
        
        correlations = {}
        for metric_name, metric_values in metrics.items():
            try:
                pearson_corr, pearson_p = pearsonr(metric_values, critical_temps)
                spearman_corr, spearman_p = spearmanr(metric_values, critical_temps)
                
                correlations[metric_name] = {
                    'pearson_correlation': pearson_corr,
                    'pearson_p_value': pearson_p,
                    'spearman_correlation': spearman_corr,
                    'spearman_p_value': spearman_p
                }
            except:
                # Skip if correlation fails (e.g., constant values)
                pass
        
        results['correlations'] = correlations
    
    # Add raw data for plotting
    results['data'] = {
        'model_names': model_names,
        'critical_temps': critical_temps,
        'mean_eigenvalues': mean_eigenvalues,
        'max_eigenvalues': max_eigenvalues,
        'eigenvalue_gaps': eigenvalue_gaps,
        'participation_ratios': participation_ratios
    }
    
    return results


def layer_wise_eigenvalue_analysis(model_weights):
    """
    Analyze eigenvalue distributions across different layers.
    
    Parameters:
    -----------
    model_weights : dict
        Dictionary containing model weights by layer.
    
    Returns:
    --------
    dict
        Dictionary with layer-wise eigenvalue analysis results.
    """
    layer_results = {}
    
    for layer_name, layer_weights in model_weights.items():
        # Skip non-matrix data
        if not isinstance(layer_weights, dict):
            continue
        
        layer_results[layer_name] = {}
        
        for matrix_name, matrix in layer_weights.items():
            if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
                continue
            
            # Calculate eigenvalues
            if matrix.shape[0] == matrix.shape[1]:
                eigenvalues = eigh(matrix, eigvals_only=True)
            else:
                # Non-square matrix: compute singular values
                s = svd(matrix, compute_uv=False)
                eigenvalues = s**2  # Eigenvalues of W*W^T
            
            # Calculate various metrics
            mean_eig = np.mean(eigenvalues)
            max_eig = np.max(eigenvalues)
            min_eig = np.min(eigenvalues)
            
            # Calculate eigenvalue gap
            sorted_eigs = np.sort(eigenvalues)
            if len(sorted_eigs) > 1:
                gap = sorted_eigs[-1] - sorted_eigs[-2]
            else:
                gap = 0
            
            # Participation ratio (measure of eigenvalue localization)
            normalized_eigs = eigenvalues / np.sum(eigenvalues)
            pr = 1.0 / np.sum(normalized_eigs**2)
            
            # Store results
            layer_results[layer_name][matrix_name] = {
                'eigenvalues': eigenvalues,
                'mean': mean_eig,
                'max': max_eig,
                'min': min_eig,
                'gap': gap,
                'participation_ratio': pr,
                'shape': matrix.shape
            }
    
    return layer_results


def random_matrix_baseline(shape, n_samples=10):
    """
    Generate baseline statistics from random matrices for comparison.
    
    Parameters:
    -----------
    shape : tuple
        Shape of the matrices to generate.
    n_samples : int, optional
        Number of random matrices to generate.
    
    Returns:
    --------
    dict
        Dictionary with baseline statistics.
    """
    from scipy.stats import norm
    
    # Generate random matrices with Gaussian entries
    random_matrices = [np.random.normal(0, 1, shape) for _ in range(n_samples)]
    
    # Analyze eigenvalues
    eigenvalues_list = []
    metrics = {
        'mean': [],
        'max': [],
        'min': [],
        'gap': [],
        'participation_ratio': []
    }
    
    for matrix in random_matrices:
        # Calculate eigenvalues
        if shape[0] == shape[1]:
            eigenvalues = eigh(matrix, eigvals_only=True)
        else:
            # For non-square matrices, use SVD
            s = svd(matrix, compute_uv=False)
            eigenvalues = s**2
        
        eigenvalues_list.append(eigenvalues)
        
        # Calculate metrics
        metrics['mean'].append(np.mean(eigenvalues))
        metrics['max'].append(np.max(eigenvalues))
        metrics['min'].append(np.min(eigenvalues))
        
        # Gap
        sorted_eigs = np.sort(eigenvalues)
        if len(sorted_eigs) > 1:
            gap = sorted_eigs[-1] - sorted_eigs[-2]
        else:
            gap = 0
        metrics['gap'].append(gap)
        
        # Participation ratio
        normalized_eigs = eigenvalues / np.sum(eigenvalues)
        pr = 1.0 / np.sum(normalized_eigs**2)
        metrics['participation_ratio'].append(pr)
    
    # Calculate statistics for each metric
    baseline_stats = {}
    for metric_name, metric_values in metrics.items():
        baseline_stats[metric_name] = {
            'mean': np.mean(metric_values),
            'std': np.std(metric_values),
            'min': np.min(metric_values),
            'max': np.max(metric_values)
        }
    
    # Theoretical Marchenko-Pastur bounds
    aspect_ratio = shape[0] / shape[1]
    if aspect_ratio > 1:
        aspect_ratio = 1.0 / aspect_ratio
    
    lambda_min = (1 - np.sqrt(1/aspect_ratio))**2
    lambda_max = (1 + np.sqrt(1/aspect_ratio))**2
    
    baseline_stats['marchenko_pastur'] = {
        'lambda_min': lambda_min,
        'lambda_max': lambda_max,
        'aspect_ratio': aspect_ratio
    }
    
    return baseline_stats