"""
Utility functions for thermodynamic scaling analysis.
"""

import numpy as np
import os
import pickle
import warnings
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def save_results(results, filepath, overwrite=False):
    """
    Save results to a pickle file.
    
    Parameters:
    -----------
    results : object
        Results to save.
    filepath : str
        Path to save the results.
    overwrite : bool, optional
        Whether to overwrite existing file.
    """
    # Check if directory exists, create if not
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Check if file exists
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(f"File {filepath} already exists. Set overwrite=True to overwrite.")
    
    # Save results
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)


def load_results(filepath):
    """
    Load results from a pickle file.
    
    Parameters:
    -----------
    filepath : str
        Path to the results file.
    
    Returns:
    --------
    object
        Loaded results.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found.")
    
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    
    return results


def batch_process(data_list, process_func, description="Processing", show_progress=True):
    """
    Process a list of data items with a given function.
    
    Parameters:
    -----------
    data_list : list
        List of data items to process.
    process_func : callable
        Function to apply to each item.
    description : str, optional
        Description for the progress bar.
    show_progress : bool, optional
        Whether to show a progress bar.
    
    Returns:
    --------
    list
        List of processed results.
    """
    if show_progress:
        iterator = tqdm(data_list, desc=description)
    else:
        iterator = data_list
    
    results = []
    for item in iterator:
        try:
            result = process_func(item)
            results.append(result)
        except Exception as e:
            warnings.warn(f"Error processing item: {e}")
    
    return results


def truncate_matrix(matrix, max_size=1000):
    """
    Truncate a large matrix for faster processing.
    
    Parameters:
    -----------
    matrix : array_like
        Matrix to truncate.
    max_size : int, optional
        Maximum size (per dimension) to keep.
    
    Returns:
    --------
    array_like
        Truncated matrix.
    """
    if matrix.shape[0] <= max_size and matrix.shape[1] <= max_size:
        return matrix
    
    # Truncate to max_size
    rows = min(matrix.shape[0], max_size)
    cols = min(matrix.shape[1], max_size)
    
    return matrix[:rows, :cols]


def plot_matrix_heatmap(matrix, title=None, cmap='viridis', figsize=(10, 8), save_path=None):
    """
    Plot a matrix as a heatmap.
    
    Parameters:
    -----------
    matrix : array_like
        Matrix to plot.
    title : str, optional
        Plot title.
    cmap : str, optional
        Colormap name.
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(matrix, cmap=cmap)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    
    # Set title
    if title:
        ax.set_title(title)
    
    # Set axis labels
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_distributions(dist1, dist2, labels=None, figsize=(10, 6), bins=30, save_path=None):
    """
    Compare two distributions with histograms.
    
    Parameters:
    -----------
    dist1 : array_like
        First distribution.
    dist2 : array_like
        Second distribution.
    labels : tuple, optional
        Labels for the distributions (label1, label2).
    figsize : tuple, optional
        Figure size.
    bins : int, optional
        Number of bins for histogram.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure.
    """
    if labels is None:
        labels = ('Distribution 1', 'Distribution 2')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histograms
    ax.hist(dist1, bins=bins, alpha=0.5, label=labels[0])
    ax.hist(dist2, bins=bins, alpha=0.5, label=labels[1])
    
    # Add legend
    ax.legend()
    
    # Set labels
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution Comparison')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def calculate_statistics(data):
    """
    Calculate common statistics for a dataset.
    
    Parameters:
    -----------
    data : array_like
        Data to analyze.
    
    Returns:
    --------
    dict
        Dictionary with statistics.
    """
    data = np.asarray(data)
    
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'range': np.max(data) - np.min(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }


def calculate_correlation_matrix(data_dict, method='pearson'):
    """
    Calculate correlation matrix between variables.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with {variable_name: values}.
    method : str, optional
        Correlation method ('pearson', 'spearman', or 'kendall').
    
    Returns:
    --------
    tuple
        (correlation_matrix, variable_names)
    """
    # Extract variable names and values
    variable_names = list(data_dict.keys())
    data_matrix = np.column_stack([data_dict[var] for var in variable_names])
    
    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = np.corrcoef(data_matrix.T)
    elif method == 'spearman':
        corr_matrix = stats.spearmanr(data_matrix, axis=0)[0]
    elif method == 'kendall':
        # Kendall's tau must be calculated pairwise
        n_vars = len(variable_names)
        corr_matrix = np.ones((n_vars, n_vars))
        
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                tau, _ = stats.kendalltau(data_dict[variable_names[i]], data_dict[variable_names[j]])
                corr_matrix[i, j] = tau
                corr_matrix[j, i] = tau
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    return corr_matrix, variable_names


def plot_correlation_matrix(corr_matrix, variable_names, figsize=(10, 8), cmap='coolwarm', 
                          save_path=None):
    """
    Plot a correlation matrix as a heatmap.
    
    Parameters:
    -----------
    corr_matrix : array_like
        Correlation matrix.
    variable_names : list
        List of variable names.
    figsize : tuple, optional
        Figure size.
    cmap : str, optional
        Colormap name.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Correlation')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(variable_names)))
    ax.set_yticks(np.arange(len(variable_names)))
    ax.set_xticklabels(variable_names)
    ax.set_yticklabels(variable_names)
    
    # Rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add values to cells
    for i in range(len(variable_names)):
        for j in range(len(variable_names)):
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center',
                   color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
    
    # Set title
    ax.set_title('Correlation Matrix')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_dataframe_from_results(results_data):
    """
    Create a pandas DataFrame from nested results data.
    
    Parameters:
    -----------
    results_data : list
        List of result dictionaries.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with flattened results.
    """
    import pandas as pd
    
    # Create a list to store flattened results
    flattened_results = []
    
    for result in results_data:
        # Start with a copy of the result
        flattened = {}
        
        # Process each key-value pair
        for key, value in result.items():
            # Skip large data structures
            if key in ['thermal_properties', 'eigenvalues']:
                continue
            
            # Include scalar values directly
            if np.isscalar(value) or isinstance(value, (str, bool)):
                flattened[key] = value
            # For small lists/arrays, convert to string
            elif isinstance(value, (list, np.ndarray)) and len(value) < 10:
                flattened[key] = str(value)
        
        flattened_results.append(flattened)
    
    # Create DataFrame
    return pd.DataFrame(flattened_results)


def robust_normalization(data, lower_quantile=0.01, upper_quantile=0.99):
    """
    Normalize data robustly using quantiles to handle outliers.
    
    Parameters:
    -----------
    data : array_like
        Data to normalize.
    lower_quantile : float, optional
        Lower quantile for normalization.
    upper_quantile : float, optional
        Upper quantile for normalization.
    
    Returns:
    --------
    array_like
        Normalized data.
    """
    data = np.asarray(data)
    
    # Calculate quantiles
    q_low = np.quantile(data, lower_quantile)
    q_high = np.quantile(data, upper_quantile)
    
    # Clip and normalize
    clipped_data = np.clip(data, q_low, q_high)
    normalized_data = (clipped_data - q_low) / (q_high - q_low)
    
    return normalized_data


def moving_average(data, window_size):
    """
    Calculate moving average of data.
    
    Parameters:
    -----------
    data : array_like
        Data to smooth.
    window_size : int
        Size of the moving average window.
    
    Returns:
    --------
    array_like
        Smoothed data.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def estimate_peak_sharpness(x, y):
    """
    Estimate the sharpness of a peak in a curve.
    
    Parameters:
    -----------
    x : array_like
        x-coordinates.
    y : array_like
        y-coordinates.
    
    Returns:
    --------
    float
        Estimated peak sharpness.
    """
    from scipy.signal import find_peaks
    
    # Find peaks
    peaks, properties = find_peaks(y)
    
    if len(peaks) == 0:
        # If no clear peak, use the maximum value
        peak_idx = np.argmax(y)
    else:
        # Get the highest peak
        highest_peak_idx = np.argmax([y[p] for p in peaks])
        peak_idx = peaks[highest_peak_idx]
    
    # Calculate peak sharpness as the second derivative at the peak
    if peak_idx > 0 and peak_idx < len(y) - 1:
        # Approximate second derivative using central difference
        dx = x[peak_idx + 1] - x[peak_idx - 1]
        d2y = (y[peak_idx + 1] - 2 * y[peak_idx] + y[peak_idx - 1]) / (dx/2)**2
        return -d2y  # Negative because a sharp peak has negative second derivative
    else:
        return 0.0  # Unable to calculate for edge cases