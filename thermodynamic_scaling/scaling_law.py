"""
Functions for fitting and analyzing scaling laws.
"""

import numpy as np
from scipy.optimize import curve_fit
import warnings
try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    warnings.warn("PyMC not available. Bayesian estimation will not be used.")


def fit_power_law(x, y, log_fit=True):
    """
    Fit power law relationship y = a * x^b using multiple methods.
    
    Parameters:
    -----------
    x : array_like
        Independent variable.
    y : array_like
        Dependent variable.
    log_fit : bool, optional
        Whether to fit in log-log space.
    
    Returns:
    --------
    dict
        Dictionary with fitting results.
    """
    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Remove any points with non-positive values
    mask = (x > 0) & (y > 0)
    x_filtered = x[mask]
    y_filtered = y[mask]
    
    if len(x_filtered) < 2:
        warnings.warn("Not enough positive data points for power law fitting.")
        return {
            'success': False,
            'message': "Not enough positive data points for power law fitting."
        }
    
    results = {'success': True}
    
    # Method 1: Linear fit in log-log space
    log_x = np.log(x_filtered)
    log_y = np.log(y_filtered)
    
    # Linear regression
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
    
    # Convert back to power law parameters
    a_log = np.exp(intercept)
    b_log = slope
    
    results['log_linear'] = {
        'a': a_log,
        'b': b_log,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }
    
    # Method 2: Direct fit in original space
    if not log_fit:
        try:
            def power_law(x, a, b):
                return a * x**b
            
            popt, pcov = curve_fit(power_law, x_filtered, y_filtered, p0=[a_log, b_log], maxfev=10000)
            
            a_direct = popt[0]
            b_direct = popt[1]
            
            # Calculate R-squared
            y_fit = power_law(x_filtered, a_direct, b_direct)
            ss_tot = np.sum((y_filtered - np.mean(y_filtered))**2)
            ss_res = np.sum((y_filtered - y_fit)**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            results['direct'] = {
                'a': a_direct,
                'b': b_direct,
                'r_squared': r_squared,
                'std_err': np.sqrt(np.diag(pcov))
            }
        except:
            warnings.warn("Direct curve fitting failed. Using log-linear fit only.")
    
    # Choose best fit (prefer log-linear for power laws)
    best_method = 'log_linear'
    results['best'] = results[best_method]
    
    # Power law equation as string
    results['equation'] = f"y = {results['best']['a']:.4e} * x^{results['best']['b']:.4f}"
    
    return results


def bayesian_power_law_fit(x, y, samples=1000):
    """
    Bayesian estimation of power law parameters with uncertainty.
    
    Parameters:
    -----------
    x : array_like
        Independent variable.
    y : array_like
        Dependent variable.
    samples : int, optional
        Number of posterior samples.
    
    Returns:
    --------
    dict
        Dictionary with Bayesian inference results.
    """
    if not HAS_PYMC:
        warnings.warn("PyMC not available. Using frequentist fit instead.")
        return fit_power_law(x, y)
    
    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Remove any points with non-positive values for log transform
    mask = (x > 0) & (y > 0)
    x_filtered = x[mask]
    y_filtered = y[mask]
    
    if len(x_filtered) < 3:
        warnings.warn("Not enough positive data points for Bayesian power law fitting.")
        return fit_power_law(x, y)
    
    # Log transform for power law
    log_x = np.log(x_filtered)
    log_y = np.log(y_filtered)
    
    # Bayesian linear regression on log-transformed data
    with pm.Model() as model:
        # Priors
        intercept = pm.Normal("intercept", mu=0, sigma=10)
        slope = pm.Normal("slope", mu=0, sigma=5)
        sigma = pm.HalfNormal("sigma", sigma=1)
        
        # Linear model in log space
        mu = intercept + slope * log_x
        
        # Likelihood
        likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=log_y)
        
        # Sample posterior
        trace = pm.sample(samples, tune=1000, return_inferencedata=True)
    
    # Extract posterior samples
    posterior_samples = trace.posterior
    
    # Convert to power law parameters
    a_samples = np.exp(posterior_samples["intercept"].values.flatten())
    b_samples = posterior_samples["slope"].values.flatten()
    
    # Compute parameter estimates and credible intervals
    a_mean = np.mean(a_samples)
    a_std = np.std(a_samples)
    a_hdi = np.percentile(a_samples, [2.5, 97.5])
    
    b_mean = np.mean(b_samples)
    b_std = np.std(b_samples)
    b_hdi = np.percentile(b_samples, [2.5, 97.5])
    
    # Compute R-squared for the mean model
    def power_law(x, a, b):
        return a * x**b
    
    y_pred = power_law(x_filtered, a_mean, b_mean)
    ss_tot = np.sum((y_filtered - np.mean(y_filtered))**2)
    ss_res = np.sum((y_filtered - y_pred)**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Compile results
    results = {
        'success': True,
        'bayesian': {
            'a_mean': a_mean,
            'a_std': a_std,
            'a_hdi': a_hdi,
            'b_mean': b_mean,
            'b_std': b_std,
            'b_hdi': b_hdi,
            'r_squared': r_squared
        },
        'best': {
            'a': a_mean,
            'b': b_mean,
            'r_squared': r_squared
        },
        'equation': f"y = {a_mean:.4e} * x^{b_mean:.4f}",
        'trace': trace
    }
    
    return results


def quantum_criticality_test(exponent, confidence_interval):
    """
    Test if exponent falls within quantum criticality range.
    
    Parameters:
    -----------
    exponent : float
        Estimated critical exponent.
    confidence_interval : tuple
        (lower, upper) bounds of the confidence interval.
    
    Returns:
    --------
    dict
        Dictionary with test results.
    """
    # Theoretical range for quantum criticality
    quantum_min = 0.3
    quantum_max = 0.7
    
    # Check if exponent is in quantum range
    is_quantum = quantum_min <= exponent <= quantum_max
    
    # Check if confidence interval overlaps with quantum range
    lower, upper = confidence_interval
    overlap = (lower <= quantum_max and upper >= quantum_min)
    
    # Calculate overlap proportion with quantum range
    if overlap:
        overlap_min = max(lower, quantum_min)
        overlap_max = min(upper, quantum_max)
        overlap_proportion = (overlap_max - overlap_min) / (upper - lower)
    else:
        overlap_proportion = 0.0
    
    # Calculate distance to quantum range (if outside)
    if exponent < quantum_min:
        distance = quantum_min - exponent
    elif exponent > quantum_max:
        distance = exponent - quantum_max
    else:
        distance = 0.0
    
    return {
        'exponent': exponent,
        'confidence_interval': confidence_interval,
        'is_in_quantum_range': is_quantum,
        'overlaps_quantum_range': overlap,
        'overlap_proportion': overlap_proportion,
        'distance_to_quantum_range': distance,
        'quantum_range': (quantum_min, quantum_max)
    }


def universal_scaling_function(data, critical_exponents):
    """
    Test for data collapse using universal scaling functions.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing data for different system sizes.
        Expected format: {
            'L1': {'x': array, 'y': array},
            'L2': {'x': array, 'y': array},
            ...
        }
        where L1, L2, ... are system size labels.
    critical_exponents : dict
        Dictionary with critical exponents.
        Expected keys: 'nu', 'beta', 'Tc'.
    
    Returns:
    --------
    dict
        Dictionary with data collapse results.
    """
    # Extract critical exponents
    nu = critical_exponents['nu']
    beta = critical_exponents.get('beta', 1.0)
    Tc = critical_exponents['Tc']
    
    # Extract system sizes
    system_sizes = []
    for label in data:
        try:
            # Try to extract numeric size from label
            size = float(''.join(filter(lambda c: c.isdigit() or c == '.', label)))
            system_sizes.append((label, size))
        except:
            warnings.warn(f"Could not extract numeric size from label {label}")
    
    # Sort by system size
    system_sizes.sort(key=lambda x: x[1])
    
    # Apply scaling transformations
    scaled_data = {}
    
    for label, L in system_sizes:
        if label not in data:
            continue
        
        x_raw = np.asarray(data[label]['x'])
        y_raw = np.asarray(data[label]['y'])
        
        # Apply scaling transformations
        # For temperature: t = (T - Tc) / Tc
        t = (x_raw - Tc) / Tc
        
        # Scaled x: L^(1/nu) * t
        x_scaled = L**(1/nu) * t
        
        # Scaled y: L^(-beta/nu) * y
        y_scaled = L**(-beta/nu) * y_raw
        
        scaled_data[label] = {
            'x_raw': x_raw,
            'y_raw': y_raw,
            'x_scaled': x_scaled,
            'y_scaled': y_scaled,
            'L': L
        }
    
    # Quantify quality of collapse
    # Use coefficient of variation of scaled curves
    if len(scaled_data) < 2:
        collapse_quality = 0.0
    else:
        # Create a common x-grid for interpolation
        x_min = np.inf
        x_max = -np.inf
        
        for label in scaled_data:
            x_min = min(x_min, np.min(scaled_data[label]['x_scaled']))
            x_max = max(x_max, np.max(scaled_data[label]['x_scaled']))
        
        x_grid = np.linspace(x_min, x_max, 100)
        
        # Interpolate all curves to the common grid
        from scipy.interpolate import interp1d
        
        y_interpolated = []
        for label in scaled_data:
            x_scaled = scaled_data[label]['x_scaled']
            y_scaled = scaled_data[label]['y_scaled']
            
            # Create interpolation function
            try:
                interp_func = interp1d(x_scaled, y_scaled, bounds_error=False, fill_value=np.nan)
                y_interp = interp_func(x_grid)
                y_interpolated.append(y_interp)
            except:
                warnings.warn(f"Interpolation failed for {label}")
        
        # Convert to array
        y_interpolated = np.array(y_interpolated)
        
        # Calculate coefficient of variation
        cv = []
        for i in range(len(x_grid)):
            y_values = y_interpolated[:, i]
            # Remove NaN values
            y_values = y_values[~np.isnan(y_values)]
            
            if len(y_values) >= 2:
                mean = np.mean(y_values)
                std = np.std(y_values)
                if mean != 0:
                    cv.append(std / abs(mean))
        
        # Average CV as a measure of collapse quality
        if cv:
            collapse_quality = 1.0 - min(1.0, np.mean(cv))
        else:
            collapse_quality = 0.0
    
    return {
        'scaled_data': scaled_data,
        'collapse_quality': collapse_quality,
        'critical_exponents': critical_exponents
    }


def multi_variable_scaling_law(x_vars, y, model='power_law'):
    """
    Fit a multivariable scaling law relating y to multiple x variables.
    
    Parameters:
    -----------
    x_vars : dict
        Dictionary of independent variables {var_name: array}.
    y : array_like
        Dependent variable.
    model : str, optional
        Model type ('power_law' or 'linear').
    
    Returns:
    --------
    dict
        Dictionary with fitting results.
    """
    # Ensure all inputs are numpy arrays of the same length
    y = np.asarray(y)
    
    var_names = list(x_vars.keys())
    x_data = {}
    
    for var in var_names:
        x_data[var] = np.asarray(x_vars[var])
        if len(x_data[var]) != len(y):
            raise ValueError(f"Length mismatch: {var} has length {len(x_data[var])}, but y has length {len(y)}")
    
    # For power law model, transform to log space
    if model == 'power_law':
        # Check for positive values
        valid_indices = np.ones(len(y), dtype=bool)
        log_x_data = {}
        
        # Filter out non-positive values
        for var in var_names:
            valid_var = x_data[var] > 0
            valid_indices = valid_indices & valid_var
        
        valid_indices = valid_indices & (y > 0)
        
        if np.sum(valid_indices) < len(y) * 0.5:
            warnings.warn(f"Less than half of the data points are valid for log transformation. Using {np.sum(valid_indices)} out of {len(y)}.")
        
        if np.sum(valid_indices) < 3:
            raise ValueError("Not enough valid data points for fitting.")
        
        log_y = np.log(y[valid_indices])
        
        for var in var_names:
            log_x_data[var] = np.log(x_data[var][valid_indices])
        
        # Prepare design matrix for linear regression in log space
        X = np.column_stack([log_x_data[var] for var in var_names])
        X = np.column_stack([np.ones(X.shape[0]), X])  # Add intercept column
        
        # Fit linear model in log space
        from sklearn.linear_model import LinearRegression
        
        model_fit = LinearRegression().fit(X, log_y)
        
        # Extract coefficients
        intercept = model_fit.intercept_
        coefficients = model_fit.coef_[1:]  # Skip intercept
        
        # Transform back to power law
        a = np.exp(intercept)
        exponents = coefficients
        
        # Calculate R-squared
        r_squared = model_fit.score(X, log_y)
        
        # Create power law formula
        formula_parts = [f"{var}^{exponents[i]:.4f}" for i, var in enumerate(var_names)]
        formula = f"y = {a:.4e} * " + " * ".join(formula_parts)
        
        results = {
            'model_type': 'power_law',
            'intercept': a,
            'exponents': {var: exponents[i] for i, var in enumerate(var_names)},
            'r_squared': r_squared,
            'formula': formula
        }
    
    else:  # Linear model
        # Prepare design matrix
        X = np.column_stack([x_data[var] for var in var_names])
        X = np.column_stack([np.ones(X.shape[0]), X])  # Add intercept column
        
        # Fit linear model
        from sklearn.linear_model import LinearRegression
        
        model_fit = LinearRegression().fit(X, y)
        
        # Extract coefficients
        intercept = model_fit.intercept_
        coefficients = model_fit.coef_[1:]  # Skip intercept
        
        # Calculate R-squared
        r_squared = model_fit.score(X, y)
        
        # Create linear formula
        formula_parts = [f"{coefficients[i]:.4e} * {var}" for i, var in enumerate(var_names)]
        formula = f"y = {intercept:.4e} + " + " + ".join(formula_parts)
        
        results = {
            'model_type': 'linear',
            'intercept': intercept,
            'coefficients': {var: coefficients[i] for i, var in enumerate(var_names)},
            'r_squared': r_squared,
            'formula': formula
        }
    
    return results


def compute_scaling_laws(results_df):
    """
    Extract scaling law relationships from results.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing model results.
    
    Returns:
    --------
    dict
        Dictionary with scaling law results.
    """
    import pandas as pd
    
    # Check if input is a DataFrame
    if not isinstance(results_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    # Extract relevant columns for scaling laws
    scaling_results = {}
    
    # 1. Critical temperature vs. model size
    if 'model_size' in results_df.columns and 'critical_temperature' in results_df.columns:
        x = results_df['model_size'].values
        y = results_df['critical_temperature'].values
        
        # Fit power law
        power_law_fit = fit_power_law(x, y)
        
        scaling_results['model_size_vs_critical_temp'] = {
            'x': x,
            'y': y,
            'fit': power_law_fit
        }
    
    # 2. Specific heat peak vs. model size
    if 'model_size' in results_df.columns and 'specific_heat_peak' in results_df.columns:
        x = results_df['model_size'].values
        y = results_df['specific_heat_peak'].values
        
        # Fit power law
        power_law_fit = fit_power_law(x, y)
        
        scaling_results['model_size_vs_specific_heat'] = {
            'x': x,
            'y': y,
            'fit': power_law_fit
        }
    
    # 3. Critical exponent vs. model architecture
    if 'model_architecture' in results_df.columns and 'critical_exponent' in results_df.columns:
        # Group by architecture and compute mean exponents
        arch_groups = results_df.groupby('model_architecture')['critical_exponent'].mean()
        
        scaling_results['architecture_vs_exponent'] = {
            'architectures': arch_groups.index.tolist(),
            'exponents': arch_groups.values.tolist()
        }
    
    # 4. Multi-variable scaling (if multiple features available)
    if all(col in results_df.columns for col in ['model_size', 'layer_depth', 'critical_temperature']):
        x_vars = {
            'model_size': results_df['model_size'].values,
            'layer_depth': results_df['layer_depth'].values
        }
        y = results_df['critical_temperature'].values
        
        # Fit multi-variable scaling law
        multi_var_fit = multi_variable_scaling_law(x_vars, y, model='power_law')
        
        scaling_results['multi_variable_scaling'] = {
            'x_vars': x_vars,
            'y': y,
            'fit': multi_var_fit
        }
    
    return scaling_results