"""
Visualization functions for thermodynamic analysis results.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import warnings

# Set Seaborn style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

# Define custom color schemes
COLORS = sns.color_palette("viridis", n_colors=10)
MODEL_COLORS = {
    'gpt2': COLORS[0],
    'gpt2-medium': COLORS[1],
    'gpt2-large': COLORS[2],
    'facebook/opt-125m': COLORS[3],
    'facebook/opt-1.3b': COLORS[4],
    'EleutherAI/pythia-70m': COLORS[5],
    'EleutherAI/pythia-410m': COLORS[6],
    'HuggingFaceTB/SmolLM2-135M': COLORS[7]
}


def plot_specific_heat_curves(model_name, results, layer_types=None, figsize=(10, 6), 
                             save_path=None, show_critical=True):
    """
    Plot specific heat vs temperature for selected layers.
    
    Parameters:
    -----------
    model_name : str
        Name of the model.
    results : dict
        Dictionary containing thermal properties and layer information.
    layer_types : list, optional
        Types of layers to include. If None, plot all layers.
    figsize : tuple, optional
        Figure size (width, height).
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    show_critical : bool, optional
        Whether to mark critical points.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter layers if layer_types is specified
    layer_results = {}
    for layer_name, layer_data in results.items():
        if layer_types is None or any(lt in layer_name for lt in layer_types):
            layer_results[layer_name] = layer_data
    
    # Plot specific heat curves for each layer
    for i, (layer_name, layer_data) in enumerate(layer_results.items()):
        if 'thermal_properties' not in layer_data:
            continue
        
        thermal_props = layer_data['thermal_properties']
        temps = thermal_props['temperatures']
        specific_heat = thermal_props['specific_heat']
        
        # Plot the curve
        color = COLORS[i % len(COLORS)]
        ax.plot(temps, specific_heat, label=layer_name, color=color, linewidth=2)
        
        # Mark critical point if requested
        if show_critical and 'critical_temperature' in layer_data:
            critical_temp = layer_data['critical_temperature']
            critical_heat = np.interp(critical_temp, temps, specific_heat)
            ax.scatter(critical_temp, critical_heat, color=color, s=100, zorder=10,
                      edgecolor='white', linewidth=1)
    
    # Set labels and title
    ax.set_xlabel('Temperature (T)')
    ax.set_ylabel('Specific Heat (C)')
    ax.set_title(f'Specific Heat Curves for {model_name}')
    
    # Add legend
    if len(layer_results) > 1:
        ax.legend(title='Layer', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_scaling_law(results_df, x_col, y_col, hue_col=None, figsize=(10, 6),
                    log_scale=True, fit=True, save_path=None):
    """
    Create scaling law plot with optional fit line.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing results.
    x_col : str
        Column name for x-axis.
    y_col : str
        Column name for y-axis.
    hue_col : str, optional
        Column name for color grouping.
    figsize : tuple, optional
        Figure size (width, height).
    log_scale : bool, optional
        Whether to use log-log scale.
    fit : bool, optional
        Whether to show power law fit.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure.
    """
    import pandas as pd
    
    # Check if input is a DataFrame
    if not isinstance(results_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    if hue_col:
        scatter = sns.scatterplot(data=results_df, x=x_col, y=y_col, hue=hue_col, 
                                 s=100, alpha=0.8, ax=ax)
    else:
        scatter = sns.scatterplot(data=results_df, x=x_col, y=y_col, 
                                 s=100, alpha=0.8, ax=ax)
    
    # Set log scale if requested
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # Add power law fit if requested
    if fit:
        from ..scaling_law import fit_power_law
        
        x = results_df[x_col].values
        y = results_df[y_col].values
        
        # Fit power law
        power_law_fit = fit_power_law(x, y)
        
        if power_law_fit['success']:
            # Get fit parameters
            a = power_law_fit['best']['a']
            b = power_law_fit['best']['b']
            
            # Generate fit line
            if log_scale:
                x_fit = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)
            else:
                x_fit = np.linspace(min(x), max(x), 100)
            
            y_fit = a * x_fit**b
            
            # Plot fit line
            ax.plot(x_fit, y_fit, 'k--', linewidth=2, 
                   label=f'$y = {a:.3e} \\cdot x^{{{b:.3f}}}$')
            
            # Add equation text
            r_squared = power_law_fit['best']['r_squared']
            equation_text = f'$y = {a:.3e} \\cdot x^{{{b:.3f}}}$\n$R^2 = {r_squared:.3f}$'
            
            # Place text in the corner (adjust position based on log scale)
            if log_scale:
                ax.text(0.05, 0.95, equation_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='left',
                       bbox=dict(facecolor='white', alpha=0.7))
            else:
                ax.text(0.95, 0.05, equation_text, transform=ax.transAxes,
                       verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(facecolor='white', alpha=0.7))
    
    # Set labels with LaTeX formatting
    x_label = x_col.replace('_', ' ').title()
    y_label = y_col.replace('_', ' ').title()
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'{y_label} vs {x_label} Scaling')
    
    # Add legend
    if fit or hue_col:
        if hue_col:
            # Move legend outside plot
            plt.legend(title=hue_col.replace('_', ' ').title(), 
                      bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_model_comparisons(results_df, property_col, groupby='model_name', 
                          figsize=(12, 6), save_path=None):
    """
    Create comparison plots across models for a given property.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing results.
    property_col : str
        Column name for the property to compare.
    groupby : str, optional
        Column name to group by.
    figsize : tuple, optional
        Figure size (width, height).
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure.
    """
    import pandas as pd
    
    # Check if input is a DataFrame
    if not isinstance(results_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group by the specified column and calculate statistics
    grouped = results_df.groupby(groupby)[property_col].agg(['mean', 'std']).reset_index()
    
    # Sort by mean value
    grouped = grouped.sort_values('mean')
    
    # Create bar plot
    bars = ax.bar(grouped[groupby], grouped['mean'], yerr=grouped['std'],
                 alpha=0.8, capsize=5, color=COLORS[:len(grouped)])
    
    # Set labels
    ax.set_xlabel(groupby.replace('_', ' ').title())
    ax.set_ylabel(property_col.replace('_', ' ').title())
    ax.set_title(f'{property_col.replace("_", " ").title()} by {groupby.replace("_", " ").title()}')
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_renormalization_flow(flow_results, figsize=(12, 8), save_path=None):
    """
    Visualize RG flow of thermodynamic quantities.
    
    Parameters:
    -----------
    flow_results : dict
        Dictionary with RG flow data.
    figsize : tuple, optional
        Figure size (width, height).
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure.
    """
    # Extract data
    temperatures = flow_results['temperatures']
    specific_heats = flow_results['specific_heat']
    critical_temps = flow_results['critical_temps']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Plot specific heat curves
    for i, C in enumerate(specific_heats):
        label = f'RG Step {i}' if i > 0 else 'Original'
        color = COLORS[i % len(COLORS)]
        axes[0].plot(temperatures, C, label=label, color=color, linewidth=2)
        
        # Mark critical point
        if i < len(critical_temps):
            critical_T = critical_temps[i]
            critical_C = np.interp(critical_T, temperatures, C)
            axes[0].scatter(critical_T, critical_C, color=color, s=100, zorder=10,
                           edgecolor='white', linewidth=1)
    
    # Set labels for the first subplot
    axes[0].set_xlabel('Temperature (T)')
    axes[0].set_ylabel('Specific Heat (C)')
    axes[0].set_title('Specific Heat Evolution Under RG Flow')
    axes[0].legend()
    
    # Plot critical temperature evolution
    axes[1].plot(range(len(critical_temps)), critical_temps, 'o-', color=COLORS[0],
                linewidth=2, markersize=10)
    
    # Set labels for the second subplot
    axes[1].set_xlabel('RG Step')
    axes[1].set_ylabel('Critical Temperature (Tc)')
    axes[1].set_title('Critical Temperature Evolution')
    
    # Add grid
    axes[1].grid(True)
    
    # If critical exponents are available, add them as text
    if 'critical_exponents' in flow_results and flow_results['critical_exponents']:
        exponents = flow_results['critical_exponents']
        exponent_text = 'Critical Exponents:\n'
        for i, exp in enumerate(exponents):
            exponent_text += f'ν{i+1} = {exp:.4f}\n'
        
        axes[1].text(0.05, 0.95, exponent_text, transform=axes[1].transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.7))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_eigenvalue_spectrum(eigenvalues, with_mp_law=True, aspect_ratio=None,
                            figsize=(10, 6), bins=50, save_path=None):
    """
    Plot eigenvalue spectrum with optional Marchenko-Pastur law.
    
    Parameters:
    -----------
    eigenvalues : array_like
        Array of eigenvalues.
    with_mp_law : bool, optional
        Whether to show the Marchenko-Pastur distribution.
    aspect_ratio : float, optional
        Aspect ratio (m/n) of the matrix. Required if with_mp_law=True.
    figsize : tuple, optional
        Figure size (width, height).
    bins : int, optional
        Number of bins for histogram.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize eigenvalues
    normalized_eigenvalues = eigenvalues / np.mean(eigenvalues)
    
    # Create histogram
    hist, bin_edges, _ = ax.hist(normalized_eigenvalues, bins=bins, density=True,
                               alpha=0.7, color=COLORS[0], label='Empirical Distribution')
    
    # Add Marchenko-Pastur distribution if requested
    if with_mp_law:
        if aspect_ratio is None:
            warnings.warn("Aspect ratio not provided. Marchenko-Pastur law not shown.")
        else:
            # Ensure aspect ratio ≤ 1 for M-P law
            q = min(aspect_ratio, 1/aspect_ratio)
            
            # Calculate M-P distribution bounds
            lambda_min = (1 - np.sqrt(1/q))**2
            lambda_max = (1 + np.sqrt(1/q))**2
            
            # Generate x values for M-P distribution
            x = np.linspace(max(0, lambda_min*0.9), lambda_max*1.1, 1000)
            
            # Calculate M-P density
            def mp_density(x, q):
                if x <= 0:
                    return 0
                if x < lambda_min or x > lambda_max:
                    return 0
                return np.sqrt(np.maximum(0, (lambda_max - x) * (x - lambda_min))) / (2 * np.pi * x * q)
            
            mp_pdf = np.array([mp_density(xi, q) for xi in x])
            
            # Add point mass at zero if q < 1
            if q < 1:
                # Fraction of eigenvalues at zero
                zero_mass = 1 - q
                
                # Find bin containing zero
                zero_bin = np.searchsorted(bin_edges, 0)
                if zero_bin < len(bin_edges):
                    # Add delta function at zero (represented as a narrow bar)
                    bin_width = bin_edges[1] - bin_edges[0]
                    ax.bar(0, zero_mass/bin_width, width=bin_width, color=COLORS[1], alpha=0.7)
            
            # Plot M-P distribution
            ax.plot(x, mp_pdf, 'r-', linewidth=2, label='Marchenko-Pastur Law')
    
    # Set labels
    ax.set_xlabel('Normalized Eigenvalue')
    ax.set_ylabel('Density')
    ax.set_title('Eigenvalue Spectrum')
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_2d_phase_diagram(param1_values, param2_values, critical_temps, 
                         param1_name='Param1', param2_name='Param2',
                         figsize=(10, 8), save_path=None):
    """
    Create a 2D phase diagram with critical temperature contours.
    
    Parameters:
    -----------
    param1_values : array_like
        Values for the first parameter.
    param2_values : array_like
        Values for the second parameter.
    critical_temps : array_like
        Critical temperatures for each parameter combination.
    param1_name : str, optional
        Name for the first parameter.
    param2_name : str, optional
        Name for the second parameter.
    figsize : tuple, optional
        Figure size (width, height).
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure.
    """
    # Convert inputs to numpy arrays
    param1_values = np.asarray(param1_values)
    param2_values = np.asarray(param2_values)
    critical_temps = np.asarray(critical_temps)
    
    # Create a mesh grid for contour plot
    unique_param1 = np.unique(param1_values)
    unique_param2 = np.unique(param2_values)
    
    if len(unique_param1) <= 1 or len(unique_param2) <= 1:
        raise ValueError("Need at least 2 unique values for each parameter")
    
    # Reshape critical_temps to a 2D grid
    Tc_grid = np.zeros((len(unique_param1), len(unique_param2)))
    
    for i, p1 in enumerate(unique_param1):
        for j, p2 in enumerate(unique_param2):
            mask = (param1_values == p1) & (param2_values == p2)
            if np.any(mask):
                Tc_grid[i, j] = np.mean(critical_temps[mask])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create contour plot
    X, Y = np.meshgrid(unique_param2, unique_param1)
    
    # Define custom colormap (from cool to warm)
    colors = plt.cm.coolwarm
    
    # Create filled contour
    contour = ax.contourf(X, Y, Tc_grid, levels=20, cmap=colors, alpha=0.8)
    
    # Add contour lines
    contour_lines = ax.contour(X, Y, Tc_grid, levels=10, colors='k', linewidths=0.5)
    
    # Add contour labels
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Critical Temperature')
    
    # Set labels
    ax.set_xlabel(param2_name)
    ax.set_ylabel(param1_name)
    ax.set_title('Critical Temperature Phase Diagram')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_free_energy_landscape(free_energy_data, temperature, figsize=(10, 6), save_path=None):
    """
    Visualize the free energy landscape at a given temperature.
    
    Parameters:
    -----------
    free_energy_data : dict
        Dictionary with free energy landscape data.
    temperature : float
        Temperature at which to plot the landscape.
    figsize : tuple, optional
        Figure size (width, height).
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract free energy and other metrics
    energy_vals = free_energy_data.get('energy_samples', [])
    if not energy_vals:
        warnings.warn("No energy samples found in free energy data")
        return fig
    
    # Create a kernel density estimate
    from scipy.stats import gaussian_kde
    
    # Create a grid of values
    energy_min = min(energy_vals)
    energy_max = max(energy_vals)
    energy_range = energy_max - energy_min
    grid = np.linspace(energy_min - 0.1*energy_range, energy_max + 0.1*energy_range, 1000)
    
    # Apply KDE
    kde = gaussian_kde(energy_vals)
    density = kde(grid)
    
    # Calculate Boltzmann distribution
    beta = 1.0 / temperature
    boltzmann = np.exp(-beta * grid)
    boltzmann = boltzmann / np.sum(boltzmann * (grid[1] - grid[0]))
    
    # Calculate free energy landscape
    F = -temperature * np.log(density + 1e-10)
    
    # Normalize free energy to have minimum at zero
    F = F - np.min(F)
    
    # Plot free energy landscape
    ax.plot(grid, F, color=COLORS[0], linewidth=2, label='Free Energy')
    
    # Mark minima
    from scipy.signal import find_peaks
    minima, _ = find_peaks(-F)
    for i, idx in enumerate(minima):
        if i == 0:
            ax.scatter(grid[idx], F[idx], color='red', s=100, zorder=10, 
                      label='Minima')
        else:
            ax.scatter(grid[idx], F[idx], color='red', s=100, zorder=10)
    
    # Add energy distribution as a secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(grid, density, color=COLORS[1], linestyle='--', linewidth=1.5,
            label='Energy Density')
    
    # Set labels
    ax.set_xlabel('Energy')
    ax.set_ylabel('Free Energy (F)')
    ax2.set_ylabel('Energy Distribution')
    
    ax.set_title(f'Free Energy Landscape at T = {temperature:.2f}')
    
    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_scaling_collapse(collapse_results, figsize=(10, 6), save_path=None):
    """
    Plot data collapse using scaled variables to test universality.
    
    Parameters:
    -----------
    collapse_results : dict
        Dictionary with data collapse results.
    figsize : tuple, optional
        Figure size (width, height).
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Extract critical exponents
    critical_exponents = collapse_results['critical_exponents']
    nu = critical_exponents['nu']
    beta = critical_exponents.get('beta', 1.0)
    Tc = critical_exponents['Tc']
    
    # Get scaled data
    scaled_data = collapse_results['scaled_data']
    
    # Plot raw data (left panel)
    for i, (label, data) in enumerate(scaled_data.items()):
        color = COLORS[i % len(COLORS)]
        axes[0].plot(data['x_raw'], data['y_raw'], 'o-', label=f'{label} (L={data["L"]:.1f})',
                    color=color, markersize=5)
        
        # Mark critical temperature
        axes[0].axvline(x=Tc, color='k', linestyle='--', alpha=0.5)
    
    # Plot scaled data (right panel)
    for i, (label, data) in enumerate(scaled_data.items()):
        color = COLORS[i % len(COLORS)]
        axes[1].plot(data['x_scaled'], data['y_scaled'], 'o', label=f'{label} (L={data["L"]:.1f})',
                    color=color, markersize=5)
    
    # Set labels
    axes[0].set_xlabel('Temperature (T)')
    axes[0].set_ylabel('Raw Data')
    axes[0].set_title('Raw Data')
    
    axes[1].set_xlabel(r'$L^{1/\nu} (T-T_c)/T_c)
    axes[1].set_ylabel(r'$L^{-\beta/\nu} \cdot Data)
    axes[1].set_title('Scaled Data')
    
    # Add critical exponents as text
    exponent_text = f'$\\nu = {nu:.4f}$\n$\\beta = {beta:.4f}$\n$T_c = {Tc:.4f}
    axes[1].text(0.05, 0.95, exponent_text, transform=axes[1].transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Add goodness of collapse
    quality = collapse_results['collapse_quality']
    quality_text = f'Collapse Quality: {quality:.3f}'
    axes[1].text(0.05, 0.80, quality_text, transform=axes[1].transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Add legend
    for ax in axes:
        ax.legend(fontsize='small')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_information_metrics(info_theory_results, figsize=(12, 10), save_path=None):
    """
    Plot information-theoretic metrics as functions of temperature.
    
    Parameters:
    -----------
    info_theory_results : dict
        Dictionary with information theory results.
    figsize : tuple, optional
        Figure size (width, height).
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Extract temperatures
    temperatures = info_theory_results['temperatures']
    
    # Plot entropy
    if 'entropy' in info_theory_results:
        axes[0].plot(temperatures, info_theory_results['entropy'], 'o-', 
                    color=COLORS[0], linewidth=2)
        axes[0].set_xlabel('Temperature (T)')
        axes[0].set_ylabel('Entropy')
        axes[0].set_title('Entropy vs Temperature')
    
    # Plot Fisher information
    if 'fisher_information' in info_theory_results:
        axes[1].plot(temperatures, info_theory_results['fisher_information'], 'o-',
                    color=COLORS[1], linewidth=2)
        axes[1].set_xlabel('Temperature (T)')
        axes[1].set_ylabel('Fisher Information')
        axes[1].set_title('Fisher Information vs Temperature')
        
        # Mark peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(info_theory_results['fisher_information'])
        for peak in peaks:
            axes[1].scatter(temperatures[peak], info_theory_results['fisher_information'][peak],
                           color='red', s=100, zorder=10)
    
    # Plot complexity measures
    if 'complexity' in info_theory_results:
        metrics = info_theory_results['complexity']
        
        if 'effective_dimension' in metrics:
            axes[2].plot(temperatures, metrics['effective_dimension'], 'o-',
                        color=COLORS[2], linewidth=2)
            axes[2].set_xlabel('Temperature (T)')
            axes[2].set_ylabel('Effective Dimension')
            axes[2].set_title('Effective Dimension vs Temperature')
        
        if 'participation_ratio' in metrics:
            axes[3].plot(temperatures, metrics['participation_ratio'], 'o-',
                        color=COLORS[3], linewidth=2)
            axes[3].set_xlabel('Temperature (T)')
            axes[3].set_ylabel('Participation Ratio')
            axes[3].set_title('Participation Ratio vs Temperature')
    
    # Add grid to all subplots
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_all_visualizations(results_df, scaling_results, info_theory_results, 
                              rg_results, eigenvalue_results, perturbation_results, 
                              save_dir=None):
    """
    Generate all visualizations for the analysis.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with model results.
    scaling_results : dict
        Dictionary with scaling law results.
    info_theory_results : dict
        Dictionary with information theory results.
    rg_results : dict
        Dictionary with renormalization group results.
    eigenvalue_results : dict
        Dictionary with eigenvalue analysis results.
    perturbation_results : dict
        Dictionary with perturbation analysis results.
    save_dir : str, optional
        Directory to save figures. If None, figures are not saved.
    
    Returns:
    --------
    dict
        Dictionary with generated figures.
    """
    import os
    figures = {}
    
    # Create save directory if specified
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # 1. Specific heat curves for each model
    for model_name in results_df['model_name'].unique():
        model_results = results_df[results_df['model_name'] == model_name]
        
        # Extract layers
        model_layers = {}
        for _, row in model_results.iterrows():
            layer_name = row['layer_name']
            model_layers[layer_name] = {
                'thermal_properties': row['thermal_properties'],
                'critical_temperature': row['critical_temperature']
            }
        
        # Create figure
        save_path = os.path.join(save_dir, f"{model_name.replace('/', '_')}_specific_heat.png") if save_dir else None
        fig = plot_specific_heat_curves(model_name, model_layers, save_path=save_path)
        figures[f"{model_name}_specific_heat"] = fig
    
    # 2. Scaling law plots
    for scaling_name, scaling_data in scaling_results.items():
        if 'fit' in scaling_data and scaling_data['fit']['success']:
            # Extract data
            x = scaling_data['x']
            y = scaling_data['y']
            
            # Create temporary DataFrame
            import pandas as pd
            temp_df = pd.DataFrame({'x': x, 'y': y})
            
            # Create figure
            save_path = os.path.join(save_dir, f"{scaling_name}.png") if save_dir else None
            fig = plot_scaling_law(temp_df, 'x', 'y', figsize=(8, 6), save_path=save_path)
            figures[scaling_name] = fig
    
    # 3. Information theory metrics
    if info_theory_results:
        save_path = os.path.join(save_dir, "information_metrics.png") if save_dir else None
        fig = plot_information_metrics(info_theory_results, save_path=save_path)
        figures["information_metrics"] = fig
    
    # 4. RG flow
    if rg_results:
        save_path = os.path.join(save_dir, "rg_flow.png") if save_dir else None
        fig = plot_renormalization_flow(rg_results, save_path=save_path)
        figures["rg_flow"] = fig
    
    # 5. Eigenvalue spectrum
    if eigenvalue_results and 'eigenvalues' in eigenvalue_results:
        eigenvalues = eigenvalue_results['eigenvalues']
        aspect_ratio = eigenvalue_results.get('aspect_ratio', None)
        
        save_path = os.path.join(save_dir, "eigenvalue_spectrum.png") if save_dir else None
        fig = plot_eigenvalue_spectrum(eigenvalues, with_mp_law=True, aspect_ratio=aspect_ratio,
                                      save_path=save_path)
        figures["eigenvalue_spectrum"] = fig
    
    return figures