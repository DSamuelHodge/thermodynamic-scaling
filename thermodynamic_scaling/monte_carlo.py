"""
Monte Carlo methods for sampling weight distributions.
"""

import numpy as np
from tqdm.auto import tqdm


def metropolis_hastings(weight_matrix, temperature, n_steps=10000, step_size=0.1):
    """
    Metropolis-Hastings algorithm for sampling Boltzmann distribution.
    
    Parameters:
    -----------
    weight_matrix : array_like
        Initial weight matrix to sample from.
    temperature : float
        Temperature parameter.
    n_steps : int, optional
        Number of sampling steps.
    step_size : float, optional
        Size of the random steps.
    
    Returns:
    --------
    tuple
        (samples, acceptance_rate)
    """
    # Flatten the weight matrix
    weights = weight_matrix.flatten()
    n_weights = len(weights)
    
    # Initialize samples array
    samples = np.zeros((n_steps, n_weights))
    samples[0] = weights.copy()
    
    # Calculate beta (inverse temperature)
    beta = 1.0 / temperature
    
    # Calculate initial energy
    current_energy = np.sum(weights**2)
    
    # Initialize acceptance counter
    accepted = 0
    
    # Perform sampling
    for i in tqdm(range(1, n_steps), desc="Metropolis-Hastings Sampling"):
        # Make a copy of the current state
        proposed_state = samples[i-1].copy()
        
        # Perturb a random element
        idx = np.random.randint(n_weights)
        proposed_state[idx] += np.random.normal(0, step_size)
        
        # Calculate new energy
        proposed_energy = np.sum(proposed_state**2)
        
        # Calculate acceptance probability
        delta_energy = proposed_energy - current_energy
        acceptance_prob = np.exp(-beta * delta_energy)
        
        # Accept or reject the proposed state
        if np.random.random() < acceptance_prob:
            samples[i] = proposed_state
            current_energy = proposed_energy
            accepted += 1
        else:
            samples[i] = samples[i-1]
    
    # Calculate acceptance rate
    acceptance_rate = accepted / (n_steps - 1)
    
    return samples, acceptance_rate


def wang_landau_sampling(weight_matrix, energy_range, bins=100, flatness_criterion=0.8, 
                        min_log_f=1e-8, n_iter_max=1000000):
    """
    Wang-Landau algorithm to estimate density of states.
    
    Parameters:
    -----------
    weight_matrix : array_like
        Weight matrix to analyze.
    energy_range : tuple
        (min_energy, max_energy) range to sample.
    bins : int, optional
        Number of energy bins.
    flatness_criterion : float, optional
        Criterion for histogram flatness (0-1).
    min_log_f : float, optional
        Minimum value of log(f) to stop the algorithm.
    n_iter_max : int, optional
        Maximum number of iterations.
    
    Returns:
    --------
    tuple
        (energy_bins, density_of_states)
    """
    # Flatten the weight matrix
    weights = weight_matrix.flatten()
    n_weights = len(weights)
    
    # Create energy bins
    min_energy, max_energy = energy_range
    energy_bins = np.linspace(min_energy, max_energy, bins+1)
    bin_centers = (energy_bins[:-1] + energy_bins[1:]) / 2
    bin_width = energy_bins[1] - energy_bins[0]
    
    # Initialize density of states (log scale) and histogram
    log_dos = np.zeros(bins)
    histogram = np.zeros(bins)
    
    # Initialize current state
    current_state = weights.copy()
    current_energy = np.sum(current_state**2)
    current_bin = int((current_energy - min_energy) / bin_width)
    
    # Check if initial state is within energy range
    if current_bin < 0 or current_bin >= bins:
        # If not, use a state in the middle of the range
        current_state = np.random.normal(0, 1, n_weights)
        current_state *= np.sqrt((min_energy + max_energy) / 2 / np.sum(current_state**2))
        current_energy = np.sum(current_state**2)
        current_bin = int((current_energy - min_energy) / bin_width)
    
    # Initialize modification factor
    log_f = 1.0
    
    # Main loop
    n_iter = 0
    step_size = 0.1
    
    with tqdm(total=n_iter_max, desc="Wang-Landau Sampling") as pbar:
        while log_f > min_log_f and n_iter < n_iter_max:
            # Propose a new state
            proposed_state = current_state.copy()
            
            # Perturb a random element
            idx = np.random.randint(n_weights)
            proposed_state[idx] += np.random.normal(0, step_size)
            
            # Calculate new energy
            proposed_energy = np.sum(proposed_state**2)
            proposed_bin = int((proposed_energy - min_energy) / bin_width)
            
            # Accept or reject the proposed state
            if 0 <= proposed_bin < bins:
                # Calculate acceptance probability
                if log_dos[proposed_bin] <= log_dos[current_bin]:
                    acceptance_prob = 1.0
                else:
                    acceptance_prob = np.exp(log_dos[current_bin] - log_dos[proposed_bin])
                
                if np.random.random() < acceptance_prob:
                    current_state = proposed_state
                    current_energy = proposed_energy
                    current_bin = proposed_bin
            
            # Update density of states and histogram
            log_dos[current_bin] += log_f
            histogram[current_bin] += 1
            
            # Check histogram flatness
            if np.min(histogram) > 0 and np.min(histogram) / np.mean(histogram) > flatness_criterion:
                # Reduce modification factor and reset histogram
                log_f /= 2.0
                histogram[:] = 0
            
            n_iter += 1
            pbar.update(1)
            
            # Adjust step size if needed
            if n_iter % 10000 == 0:
                acceptance_rate = np.sum(histogram > 0) / bins
                if acceptance_rate < 0.2:
                    step_size /= 1.5
                elif acceptance_rate > 0.8:
                    step_size *= 1.5
    
    # Normalize density of states
    density_of_states = np.exp(log_dos - np.max(log_dos))
    density_of_states /= np.sum(density_of_states * bin_width)
    
    return bin_centers, density_of_states


def parallel_tempering(weight_matrix, temperature_range, n_replicas=10, n_steps=10000, 
                      exchange_interval=10):
    """
    Parallel tempering to improve sampling at low temperatures.
    
    Parameters:
    -----------
    weight_matrix : array_like
        Initial weight matrix to sample from.
    temperature_range : tuple
        (min_temp, max_temp) range to sample.
    n_replicas : int, optional
        Number of replicas at different temperatures.
    n_steps : int, optional
        Number of sampling steps.
    exchange_interval : int, optional
        Interval for attempting exchange between replicas.
    
    Returns:
    --------
    dict
        Dictionary with sampling results.
    """
    # Flatten the weight matrix
    weights = weight_matrix.flatten()
    n_weights = len(weights)
    
    # Create temperature ladder
    min_temp, max_temp = temperature_range
    temperatures = np.logspace(np.log10(min_temp), np.log10(max_temp), n_replicas)
    betas = 1.0 / temperatures
    
    # Initialize replicas
    replicas = np.zeros((n_replicas, n_weights))
    for i in range(n_replicas):
        # Add some random noise to create different initial states
        replicas[i] = weights + np.random.normal(0, 0.01, n_weights)
    
    # Calculate initial energies
    energies = np.sum(replicas**2, axis=1)
    
    # Initialize samples array
    samples = np.zeros((n_replicas, n_steps, n_weights))
    
    # Store temperatures and energies
    temp_history = np.zeros((n_replicas, n_steps))
    energy_history = np.zeros((n_replicas, n_steps))
    
    # Initialize exchange acceptance counter
    exchange_attempts = 0
    exchange_accepted = 0
    
    # Perform sampling
    for step in tqdm(range(n_steps), desc="Parallel Tempering"):
        # Metropolis-Hastings updates for each replica
        for i in range(n_replicas):
            # Make a copy of the current state
            proposed_state = replicas[i].copy()
            
            # Perturb a random element
            idx = np.random.randint(n_weights)
            proposed_state[idx] += np.random.normal(0, 0.1)
            
            # Calculate new energy
            proposed_energy = np.sum(proposed_state**2)
            
            # Calculate acceptance probability
            delta_energy = proposed_energy - energies[i]
            acceptance_prob = np.exp(-betas[i] * delta_energy)
            
            # Accept or reject the proposed state
            if np.random.random() < acceptance_prob:
                replicas[i] = proposed_state
                energies[i] = proposed_energy
        
        # Try to exchange replicas
        if step % exchange_interval == 0:
            # Pick a random adjacent pair
            i = np.random.randint(n_replicas - 1)
            j = i + 1
            
            # Calculate acceptance probability for exchange
            delta_beta = betas[i] - betas[j]
            delta_energy = energies[i] - energies[j]
            exchange_prob = np.exp(delta_beta * delta_energy)
            
            exchange_attempts += 1
            
            # Try to exchange
            if np.random.random() < exchange_prob:
                # Swap replicas
                replicas[i], replicas[j] = replicas[j].copy(), replicas[i].copy()
                energies[i], energies[j] = energies[j], energies[i]
                exchange_accepted += 1
        
        # Store samples and data
        for i in range(n_replicas):
            samples[i, step] = replicas[i]
            temp_history[i, step] = temperatures[i]
            energy_history[i, step] = energies[i]
    
    # Calculate exchange acceptance rate
    exchange_rate = exchange_accepted / exchange_attempts if exchange_attempts > 0 else 0
    
    # Return results
    results = {
        'samples': samples,
        'temperatures': temperatures,
        'temp_history': temp_history,
        'energy_history': energy_history,
        'exchange_rate': exchange_rate
    }
    
    return results


def calculate_thermal_properties_mcmc(samples, temperature_range, n_temps=50):
    """
    Calculate thermodynamic properties from MCMC samples.
    
    Parameters:
    -----------
    samples : array_like
        Samples from MCMC simulation.
    temperature_range : tuple
        (min_temp, max_temp) range to evaluate.
    n_temps : int, optional
        Number of temperature points.
    
    Returns:
    --------
    dict
        Dictionary with thermodynamic properties.
    """
    temps = np.linspace(temperature_range[0], temperature_range[1], n_temps)
    
    # Calculate energies for all samples
    if samples.ndim == 3:  # Samples from parallel tempering
        n_replicas, n_steps, n_weights = samples.shape
        all_samples = samples.reshape(n_replicas * n_steps, n_weights)
    else:  # Samples from metropolis
        all_samples = samples
    
    energies = np.sum(all_samples**2, axis=1)
    
    # Initialize arrays for results
    specific_heats = np.zeros_like(temps)
    energies_mean = np.zeros_like(temps)
    entropies = np.zeros_like(temps)
    free_energies = np.zeros_like(temps)
    
    # Calculate properties for each temperature
    for i, T in enumerate(temps):
        beta = 1.0 / T
        
        # Calculate Boltzmann weights
        boltzmann_weights = np.exp(-beta * energies)
        Z = np.sum(boltzmann_weights)
        
        # Reweight samples using Boltzmann weights
        reweighted_energies = energies * boltzmann_weights / Z
        
        # Mean energy
        energies_mean[i] = np.sum(reweighted_energies)
        
        # Energy squared
        energy_squared = np.sum((energies**2) * boltzmann_weights / Z)
        
        # Specific heat
        specific_heats[i] = (energy_squared - energies_mean[i]**2) / (T**2)
        
        # Entropy
        entropies[i] = np.log(Z) + beta * energies_mean[i]
        
        # Free energy
        free_energies[i] = -T * np.log(Z)
    
    # Return results
    results = {
        'temperatures': temps,
        'specific_heat': specific_heats,
        'energy': energies_mean,
        'entropy': entropies,
        'free_energy': free_energies
    }
    
    return results


def multicanonical_sampling(weight_matrix, energy_range, n_steps=10000, n_bins=100, 
                           n_warmup=1000):
    """
    Multicanonical Monte Carlo sampling for flat energy histogram.
    
    Parameters:
    -----------
    weight_matrix : array_like
        Initial weight matrix to sample from.
    energy_range : tuple
        (min_energy, max_energy) range to sample.
    n_steps : int, optional
        Number of sampling steps.
    n_bins : int, optional
        Number of energy bins.
    n_warmup : int, optional
        Number of warmup steps to estimate weights.
    
    Returns:
    --------
    dict
        Dictionary with sampling results.
    """
    # Flatten the weight matrix
    weights = weight_matrix.flatten()
    n_weights = len(weights)
    
    # Create energy bins
    min_energy, max_energy = energy_range
    energy_bins = np.linspace(min_energy, max_energy, n_bins+1)
    bin_centers = (energy_bins[:-1] + energy_bins[1:]) / 2
    bin_width = energy_bins[1] - energy_bins[0]
    
    # Initialize current state
    current_state = weights.copy()
    current_energy = np.sum(current_state**2)
    
    # Initialize histogram and weights
    # Start with uniform weights (canonical ensemble)
    histogram = np.zeros(n_bins)
    log_weights = np.zeros(n_bins)
    
    # Warmup phase: use Wang-Landau to estimate weights
    if n_warmup > 0:
        print("Performing warmup using Wang-Landau...")
        
        # Initialize modification factor
        log_f = 1.0
        
        for step in tqdm(range(n_warmup), desc="Wang-Landau Warmup"):
            # Propose a new state
            proposed_state = current_state.copy()
            idx = np.random.randint(n_weights)
            proposed_state[idx] += np.random.normal(0, 0.1)
            
            proposed_energy = np.sum(proposed_state**2)
            
            # Get bin indices
            current_bin = int((current_energy - min_energy) / bin_width)
            current_bin = max(0, min(n_bins-1, current_bin))
            
            proposed_bin = int((proposed_energy - min_energy) / bin_width)
            proposed_bin = max(0, min(n_bins-1, proposed_bin))
            
            # Calculate acceptance probability
            # exp(current_log_weight - proposed_log_weight)
            if log_weights[proposed_bin] <= log_weights[current_bin]:
                acceptance_prob = 1.0
            else:
                acceptance_prob = np.exp(log_weights[current_bin] - log_weights[proposed_bin])
            
            # Accept or reject
            if np.random.random() < acceptance_prob:
                current_state = proposed_state
                current_energy = proposed_energy
                current_bin = proposed_bin
            
            # Update weights
            log_weights[current_bin] += log_f
            histogram[current_bin] += 1
            
            # Reduce modification factor
            if step % 1000 == 0 and step > 0:
                log_f /= 2.0
    
    # Reset for production run
    histogram = np.zeros(n_bins)
    samples = np.zeros((n_steps, n_weights))
    energies = np.zeros(n_steps)
    
    # Production run with fixed weights
    for step in tqdm(range(n_steps), desc="Multicanonical Sampling"):
        # Propose a new state
        proposed_state = current_state.copy()
        idx = np.random.randint(n_weights)
        proposed_state[idx] += np.random.normal(0, 0.1)
        
        proposed_energy = np.sum(proposed_state**2)
        
        # Get bin indices
        current_bin = int((current_energy - min_energy) / bin_width)
        current_bin = max(0, min(n_bins-1, current_bin))
        
        proposed_bin = int((proposed_energy - min_energy) / bin_width)
        proposed_bin = max(0, min(n_bins-1, proposed_bin))
        
        # Calculate acceptance probability
        if log_weights[proposed_bin] <= log_weights[current_bin]:
            acceptance_prob = 1.0
        else:
            acceptance_prob = np.exp(log_weights[current_bin] - log_weights[proposed_bin])
        
        # Accept or reject
        if np.random.random() < acceptance_prob:
            current_state = proposed_state
            current_energy = proposed_energy
            current_bin = proposed_bin
        
        # Store sample
        samples[step] = current_state
        energies[step] = current_energy
        histogram[current_bin] += 1
    
    # Calculate density of states from weights
    log_dos = -log_weights
    log_dos -= np.min(log_dos)  # Normalize
    
    return {
        'samples': samples,
        'energies': energies,
        'histogram': histogram,
        'energy_bins': bin_centers,
        'log_weights': log_weights,
        'log_dos': log_dos
    }