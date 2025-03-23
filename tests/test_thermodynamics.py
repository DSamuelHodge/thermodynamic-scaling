import pytest
import numpy as np
from thermodynamic_scaling.thermodynamics import compute_thermal_properties

def test_compute_thermal_properties_basic():
    # Create a simple 2x2 weight matrix
    matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    temps = np.array([0.5, 1.0, 1.5])
    
    # Compute properties
    results = compute_thermal_properties(matrix, temps)
    
    # Check expected dictionary keys
    expected_keys = {'temperatures', 'specific_heat', 'energy', 'entropy', 'free_energy'}
    assert set(results.keys()) == expected_keys
    
    # Check array shapes
    for key in expected_keys:
        assert len(results[key]) == len(temps)
        assert isinstance(results[key], np.ndarray)

def test_compute_thermal_properties_physical_validity():
    # Create random weight matrix
    np.random.seed(42)
    matrix = np.random.randn(5, 5)
    temps = np.linspace(0.1, 2.0, 20)
    
    results = compute_thermal_properties(matrix, temps)
    
    # Test entropy increases with temperature (second law of thermodynamics)
    entropy = results['entropy']
    assert np.all(np.diff(entropy) >= -1e-10)  # Allow small numerical errors
    
    # Test specific heat is positive
    assert np.all(results['specific_heat'] >= -1e-10)

def test_compute_thermal_properties_temperature_invariance():
    matrix = np.array([[1.0, -1.0], [-1.0, 1.0]])
    temps1 = np.array([1.0, 2.0, 3.0])
    temps2 = temps1 * 2  # Scale temperatures
    
    results1 = compute_thermal_properties(matrix, temps1)
    results2 = compute_thermal_properties(matrix, temps2)
    
    # Check scaling behavior of energy
    energy_ratio = results2['energy'] / results1['energy']
    assert np.allclose(energy_ratio, 1.0, rtol=1e-5)

HAS_GPU = False  # Set to True if GPU support is available
@pytest.mark.skipif(not HAS_GPU, reason="GPU not available")
def test_compute_thermal_properties_gpu():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    temps = np.array([0.5, 1.0])
    
    # Compare CPU and GPU results
    cpu_results = compute_thermal_properties(matrix, temps, use_gpu=False)
    gpu_results = compute_thermal_properties(matrix, temps, use_gpu=True)
    
    for key in cpu_results:
        assert np.allclose(cpu_results[key], gpu_results[key], rtol=1e-5)

def test_compute_thermal_properties_invalid_input():
    # Test with invalid temperature values
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    temps = np.array([-1.0, 0.0])  # Invalid negative temperature
    
    with pytest.raises(ValueError):
        compute_thermal_properties(matrix, temps)
    
    # Test with empty matrix
    empty_matrix = np.array([])
    valid_temps = np.array([1.0, 2.0])
    
    with pytest.raises(ValueError):
        compute_thermal_properties(empty_matrix, valid_temps)

def test_compute_thermal_properties_k_b_scaling():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    temps = np.array([1.0, 2.0])
    
    # Compare results with different Boltzmann constants
    results1 = compute_thermal_properties(matrix, temps, k_B=1.0)
    results2 = compute_thermal_properties(matrix, temps, k_B=2.0)
    
    # Energy should scale with k_B
    assert np.allclose(results2['energy'], results1['energy'], rtol=1e-5)
    
    # Entropy should scale with k_B
    assert np.allclose(results2['entropy']/2.0, results1['entropy'], rtol=1e-5)
