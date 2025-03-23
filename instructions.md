# Instructions for Running Thermodynamic Scaling Analysis

This document provides step-by-step instructions for setting up and running the thermodynamic scaling analysis code.

## Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Conda (recommended for environment management)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DSamuelHodge/thermodynamic-scaling.git
   cd thermodynamic-scaling
   ```

2. Create a Conda environment:
   ```bash
   conda create -n thermo-scaling python=3.9
   conda activate thermo-scaling
   ```

3. Install the package and dependencies:
   ```bash
   # Basic installation
   pip install -e .
   
   # With GPU support (optional)
   pip install -e ".[gpu]"
   
   # With Bayesian analysis support (optional)
   pip install -e ".[bayesian]"
   
   # With all optional dependencies
   pip install -e ".[gpu,bayesian]"
   ```

## Running the Analysis

### Using the Jupyter Notebook

The main analysis is implemented in a Jupyter notebook:

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open the notebook at `notebooks/thermodynamic_scaling_analysis.ipynb`

3. Run the cells in sequence. The notebook is structured as follows:
   - Setup and configuration
   - Model loading and weight extraction
   - Core thermodynamic analysis
   - Theoretical extensions
   - Visualization
   - Scaling law analysis

### Using the Python Package

You can also import and use the package in your own code:

```python
import thermodynamic_scaling as ts

# Load and extract weights from a model
model_name = "gpt2"
weights, layer_info = ts.model_loading.extract_model_weights(model_name)

# Compute thermodynamic properties
temp_range = (0.05, 2.0)
temps = np.linspace(temp_range[0], temp_range[1], 50)
weight_matrix = weights["layer_0"]["self_attention.query"]
results = ts.thermodynamics.compute_thermal_properties(weight_matrix, temps)

# Find critical points
critical_temp, critical_height = ts.thermodynamics.find_critical_point(
    results["temperatures"], results["specific_heat"]
)

# Visualize results
ts.visualization.plot_specific_heat_curves(
    model_name, {"layer_0": {"thermal_properties": results, "critical_temperature": critical_temp}}
)
```

## Customization

### Analyzing Different Models

To analyze different models, modify the `MODELS` list in the notebook or in your script:

```python
MODELS = [
    "gpt2",
    "gpt2-medium",
    "bert-base-uncased",
    # Add other models here
]
```

### GPU Acceleration

GPU acceleration is used automatically if available. To disable it:

```python
USE_GPU = False

# In function calls
ts.thermodynamics.compute_thermal_properties(weight_matrix, temps, use_gpu=USE_GPU)
```

### Saving Results

Results are saved automatically to the `results` directory. Visualizations are saved to the `figures` directory.

## Extending the Analysis

### Adding New Theoretical Extensions

1. Create a new module in `thermodynamic_scaling/theoretical_extensions/`
2. Update the `__init__.py` file to import your new module
3. Import your new module in the notebook or script

### Adding New Methodological Enhancements

1. Create a new module in `thermodynamic_scaling/methodological_enhancements/`
2. Update the `__init__.py` file to import your new module
3. Import your new module in the notebook or script

## Troubleshooting

### Memory Issues

If you encounter memory issues with large models:

1. Use a smaller subset of models
2. Limit the analysis to specific layer types
3. Use the `truncate_matrix` function in `utils.py` to reduce matrix sizes:
   ```python
   from thermodynamic_scaling.utils import truncate_matrix
   reduced_matrix = truncate_matrix(weight_matrix, max_size=1000)
   ```

### CUDA Issues

If you encounter CUDA errors while using GPU acceleration:

1. Disable GPU acceleration: `USE_GPU = False`
2. Ensure your CUDA and CuPy versions are compatible
3. Try a smaller model if you're running out of GPU memory

## Citation

If you use this code in your research, please cite:

```
@software{hodge2025thermodynamic,
  author = {Hodge, D. Samuel},
  title = {Thermodynamic Scaling Analysis for Language Models},
  year = {2025},
  url = {https://github.com/DSamuelHodge/thermodynamic-scaling}
}
```