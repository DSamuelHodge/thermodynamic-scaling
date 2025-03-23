# Thermodynamic Scaling Analysis for Language Models

This repository provides a comprehensive framework for analyzing thermodynamic properties of transformer-based language models, with a focus on detecting and characterizing quantum-like criticality and scaling laws.

## Overview

This project investigates the thermodynamic properties of neural network weight matrices, specifically in transformer models, through the lens of statistical physics. It implements a suite of analyses to detect phase transitions, criticality, and universal scaling behavior.

Key features:
- Extract and analyze weight matrices from various pre-trained language models
- Calculate core thermodynamic quantities (energy, entropy, specific heat)
- Detect critical points and phase transitions
- Fit and analyze scaling laws
- Apply advanced theoretical frameworks (information theory, renormalization group, replica theory)
- Implement methodological enhancements (eigenvalue analysis, perturbation analysis, etc.)
- Visualize results with publication-quality plots

## Installation

```bash
# Clone the repository
git clone https://github.com/DSamuelHodge/thermodynamic-scaling.git
cd thermodynamic-scaling

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook

The main analysis is implemented in a Jupyter notebook:

```bash
jupyter notebook notebooks/thermodynamic_scaling_analysis.ipynb
```

### Python Package

The package can also be imported and used in your own code:

```python
import thermodynamic_scaling as ts

# Load and extract weights from a model
model_name = "gpt2"
weights, layer_info = ts.model_loading.extract_model_weights(model_name)

# Compute thermodynamic properties
temp_range = (0.05, 2.0)
results = ts.thermodynamics.compute_thermal_properties(weights["self_attention.query"], temp_range)

# Find critical points
critical_temp, critical_height = ts.thermodynamics.find_critical_point(results["temperatures"], results["specific_heat"])

# Visualize results
ts.visualization.plot_specific_heat_curve(results["temperatures"], results["specific_heat"], model_name, "Query Matrix")
```

## Theoretical Background

This project draws on principles from statistical mechanics, particularly the study of phase transitions and critical phenomena. We apply these concepts to understand the structure and behavior of neural network weight spaces.

The thermodynamic formalism allows us to:
1. Map neural network weights to physical systems
2. Detect phase transitions in the weight space
3. Identify universal scaling behaviors
4. Make connections to theories of learning and generalization

For more details, see the theoretical discussions in the notebook.

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This work builds upon research in the intersection of deep learning and statistical physics, particularly studies of criticality and scaling laws in neural networks.
