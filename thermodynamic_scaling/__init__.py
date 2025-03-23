"""
Thermodynamic Scaling Analysis for Language Models

This package provides tools for analyzing the thermodynamic properties
of transformer-based language models, focusing on detecting and 
characterizing quantum-like criticality and scaling laws.
"""

__version__ = "0.1.0"

from . import model_loading
from . import thermodynamics
from . import theoretical_extensions
from . import scaling_law
from . import visualization
from . import monte_carlo

__all__ = [
    "model_loading",
    "thermodynamics",
    "theoretical_extensions",
    "scaling_law",
    "visualization",
    "monte_carlo"
]