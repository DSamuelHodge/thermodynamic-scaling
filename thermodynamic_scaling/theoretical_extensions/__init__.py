"""
Theoretical extensions for thermodynamic analysis of neural networks.

This package contains implementations of advanced theoretical frameworks
for analyzing the thermodynamic properties of neural network weights.
"""

from . import information_theory
from . import renormalization_group
from . import replica_theory

__all__ = [
    'information_theory',
    'renormalization_group',
    'replica_theory'
]