"""Physics-inspired optimization algorithms.

This module provides implementations of optimization algorithms inspired by
physical phenomena such as gravity, thermodynamics, and electromagnetic forces.

Available Algorithms:
    - AtomSearchOptimizer: Atom Search Optimization (ASO)
    - EquilibriumOptimizer: Equilibrium Optimizer (EO)
    - GravitationalSearchOptimizer: Gravitational Search Algorithm (GSA)
    - RIMEOptimizer: RIME ice formation optimization

References:
    Rashedi, E., Nezamabadi-pour, H., & Saryazdi, S. (2009). GSA: A Gravitational
    Search Algorithm. Information Sciences, 179(13), 2232-2248.

    Zhao, W., Wang, L., & Zhang, Z. (2019). Atom search optimization.
    Knowledge-Based Systems, 163, 283-304.
"""

from __future__ import annotations

from opt.physics_inspired.atom_search import AtomSearchOptimizer
from opt.physics_inspired.equilibrium_optimizer import EquilibriumOptimizer
from opt.physics_inspired.gravitational_search import GravitationalSearchOptimizer
from opt.physics_inspired.rime_optimizer import RIMEOptimizer


__all__: list[str] = [
    "AtomSearchOptimizer",
    "EquilibriumOptimizer",
    "GravitationalSearchOptimizer",
    "RIMEOptimizer",
]
