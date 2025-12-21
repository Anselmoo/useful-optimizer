"""Multi-objective optimization algorithms.

This module provides implementations of multi-objective optimization algorithms
that find Pareto-optimal solutions for problems with multiple competing objectives.

Available Algorithms:
    - AbstractMultiObjectiveOptimizer: Base class for multi-objective optimizers
    - MOEAD: Multi-Objective EA based on Decomposition
    - NSGAII: Non-dominated Sorting Genetic Algorithm II
    - SPEA2: Strength Pareto Evolutionary Algorithm 2

References:
    Deb, K. (2001). Multi-Objective Optimization using Evolutionary Algorithms.
    Wiley, Chichester, UK.

    Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm
    based on decomposition. IEEE Trans. Evol. Comput., 11(6), 712-731.

    Zitzler, E., Laumanns, M., & Thiele, L. (2001). SPEA2: Improving the
    strength pareto evolutionary algorithm. TIK-Report 103, ETH Zurich.
"""

from __future__ import annotations

from opt.multi_objective.abstract_multi_objective import AbstractMultiObjectiveOptimizer
from opt.multi_objective.moead import MOEAD
from opt.multi_objective.nsga_ii import NSGAII
from opt.multi_objective.spea2 import SPEA2


__all__ = ["AbstractMultiObjectiveOptimizer", "MOEAD", "NSGAII", "SPEA2"]
