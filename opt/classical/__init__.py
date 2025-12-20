"""Classical optimization algorithms.

This module contains traditional mathematical optimization methods including derivative-based
and derivative-free approaches. Includes: BFGS, Conjugate Gradient, Hill Climbing, L-BFGS,
Nelder-Mead, Powell, Simulated Annealing, Tabu Search, and Trust Region methods.
"""

from __future__ import annotations

from opt.classical.bfgs import BFGS
from opt.classical.conjugate_gradient import ConjugateGradient
from opt.classical.hill_climbing import HillClimbing
from opt.classical.lbfgs import LBFGS
from opt.classical.nelder_mead import NelderMead
from opt.classical.powell import Powell
from opt.classical.simulated_annealing import SimulatedAnnealing
from opt.classical.tabu_search import TabuSearch
from opt.classical.trust_region import TrustRegion


__all__: list[str] = [
    "BFGS",
    "LBFGS",
    "ConjugateGradient",
    "HillClimbing",
    "NelderMead",
    "Powell",
    "SimulatedAnnealing",
    "TabuSearch",
    "TrustRegion",
]
