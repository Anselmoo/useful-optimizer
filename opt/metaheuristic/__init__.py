"""Metaheuristic optimization algorithms.

This module contains high-level problem-independent algorithmic frameworks that provide
strategies for exploring search spaces. Includes: Colliding Bodies, Cross-Entropy,
Eagle Strategy, Harmony Search, Particle Filter, Shuffled Frog Leaping, Sine Cosine,
Stochastic Diffusion/Fractal Search, and Variable Neighborhood Search variants.
"""

from __future__ import annotations

from opt.metaheuristic.colliding_bodies_optimization import CollidingBodiesOptimization
from opt.metaheuristic.cross_entropy_method import CrossEntropyMethod
from opt.metaheuristic.eagle_strategy import EagleStrategy
from opt.metaheuristic.harmony_search import HarmonySearch
from opt.metaheuristic.particle_filter import ParticleFilter
from opt.metaheuristic.shuffled_frog_leaping_algorithm import ShuffledFrogLeapingAlgorithm
from opt.metaheuristic.sine_cosine_algorithm import SineCosineAlgorithm
from opt.metaheuristic.stochastic_diffusion_search import StochasticDiffusionSearch
from opt.metaheuristic.stochastic_fractal_search import StochasticFractalSearch
from opt.metaheuristic.variable_depth_search import VariableDepthSearch
from opt.metaheuristic.variable_neighbourhood_search import VariableNeighborhoodSearch
from opt.metaheuristic.very_large_scale_neighborhood_search import (
    VeryLargeScaleNeighborhood,
)

__all__: list[str] = [
    "CollidingBodiesOptimization",
    "CrossEntropyMethod",
    "EagleStrategy",
    "HarmonySearch",
    "ParticleFilter",
    "ShuffledFrogLeapingAlgorithm",
    "SineCosineAlgorithm",
    "StochasticDiffusionSearch",
    "StochasticFractalSearch",
    "VariableDepthSearch",
    "VariableNeighborhoodSearch",
    "VeryLargeScaleNeighborhood",
]
