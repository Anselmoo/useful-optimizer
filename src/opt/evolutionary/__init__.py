"""Evolutionary optimization algorithms.

This module contains population-based metaheuristic optimizers inspired by biological
evolution. Includes: CMA-ES, Cultural Algorithm, Differential Evolution, Estimation of
Distribution Algorithm, Genetic Algorithm, and Imperialist Competitive Algorithm.
"""

from __future__ import annotations

from opt.evolutionary.cma_es import CMAESAlgorithm
from opt.evolutionary.cultural_algorithm import CulturalAlgorithm
from opt.evolutionary.differential_evolution import DifferentialEvolution
from opt.evolutionary.estimation_of_distribution_algorithm import (
    EstimationOfDistributionAlgorithm,
)
from opt.evolutionary.genetic_algorithm import GeneticAlgorithm
from opt.evolutionary.imperialist_competitive_algorithm import (
    ImperialistCompetitiveAlgorithm,
)


__all__: list[str] = [
    "CMAESAlgorithm",
    "CulturalAlgorithm",
    "DifferentialEvolution",
    "EstimationOfDistributionAlgorithm",
    "GeneticAlgorithm",
    "ImperialistCompetitiveAlgorithm",
]
