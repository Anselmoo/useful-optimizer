"""Useful optimizers, a set of optimization algorithms.

This package provides 64+ optimization algorithms organized into categories:
- gradient_based: Gradient-based optimizers (AdaDelta, AdaGrad, Adam, etc.)
- swarm_intelligence: Nature-inspired swarm algorithms (PSO, ACO, HHO, MPA, etc.)
- evolutionary: Evolutionary algorithms (GA, DE, CMA-ES, etc.)
- classical: Classical optimization methods (BFGS, Nelder-Mead, etc.)
- metaheuristic: Metaheuristic algorithms (Harmony Search, etc.)
- constrained: Constrained optimization methods
- probabilistic: Probabilistic optimization methods
- multi_objective: Multi-objective optimization (NSGA-II, etc.)
- physics_inspired: Physics-inspired algorithms (GSA, EO, etc.)

All optimizers are re-exported at the package level for backward compatibility.
"""

from __future__ import annotations

# Base class
from opt.abstract_optimizer import AbstractOptimizer

# Classical algorithms
from opt.classical import BFGS
from opt.classical import LBFGS
from opt.classical import ConjugateGradient
from opt.classical import HillClimbing
from opt.classical import NelderMead
from opt.classical import Powell
from opt.classical import SimulatedAnnealing
from opt.classical import TabuSearch
from opt.classical import TrustRegion

# Constrained optimization
from opt.constrained import AugmentedLagrangian
from opt.constrained import SuccessiveLinearProgramming

# Evolutionary algorithms
from opt.evolutionary import CMAESAlgorithm
from opt.evolutionary import CulturalAlgorithm
from opt.evolutionary import DifferentialEvolution
from opt.evolutionary import EstimationOfDistributionAlgorithm
from opt.evolutionary import GeneticAlgorithm
from opt.evolutionary import ImperialistCompetitiveAlgorithm
from opt.gradient_based import SGD

# Gradient-based algorithms
from opt.gradient_based import ADAGrad
from opt.gradient_based import ADAMOptimization
from opt.gradient_based import AMSGrad
from opt.gradient_based import AdaDelta
from opt.gradient_based import AdaMax
from opt.gradient_based import AdamW
from opt.gradient_based import Nadam
from opt.gradient_based import NesterovAcceleratedGradient
from opt.gradient_based import RMSprop
from opt.gradient_based import SGDMomentum

# Metaheuristic algorithms
from opt.metaheuristic import CollidingBodiesOptimization
from opt.metaheuristic import CrossEntropyMethod
from opt.metaheuristic import EagleStrategy
from opt.metaheuristic import HarmonySearch
from opt.metaheuristic import ParticleFilter
from opt.metaheuristic import ShuffledFrogLeapingAlgorithm
from opt.metaheuristic import SineCosineAlgorithm
from opt.metaheuristic import StochasticDiffusionSearch
from opt.metaheuristic import StochasticFractalSearch
from opt.metaheuristic import VariableDepthSearch
from opt.metaheuristic import VariableNeighborhoodSearch
from opt.metaheuristic import VeryLargeScaleNeighborhood

# Multi-objective algorithms (additional)
from opt.multi_objective import MOEAD
from opt.multi_objective import NSGAII

# Multi-objective algorithms
from opt.multi_objective import AbstractMultiObjectiveOptimizer

# Physics-inspired algorithms (additional)
from opt.physics_inspired import AtomSearchOptimizer

# Physics-inspired algorithms
from opt.physics_inspired import EquilibriumOptimizer
from opt.physics_inspired import GravitationalSearchOptimizer

# Probabilistic algorithms
from opt.probabilistic import LDAnalysis
from opt.probabilistic import ParzenTreeEstimator

# Social-inspired algorithms
from opt.social_inspired import TeachingLearningOptimizer
from opt.swarm_intelligence import AfricanVulturesOptimizer

# Swarm intelligence algorithms
from opt.swarm_intelligence import AntColony
from opt.swarm_intelligence import AntLionOptimizer
from opt.swarm_intelligence import AquilaOptimizer
from opt.swarm_intelligence import ArtificialFishSwarm
from opt.swarm_intelligence import ArtificialGorillaTroopsOptimizer
from opt.swarm_intelligence import BatAlgorithm
from opt.swarm_intelligence import BeeAlgorithm
from opt.swarm_intelligence import CatSwarmOptimization
from opt.swarm_intelligence import CuckooSearch
from opt.swarm_intelligence import DragonflyOptimizer
from opt.swarm_intelligence import FireflyAlgorithm
from opt.swarm_intelligence import GlowwormSwarmOptimization
from opt.swarm_intelligence import GrasshopperOptimizer
from opt.swarm_intelligence import GreyWolfOptimizer
from opt.swarm_intelligence import HarrisHawksOptimizer
from opt.swarm_intelligence import MarinePredatorsOptimizer
from opt.swarm_intelligence import MothFlameOptimizer
from opt.swarm_intelligence import ParticleSwarm
from opt.swarm_intelligence import SalpSwarmOptimizer
from opt.swarm_intelligence import SquirrelSearchAlgorithm
from opt.swarm_intelligence import WhaleOptimizationAlgorithm


__version__ = "0.1.2"

__all__: list[str] = [
    # Classical
    "BFGS",
    "LBFGS",
    # Multi-objective
    "MOEAD",
    "NSGAII",
    # Gradient-based
    "SGD",
    "ADAGrad",
    "ADAMOptimization",
    "AMSGrad",
    "AbstractMultiObjectiveOptimizer",
    # Base class
    "AbstractOptimizer",
    "AdaDelta",
    "AdaMax",
    "AdamW",
    # Swarm intelligence
    "AfricanVulturesOptimizer",
    "AntColony",
    "AntLionOptimizer",
    "AquilaOptimizer",
    "ArtificialFishSwarm",
    "ArtificialGorillaTroopsOptimizer",
    # Physics-inspired
    "AtomSearchOptimizer",
    # Constrained
    "AugmentedLagrangian",
    "BatAlgorithm",
    "BeeAlgorithm",
    # Evolutionary
    "CMAESAlgorithm",
    "CatSwarmOptimization",
    # Metaheuristic
    "CollidingBodiesOptimization",
    "ConjugateGradient",
    "CrossEntropyMethod",
    "CuckooSearch",
    "CulturalAlgorithm",
    "DifferentialEvolution",
    "DragonflyOptimizer",
    "EagleStrategy",
    "EquilibriumOptimizer",
    "EstimationOfDistributionAlgorithm",
    "FireflyAlgorithm",
    "GeneticAlgorithm",
    "GlowwormSwarmOptimization",
    "GrasshopperOptimizer",
    "GravitationalSearchOptimizer",
    "GreyWolfOptimizer",
    "HarmonySearch",
    "HarrisHawksOptimizer",
    "HillClimbing",
    "ImperialistCompetitiveAlgorithm",
    # Probabilistic
    "LDAnalysis",
    "MarinePredatorsOptimizer",
    "MothFlameOptimizer",
    "Nadam",
    "NelderMead",
    "NesterovAcceleratedGradient",
    "ParticleFilter",
    "ParticleSwarm",
    "ParzenTreeEstimator",
    "Powell",
    "RMSprop",
    "SGDMomentum",
    "SalpSwarmOptimizer",
    "ShuffledFrogLeapingAlgorithm",
    "SimulatedAnnealing",
    "SineCosineAlgorithm",
    "SquirrelSearchAlgorithm",
    "StochasticDiffusionSearch",
    "StochasticFractalSearch",
    "SuccessiveLinearProgramming",
    "TabuSearch",
    # Social-inspired
    "TeachingLearningOptimizer",
    "TrustRegion",
    "VariableDepthSearch",
    "VariableNeighborhoodSearch",
    "VeryLargeScaleNeighborhood",
    "WhaleOptimizationAlgorithm",
]
