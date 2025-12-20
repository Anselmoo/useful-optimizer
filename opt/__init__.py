"""Useful optimizers, a set of optimization algorithms.

This package provides 54 optimization algorithms organized into categories:
- gradient_based: Gradient-based optimizers (AdaDelta, AdaGrad, Adam, etc.)
- swarm_intelligence: Nature-inspired swarm algorithms (PSO, ACO, etc.)
- evolutionary: Evolutionary algorithms (GA, DE, CMA-ES, etc.)
- classical: Classical optimization methods (BFGS, Nelder-Mead, etc.)
- metaheuristic: Metaheuristic algorithms (Harmony Search, etc.)
- constrained: Constrained optimization methods
- probabilistic: Probabilistic optimization methods

All optimizers are re-exported at the package level for backward compatibility.
"""

from __future__ import annotations

# Base class
from opt.abstract_optimizer import AbstractOptimizer

# Classical algorithms
from opt.classical import (
    BFGS,
    LBFGS,
    ConjugateGradient,
    HillClimbing,
    NelderMead,
    Powell,
    SimulatedAnnealing,
    TabuSearch,
    TrustRegion,
)

# Constrained optimization
from opt.constrained import (
    AugmentedLagrangian,
    SuccessiveLinearProgramming,
)

# Evolutionary algorithms
from opt.evolutionary import (
    CMAESAlgorithm,
    CulturalAlgorithm,
    DifferentialEvolution,
    EstimationOfDistributionAlgorithm,
    GeneticAlgorithm,
    ImperialistCompetitiveAlgorithm,
)

# Gradient-based algorithms
from opt.gradient_based import (
    ADAGrad,
    ADAMOptimization,
    AMSGrad,
    AdaDelta,
    AdaMax,
    AdamW,
    Nadam,
    NesterovAcceleratedGradient,
    RMSprop,
    SGD,
    SGDMomentum,
)

# Metaheuristic algorithms
from opt.metaheuristic import (
    CollidingBodiesOptimization,
    CrossEntropyMethod,
    EagleStrategy,
    HarmonySearch,
    ParticleFilter,
    ShuffledFrogLeapingAlgorithm,
    SineCosineAlgorithm,
    StochasticDiffusionSearch,
    StochasticFractalSearch,
    VariableDepthSearch,
    VariableNeighborhoodSearch,
    VeryLargeScaleNeighborhood,
)

# Probabilistic algorithms
from opt.probabilistic import (
    LDAnalysis,
    ParzenTreeEstimator,
)

# Swarm intelligence algorithms
from opt.swarm_intelligence import (
    AntColony,
    ArtificialFishSwarm,
    BatAlgorithm,
    BeeAlgorithm,
    CatSwarmOptimization,
    CuckooSearch,
    FireflyAlgorithm,
    GlowwormSwarmOptimization,
    GreyWolfOptimizer,
    ParticleSwarm,
    SquirrelSearchAlgorithm,
    WhaleOptimizationAlgorithm,
)

__version__ = "0.1.2"

__all__: list[str] = [
    # Base class
    "AbstractOptimizer",
    # Classical
    "BFGS",
    "ConjugateGradient",
    "HillClimbing",
    "LBFGS",
    "NelderMead",
    "Powell",
    "SimulatedAnnealing",
    "TabuSearch",
    "TrustRegion",
    # Constrained
    "AugmentedLagrangian",
    "SuccessiveLinearProgramming",
    # Evolutionary
    "CMAESAlgorithm",
    "CulturalAlgorithm",
    "DifferentialEvolution",
    "EstimationOfDistributionAlgorithm",
    "GeneticAlgorithm",
    "ImperialistCompetitiveAlgorithm",
    # Gradient-based
    "AdaDelta",
    "ADAGrad",
    "AdaMax",
    "AdamW",
    "ADAMOptimization",
    "AMSGrad",
    "Nadam",
    "NesterovAcceleratedGradient",
    "RMSprop",
    "SGD",
    "SGDMomentum",
    # Metaheuristic
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
    # Probabilistic
    "LDAnalysis",
    "ParzenTreeEstimator",
    # Swarm intelligence
    "AntColony",
    "ArtificialFishSwarm",
    "BatAlgorithm",
    "BeeAlgorithm",
    "CatSwarmOptimization",
    "CuckooSearch",
    "FireflyAlgorithm",
    "GlowwormSwarmOptimization",
    "GreyWolfOptimizer",
    "ParticleSwarm",
    "SquirrelSearchAlgorithm",
    "WhaleOptimizationAlgorithm",
]
