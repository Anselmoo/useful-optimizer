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
from opt.constrained import BarrierMethodOptimizer
from opt.constrained import PenaltyMethodOptimizer
from opt.constrained import SequentialQuadraticProgramming
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
from opt.metaheuristic import ArithmeticOptimizationAlgorithm
from opt.metaheuristic import CollidingBodiesOptimization
from opt.metaheuristic import CrossEntropyMethod
from opt.metaheuristic import EagleStrategy
from opt.metaheuristic import ForensicBasedInvestigationOptimizer
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
from opt.multi_objective import SPEA2

# Multi-objective algorithms
from opt.multi_objective import AbstractMultiObjectiveOptimizer

# Physics-inspired algorithms (additional)
from opt.physics_inspired import AtomSearchOptimizer

# Physics-inspired algorithms
from opt.physics_inspired import EquilibriumOptimizer
from opt.physics_inspired import GravitationalSearchOptimizer
from opt.physics_inspired import RIMEOptimizer

# Probabilistic algorithms
from opt.probabilistic import AdaptiveMetropolisOptimizer
from opt.probabilistic import BayesianOptimizer
from opt.probabilistic import LDAnalysis
from opt.probabilistic import ParzenTreeEstimator
from opt.probabilistic import SequentialMonteCarloOptimizer

# Social-inspired algorithms
from opt.social_inspired import PoliticalOptimizer
from opt.social_inspired import SoccerLeagueOptimizer
from opt.social_inspired import SocialGroupOptimizer
from opt.social_inspired import TeachingLearningOptimizer
from opt.swarm_intelligence import AfricanBuffaloOptimizer
from opt.swarm_intelligence import AfricanVulturesOptimizer

# Swarm intelligence algorithms
from opt.swarm_intelligence import AntColony
from opt.swarm_intelligence import AntLionOptimizer
from opt.swarm_intelligence import AquilaOptimizer
from opt.swarm_intelligence import ArtificialFishSwarm
from opt.swarm_intelligence import ArtificialGorillaTroopsOptimizer
from opt.swarm_intelligence import ArtificialHummingbirdAlgorithm
from opt.swarm_intelligence import ArtificialRabbitsOptimizer
from opt.swarm_intelligence import BarnaclesMatingOptimizer
from opt.swarm_intelligence import BatAlgorithm
from opt.swarm_intelligence import BeeAlgorithm
from opt.swarm_intelligence import BlackWidowOptimizer
from opt.swarm_intelligence import BrownBearOptimizer
from opt.swarm_intelligence import CatSwarmOptimization
from opt.swarm_intelligence import ChimpOptimizationAlgorithm
from opt.swarm_intelligence import CoatiOptimizer
from opt.swarm_intelligence import CuckooSearch
from opt.swarm_intelligence import DandelionOptimizer
from opt.swarm_intelligence import DingoOptimizer
from opt.swarm_intelligence import DragonflyOptimizer
from opt.swarm_intelligence import EmperorPenguinOptimizer
from opt.swarm_intelligence import FennecFoxOptimizer
from opt.swarm_intelligence import FireflyAlgorithm
from opt.swarm_intelligence import FlowerPollinationAlgorithm
from opt.swarm_intelligence import GiantTrevallyOptimizer
from opt.swarm_intelligence import GlowwormSwarmOptimization
from opt.swarm_intelligence import GoldenEagleOptimizer
from opt.swarm_intelligence import GrasshopperOptimizer
from opt.swarm_intelligence import GreyWolfOptimizer
from opt.swarm_intelligence import HarrisHawksOptimizer
from opt.swarm_intelligence import HoneyBadgerAlgorithm
from opt.swarm_intelligence import MantaRayForagingOptimization
from opt.swarm_intelligence import MarinePredatorsOptimizer
from opt.swarm_intelligence import MayflyOptimizer
from opt.swarm_intelligence import MothFlameOptimizer
from opt.swarm_intelligence import MothSearchAlgorithm
from opt.swarm_intelligence import MountainGazelleOptimizer
from opt.swarm_intelligence import OrcaPredatorAlgorithm
from opt.swarm_intelligence import OspreyOptimizer
from opt.swarm_intelligence import ParticleSwarm
from opt.swarm_intelligence import PathfinderAlgorithm
from opt.swarm_intelligence import PelicanOptimizer
from opt.swarm_intelligence import ReptileSearchAlgorithm
from opt.swarm_intelligence import SalpSwarmOptimizer
from opt.swarm_intelligence import SandCatSwarmOptimizer
from opt.swarm_intelligence import SeagullOptimizationAlgorithm
from opt.swarm_intelligence import SlimeMouldAlgorithm
from opt.swarm_intelligence import SnowGeeseOptimizer
from opt.swarm_intelligence import SpottedHyenaOptimizer
from opt.swarm_intelligence import SquirrelSearchAlgorithm
from opt.swarm_intelligence import StarlingMurmurationOptimizer
from opt.swarm_intelligence import TunicateSwarmAlgorithm
from opt.swarm_intelligence import WhaleOptimizationAlgorithm
from opt.swarm_intelligence import WildHorseOptimizer
from opt.swarm_intelligence import ZebraOptimizer


__version__ = "0.1.2"

__all__: list[str] = [
    "BFGS",
    "LBFGS",
    "MOEAD",
    "NSGAII",
    "SGD",
    "SPEA2",
    "ADAGrad",
    "ADAMOptimization",
    "AMSGrad",
    "AbstractMultiObjectiveOptimizer",
    "AbstractOptimizer",
    "AdaDelta",
    "AdaMax",
    "AdamW",
    "AdaptiveMetropolisOptimizer",
    "AfricanBuffaloOptimizer",
    "AfricanVulturesOptimizer",
    "AntColony",
    "AntLionOptimizer",
    "AquilaOptimizer",
    "ArithmeticOptimizationAlgorithm",
    "ArtificialFishSwarm",
    "ArtificialGorillaTroopsOptimizer",
    "ArtificialHummingbirdAlgorithm",
    "ArtificialRabbitsOptimizer",
    "AtomSearchOptimizer",
    "AugmentedLagrangian",
    "BarnaclesMatingOptimizer",
    "BarrierMethodOptimizer",
    "BatAlgorithm",
    "BayesianOptimizer",
    "BeeAlgorithm",
    "BlackWidowOptimizer",
    "BrownBearOptimizer",
    "CMAESAlgorithm",
    "CatSwarmOptimization",
    "ChimpOptimizationAlgorithm",
    "CoatiOptimizer",
    "CollidingBodiesOptimization",
    "ConjugateGradient",
    "CrossEntropyMethod",
    "CuckooSearch",
    "CulturalAlgorithm",
    "DandelionOptimizer",
    "DifferentialEvolution",
    "DingoOptimizer",
    "DragonflyOptimizer",
    "EagleStrategy",
    "EmperorPenguinOptimizer",
    "EquilibriumOptimizer",
    "EstimationOfDistributionAlgorithm",
    "FennecFoxOptimizer",
    "FireflyAlgorithm",
    "FlowerPollinationAlgorithm",
    "ForensicBasedInvestigationOptimizer",
    "GeneticAlgorithm",
    "GiantTrevallyOptimizer",
    "GlowwormSwarmOptimization",
    "GoldenEagleOptimizer",
    "GrasshopperOptimizer",
    "GravitationalSearchOptimizer",
    "GreyWolfOptimizer",
    "HarmonySearch",
    "HarrisHawksOptimizer",
    "HillClimbing",
    "HoneyBadgerAlgorithm",
    "ImperialistCompetitiveAlgorithm",
    "LDAnalysis",
    "MantaRayForagingOptimization",
    "MarinePredatorsOptimizer",
    "MayflyOptimizer",
    "MothFlameOptimizer",
    "MothSearchAlgorithm",
    "MountainGazelleOptimizer",
    "Nadam",
    "NelderMead",
    "NesterovAcceleratedGradient",
    "OrcaPredatorAlgorithm",
    "OspreyOptimizer",
    "ParticleFilter",
    "ParticleSwarm",
    "ParzenTreeEstimator",
    "PathfinderAlgorithm",
    "PelicanOptimizer",
    "PenaltyMethodOptimizer",
    "PoliticalOptimizer",
    "Powell",
    "RIMEOptimizer",
    "RMSprop",
    "ReptileSearchAlgorithm",
    "SGDMomentum",
    "SalpSwarmOptimizer",
    "SandCatSwarmOptimizer",
    "SeagullOptimizationAlgorithm",
    "SequentialMonteCarloOptimizer",
    "SequentialQuadraticProgramming",
    "ShuffledFrogLeapingAlgorithm",
    "SimulatedAnnealing",
    "SineCosineAlgorithm",
    "SlimeMouldAlgorithm",
    "SnowGeeseOptimizer",
    "SoccerLeagueOptimizer",
    "SocialGroupOptimizer",
    "SpottedHyenaOptimizer",
    "SquirrelSearchAlgorithm",
    "StarlingMurmurationOptimizer",
    "StochasticDiffusionSearch",
    "StochasticFractalSearch",
    "SuccessiveLinearProgramming",
    "TabuSearch",
    "TeachingLearningOptimizer",
    "TrustRegion",
    "TunicateSwarmAlgorithm",
    "VariableDepthSearch",
    "VariableNeighborhoodSearch",
    "VeryLargeScaleNeighborhood",
    "WhaleOptimizationAlgorithm",
    "WildHorseOptimizer",
    "ZebraOptimizer",
]
