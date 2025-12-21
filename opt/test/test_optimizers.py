"""Tests for all optimizer classes."""

from __future__ import annotations

import numpy as np
import pytest

from opt import BFGS
from opt import LBFGS

# New multi-objective algorithms
from opt import MOEAD
from opt import NSGAII
from opt import SGD
from opt import SPEA2
from opt import ADAGrad
from opt import ADAMOptimization
from opt import AMSGrad
from opt import AbstractOptimizer
from opt import AdaDelta
from opt import AdaMax
from opt import AdamW

# New swarm intelligence algorithms
from opt import AfricanBuffaloOptimizer
from opt import AfricanVulturesOptimizer
from opt import AntColony
from opt import AntLionOptimizer
from opt import AquilaOptimizer

# New metaheuristic algorithms
from opt import ArithmeticOptimizationAlgorithm
from opt import ArtificialFishSwarm
from opt import ArtificialGorillaTroopsOptimizer
from opt import ArtificialHummingbirdAlgorithm

# New physics-inspired algorithms
from opt import AtomSearchOptimizer
from opt import AugmentedLagrangian
from opt import BarnaclesMatingOptimizer
from opt import BatAlgorithm
from opt import BeeAlgorithm
from opt import BlackWidowOptimizer
from opt import BrownBearOptimizer
from opt import CMAESAlgorithm
from opt import CatSwarmOptimization
from opt import ChimpOptimizationAlgorithm
from opt import CoatiOptimizer
from opt import CollidingBodiesOptimization
from opt import ConjugateGradient
from opt import CrossEntropyMethod
from opt import CuckooSearch
from opt import CulturalAlgorithm
from opt import DifferentialEvolution
from opt import DingoOptimizer
from opt import DragonflyOptimizer
from opt import EagleStrategy
from opt import EmperorPenguinOptimizer
from opt import EquilibriumOptimizer
from opt import EstimationOfDistributionAlgorithm
from opt import FireflyAlgorithm
from opt import FlowerPollinationAlgorithm
from opt import ForensicBasedInvestigationOptimizer
from opt import GeneticAlgorithm
from opt import GlowwormSwarmOptimization
from opt import GoldenEagleOptimizer
from opt import GrasshopperOptimizer
from opt import GravitationalSearchOptimizer
from opt import GreyWolfOptimizer
from opt import HarmonySearch
from opt import HarrisHawksOptimizer
from opt import HillClimbing
from opt import HoneyBadgerAlgorithm
from opt import ImperialistCompetitiveAlgorithm
from opt import LDAnalysis
from opt import MantaRayForagingOptimization
from opt import MarinePredatorsOptimizer
from opt import MayflyOptimizer
from opt import MothFlameOptimizer
from opt import MothSearchAlgorithm
from opt import MountainGazelleOptimizer
from opt import Nadam
from opt import NelderMead
from opt import NesterovAcceleratedGradient
from opt import OrcaPredatorAlgorithm
from opt import ParticleFilter
from opt import ParticleSwarm
from opt import ParzenTreeEstimator
from opt import PathfinderAlgorithm
from opt import Powell
from opt import RMSprop
from opt import ReptileSearchAlgorithm
from opt import SGDMomentum
from opt import SalpSwarmOptimizer
from opt import SandCatSwarmOptimizer
from opt import SeagullOptimizationAlgorithm
from opt import ShuffledFrogLeapingAlgorithm
from opt import SimulatedAnnealing
from opt import SineCosineAlgorithm
from opt import SlimeMouldAlgorithm
from opt import SpottedHyenaOptimizer
from opt import SquirrelSearchAlgorithm
from opt import StochasticDiffusionSearch
from opt import StochasticFractalSearch
from opt import SuccessiveLinearProgramming
from opt import TabuSearch

# New social-inspired algorithms
from opt import TeachingLearningOptimizer
from opt import TrustRegion
from opt import TunicateSwarmAlgorithm
from opt import VariableDepthSearch
from opt import VariableNeighborhoodSearch
from opt import VeryLargeScaleNeighborhood
from opt import WhaleOptimizationAlgorithm
from opt import WildHorseOptimizer
from opt.benchmark.functions import shifted_ackley
from opt.benchmark.functions import sphere


# List of all optimizer classes for parametrized testing
SWARM_OPTIMIZERS = [
    AfricanBuffaloOptimizer,
    AfricanVulturesOptimizer,
    AntColony,
    AntLionOptimizer,
    AquilaOptimizer,
    ArtificialFishSwarm,
    ArtificialGorillaTroopsOptimizer,
    ArtificialHummingbirdAlgorithm,
    BarnaclesMatingOptimizer,
    # BatAlgorithm excluded - requires n_bats parameter (tested separately)
    BeeAlgorithm,
    BlackWidowOptimizer,
    BrownBearOptimizer,
    CatSwarmOptimization,
    ChimpOptimizationAlgorithm,
    CoatiOptimizer,
    CuckooSearch,
    DingoOptimizer,
    DragonflyOptimizer,
    EmperorPenguinOptimizer,
    FireflyAlgorithm,
    FlowerPollinationAlgorithm,
    GlowwormSwarmOptimization,
    GoldenEagleOptimizer,
    GrasshopperOptimizer,
    GreyWolfOptimizer,
    HarrisHawksOptimizer,
    HoneyBadgerAlgorithm,
    MantaRayForagingOptimization,
    MarinePredatorsOptimizer,
    MayflyOptimizer,
    MothFlameOptimizer,
    MothSearchAlgorithm,
    MountainGazelleOptimizer,
    OrcaPredatorAlgorithm,
    ParticleSwarm,
    PathfinderAlgorithm,
    ReptileSearchAlgorithm,
    SalpSwarmOptimizer,
    SandCatSwarmOptimizer,
    SeagullOptimizationAlgorithm,
    SlimeMouldAlgorithm,
    SpottedHyenaOptimizer,
    SquirrelSearchAlgorithm,
    TunicateSwarmAlgorithm,
    WhaleOptimizationAlgorithm,
    WildHorseOptimizer,
]

EVOLUTIONARY_OPTIMIZERS = [
    CMAESAlgorithm,
    CulturalAlgorithm,
    DifferentialEvolution,
    EstimationOfDistributionAlgorithm,
    GeneticAlgorithm,
    ImperialistCompetitiveAlgorithm,
]

GRADIENT_OPTIMIZERS = [
    AdaDelta,
    ADAGrad,
    AdaMax,
    AdamW,
    ADAMOptimization,
    AMSGrad,
    Nadam,
    NesterovAcceleratedGradient,
    RMSprop,
    SGD,
    SGDMomentum,
]

CLASSICAL_OPTIMIZERS = [
    BFGS,
    ConjugateGradient,
    HillClimbing,
    LBFGS,
    NelderMead,
    Powell,
    SimulatedAnnealing,
    TabuSearch,
    TrustRegion,
]

METAHEURISTIC_OPTIMIZERS = [
    ArithmeticOptimizationAlgorithm,
    CollidingBodiesOptimization,
    CrossEntropyMethod,
    EagleStrategy,
    ForensicBasedInvestigationOptimizer,
    HarmonySearch,
    ParticleFilter,
    ShuffledFrogLeapingAlgorithm,
    SineCosineAlgorithm,
    StochasticDiffusionSearch,
    StochasticFractalSearch,
    VariableDepthSearch,
    VariableNeighborhoodSearch,
    VeryLargeScaleNeighborhood,
]

PHYSICS_INSPIRED_OPTIMIZERS = [
    AtomSearchOptimizer,
    EquilibriumOptimizer,
    GravitationalSearchOptimizer,
]

SOCIAL_INSPIRED_OPTIMIZERS = [TeachingLearningOptimizer]

CONSTRAINED_OPTIMIZERS = [AugmentedLagrangian, SuccessiveLinearProgramming]

PROBABILISTIC_OPTIMIZERS = [LDAnalysis, ParzenTreeEstimator]

ALL_OPTIMIZERS = (
    SWARM_OPTIMIZERS
    + EVOLUTIONARY_OPTIMIZERS
    + GRADIENT_OPTIMIZERS
    + CLASSICAL_OPTIMIZERS
    + METAHEURISTIC_OPTIMIZERS
    + PHYSICS_INSPIRED_OPTIMIZERS
    + SOCIAL_INSPIRED_OPTIMIZERS
    + CONSTRAINED_OPTIMIZERS
    + PROBABILISTIC_OPTIMIZERS
)


class TestAbstractOptimizer:
    """Tests for the AbstractOptimizer base class."""

    def test_abstract_optimizer_cannot_be_instantiated(self) -> None:
        """Test that AbstractOptimizer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractOptimizer(  # type: ignore[abstract]
                func=sphere, lower_bound=-5, upper_bound=5, dim=2
            )

    def test_all_optimizers_inherit_from_abstract(self) -> None:
        """Test that all optimizers inherit from AbstractOptimizer."""
        for optimizer_class in ALL_OPTIMIZERS:
            assert issubclass(optimizer_class, AbstractOptimizer)


class TestOptimizerInstantiation:
    """Tests for optimizer instantiation."""

    @pytest.mark.parametrize("optimizer_class", ALL_OPTIMIZERS)
    def test_optimizer_instantiation(
        self, optimizer_class: type[AbstractOptimizer]
    ) -> None:
        """Test that all optimizers can be instantiated."""
        optimizer = optimizer_class(
            func=sphere, lower_bound=-5, upper_bound=5, dim=2, max_iter=10
        )
        assert optimizer is not None
        assert optimizer.func == sphere
        assert optimizer.lower_bound == -5
        assert optimizer.upper_bound == 5
        assert optimizer.dim == 2
        assert optimizer.max_iter == 10


class TestOptimizerSearch:
    """Tests for optimizer search functionality."""

    @pytest.mark.parametrize("optimizer_class", SWARM_OPTIMIZERS)
    def test_swarm_optimizer_search(
        self, optimizer_class: type[AbstractOptimizer]
    ) -> None:
        """Test that swarm optimizers can perform search."""
        optimizer = optimizer_class(
            func=shifted_ackley,
            lower_bound=-2.768,
            upper_bound=2.768,
            dim=2,
            max_iter=20,
        )
        solution, fitness = optimizer.search()
        assert isinstance(solution, np.ndarray)
        assert isinstance(fitness, float)
        assert solution.shape == (2,)

    @pytest.mark.parametrize("optimizer_class", EVOLUTIONARY_OPTIMIZERS)
    def test_evolutionary_optimizer_search(
        self, optimizer_class: type[AbstractOptimizer]
    ) -> None:
        """Test that evolutionary optimizers can perform search."""
        optimizer = optimizer_class(
            func=shifted_ackley,
            lower_bound=-2.768,
            upper_bound=2.768,
            dim=2,
            max_iter=20,
        )
        solution, fitness = optimizer.search()
        assert isinstance(solution, np.ndarray)
        assert isinstance(fitness, float)
        assert solution.shape == (2,)

    @pytest.mark.parametrize("optimizer_class", GRADIENT_OPTIMIZERS)
    def test_gradient_optimizer_search(
        self, optimizer_class: type[AbstractOptimizer]
    ) -> None:
        """Test that gradient optimizers can perform search."""
        optimizer = optimizer_class(
            func=shifted_ackley,
            lower_bound=-2.768,
            upper_bound=2.768,
            dim=2,
            max_iter=20,
        )
        solution, fitness = optimizer.search()
        assert isinstance(solution, np.ndarray)
        assert isinstance(fitness, float)
        assert solution.shape == (2,)

    @pytest.mark.parametrize("optimizer_class", CLASSICAL_OPTIMIZERS)
    def test_classical_optimizer_search(
        self, optimizer_class: type[AbstractOptimizer]
    ) -> None:
        """Test that classical optimizers can perform search."""
        optimizer = optimizer_class(
            func=shifted_ackley,
            lower_bound=-2.768,
            upper_bound=2.768,
            dim=2,
            max_iter=20,
        )
        solution, fitness = optimizer.search()
        assert isinstance(solution, np.ndarray)
        assert isinstance(fitness, float)
        assert solution.shape == (2,)

    @pytest.mark.parametrize("optimizer_class", METAHEURISTIC_OPTIMIZERS)
    def test_metaheuristic_optimizer_search(
        self, optimizer_class: type[AbstractOptimizer]
    ) -> None:
        """Test that metaheuristic optimizers can perform search."""
        optimizer = optimizer_class(
            func=shifted_ackley,
            lower_bound=-2.768,
            upper_bound=2.768,
            dim=2,
            max_iter=20,
        )
        solution, fitness = optimizer.search()
        assert isinstance(solution, np.ndarray)
        assert isinstance(fitness, float)
        assert solution.shape == (2,)

    @pytest.mark.parametrize("optimizer_class", CONSTRAINED_OPTIMIZERS)
    def test_constrained_optimizer_search(
        self, optimizer_class: type[AbstractOptimizer]
    ) -> None:
        """Test that constrained optimizers can perform search."""
        optimizer = optimizer_class(
            func=shifted_ackley,
            lower_bound=-2.768,
            upper_bound=2.768,
            dim=2,
            max_iter=20,
        )
        solution, fitness = optimizer.search()
        assert isinstance(solution, np.ndarray)
        assert isinstance(fitness, float)
        assert solution.shape == (2,)

    @pytest.mark.parametrize("optimizer_class", PHYSICS_INSPIRED_OPTIMIZERS)
    def test_physics_inspired_optimizer_search(
        self, optimizer_class: type[AbstractOptimizer]
    ) -> None:
        """Test that physics-inspired optimizers can perform search."""
        optimizer = optimizer_class(
            func=shifted_ackley,
            lower_bound=-2.768,
            upper_bound=2.768,
            dim=2,
            max_iter=20,
        )
        solution, fitness = optimizer.search()
        assert isinstance(solution, np.ndarray)
        assert isinstance(fitness, float)
        assert solution.shape == (2,)

    @pytest.mark.parametrize("optimizer_class", SOCIAL_INSPIRED_OPTIMIZERS)
    def test_social_inspired_optimizer_search(
        self, optimizer_class: type[AbstractOptimizer]
    ) -> None:
        """Test that social-inspired optimizers can perform search."""
        optimizer = optimizer_class(
            func=shifted_ackley,
            lower_bound=-2.768,
            upper_bound=2.768,
            dim=2,
            max_iter=20,
        )
        solution, fitness = optimizer.search()
        assert isinstance(solution, np.ndarray)
        assert isinstance(fitness, float)
        assert solution.shape == (2,)

    @pytest.mark.parametrize("optimizer_class", PROBABILISTIC_OPTIMIZERS)
    def test_probabilistic_optimizer_search(
        self, optimizer_class: type[AbstractOptimizer]
    ) -> None:
        """Test that probabilistic optimizers can perform search."""
        optimizer = optimizer_class(
            func=shifted_ackley,
            lower_bound=-2.768,
            upper_bound=2.768,
            dim=2,
            max_iter=20,
        )
        solution, fitness = optimizer.search()
        assert isinstance(solution, np.ndarray)
        assert isinstance(fitness, float)
        assert solution.shape == (2,)


class TestSpecialOptimizers:
    """Tests for optimizers with special parameter requirements."""

    def test_bat_algorithm_with_n_bats(self) -> None:
        """Test BatAlgorithm with required n_bats parameter."""
        optimizer = BatAlgorithm(
            func=shifted_ackley,
            lower_bound=-2.768,
            upper_bound=2.768,
            dim=2,
            n_bats=10,
            max_iter=20,
        )
        solution, fitness = optimizer.search()
        assert isinstance(solution, np.ndarray)
        assert isinstance(fitness, float)
        assert solution.shape == (2,)

    def test_bat_algorithm_instantiation(self) -> None:
        """Test that BatAlgorithm can be instantiated with n_bats parameter."""
        optimizer = BatAlgorithm(
            func=sphere, lower_bound=-5, upper_bound=5, dim=2, n_bats=5, max_iter=10
        )
        assert optimizer is not None
        assert optimizer.func == sphere
        assert optimizer.lower_bound == -5
        assert optimizer.upper_bound == 5
        assert optimizer.dim == 2
        assert optimizer.max_iter == 10


MULTI_OBJECTIVE_OPTIMIZERS = [MOEAD, NSGAII, SPEA2]


class TestMultiObjectiveOptimizers:
    """Tests for multi-objective optimization algorithms."""

    @staticmethod
    def objective_f1(x: np.ndarray) -> float:
        """First objective function (sphere)."""
        return float(np.sum(x**2))

    @staticmethod
    def objective_f2(x: np.ndarray) -> float:
        """Second objective function (shifted sphere)."""
        return float(np.sum((x - 2) ** 2))

    @pytest.mark.parametrize("optimizer_class", MULTI_OBJECTIVE_OPTIMIZERS)
    def test_multi_objective_optimizer_instantiation(
        self, optimizer_class: type
    ) -> None:
        """Test that multi-objective optimizers can be instantiated."""
        optimizer = optimizer_class(
            objectives=[self.objective_f1, self.objective_f2],
            lower_bound=-5,
            upper_bound=5,
            dim=2,
            population_size=10,
            max_iter=5,
        )
        assert optimizer is not None

    @pytest.mark.parametrize("optimizer_class", MULTI_OBJECTIVE_OPTIMIZERS)
    def test_multi_objective_optimizer_search(self, optimizer_class: type) -> None:
        """Test that multi-objective optimizers return Pareto-optimal solutions."""
        optimizer = optimizer_class(
            objectives=[self.objective_f1, self.objective_f2],
            lower_bound=-5,
            upper_bound=5,
            dim=2,
            population_size=10,
            max_iter=5,
        )
        solutions, fitness = optimizer.search()
        # Multi-objective returns list of solutions and fitness array
        assert isinstance(solutions, (list, np.ndarray))
        assert isinstance(fitness, np.ndarray)
        assert len(solutions) > 0
        # Fitness should be 2D: (num_solutions, num_objectives)
        assert fitness.ndim == 2


class TestBenchmarkFunctions:
    """Tests for benchmark functions."""

    def test_sphere_function(self) -> None:
        """Test sphere function at known points."""
        assert sphere(np.array([0, 0])) == 0
        assert sphere(np.array([1, 0])) == 1
        assert sphere(np.array([1, 1])) == 2

    def test_shifted_ackley_function(self) -> None:
        """Test shifted_ackley function returns float."""
        result = shifted_ackley(np.array([0, 0]))
        assert isinstance(result, float)


class TestCategoricalImports:
    """Tests for categorical module imports."""

    def test_gradient_based_import(self) -> None:
        """Test importing from gradient_based submodule."""
        from opt.gradient_based import AdamW
        from opt.gradient_based import SGDMomentum

        assert AdamW is not None
        assert SGDMomentum is not None

    def test_swarm_intelligence_import(self) -> None:
        """Test importing from swarm_intelligence submodule."""
        from opt.swarm_intelligence import AntColony
        from opt.swarm_intelligence import ParticleSwarm

        assert ParticleSwarm is not None
        assert AntColony is not None

    def test_evolutionary_import(self) -> None:
        """Test importing from evolutionary submodule."""
        from opt.evolutionary import DifferentialEvolution
        from opt.evolutionary import GeneticAlgorithm

        assert GeneticAlgorithm is not None
        assert DifferentialEvolution is not None

    def test_classical_import(self) -> None:
        """Test importing from classical submodule."""
        from opt.classical import BFGS
        from opt.classical import NelderMead

        assert BFGS is not None
        assert NelderMead is not None

    def test_metaheuristic_import(self) -> None:
        """Test importing from metaheuristic submodule."""
        from opt.metaheuristic import CrossEntropyMethod
        from opt.metaheuristic import HarmonySearch

        assert HarmonySearch is not None
        assert CrossEntropyMethod is not None

    def test_constrained_import(self) -> None:
        """Test importing from constrained submodule."""
        from opt.constrained import AugmentedLagrangian

        assert AugmentedLagrangian is not None

    def test_probabilistic_import(self) -> None:
        """Test importing from probabilistic submodule."""
        from opt.probabilistic import ParzenTreeEstimator

        assert ParzenTreeEstimator is not None

    def test_multi_objective_import(self) -> None:
        """Test importing from multi_objective submodule."""
        from opt.multi_objective import MOEAD
        from opt.multi_objective import NSGAII
        from opt.multi_objective import SPEA2

        assert NSGAII is not None
        assert MOEAD is not None
        assert SPEA2 is not None

    def test_physics_inspired_import(self) -> None:
        """Test importing from physics_inspired submodule."""
        from opt.physics_inspired import AtomSearchOptimizer
        from opt.physics_inspired import EquilibriumOptimizer
        from opt.physics_inspired import GravitationalSearchOptimizer

        assert AtomSearchOptimizer is not None
        assert EquilibriumOptimizer is not None
        assert GravitationalSearchOptimizer is not None

    def test_social_inspired_import(self) -> None:
        """Test importing from social_inspired submodule."""
        from opt.social_inspired import TeachingLearningOptimizer

        assert TeachingLearningOptimizer is not None

    def test_backward_compatible_import(self) -> None:
        """Test backward compatible imports from root opt module."""
        from opt import BFGS
        from opt import AdamW
        from opt import HarmonySearch
        from opt import ParticleSwarm

        assert ParticleSwarm is not None
        assert AdamW is not None
        assert BFGS is not None
        assert HarmonySearch is not None
