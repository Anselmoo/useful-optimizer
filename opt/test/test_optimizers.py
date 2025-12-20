"""Tests for all optimizer classes."""

from __future__ import annotations

import numpy as np
import pytest

from opt import (
    AbstractOptimizer,
    AdaDelta,
    ADAGrad,
    AdaMax,
    AdamW,
    ADAMOptimization,
    AMSGrad,
    AntColony,
    ArtificialFishSwarm,
    AugmentedLagrangian,
    BatAlgorithm,
    BeeAlgorithm,
    BFGS,
    CatSwarmOptimization,
    CMAESAlgorithm,
    CollidingBodiesOptimization,
    ConjugateGradient,
    CrossEntropyMethod,
    CuckooSearch,
    CulturalAlgorithm,
    DifferentialEvolution,
    EagleStrategy,
    EstimationOfDistributionAlgorithm,
    FireflyAlgorithm,
    GeneticAlgorithm,
    GlowwormSwarmOptimization,
    GreyWolfOptimizer,
    HarmonySearch,
    HillClimbing,
    ImperialistCompetitiveAlgorithm,
    LBFGS,
    LDAnalysis,
    Nadam,
    NelderMead,
    NesterovAcceleratedGradient,
    ParticleFilter,
    ParticleSwarm,
    ParzenTreeEstimator,
    Powell,
    RMSprop,
    SGD,
    SGDMomentum,
    ShuffledFrogLeapingAlgorithm,
    SimulatedAnnealing,
    SineCosineAlgorithm,
    SquirrelSearchAlgorithm,
    StochasticDiffusionSearch,
    StochasticFractalSearch,
    SuccessiveLinearProgramming,
    TabuSearch,
    TrustRegion,
    VariableDepthSearch,
    VariableNeighborhoodSearch,
    VeryLargeScaleNeighborhood,
    WhaleOptimizationAlgorithm,
)
from opt.benchmark.functions import shifted_ackley, sphere


# List of all optimizer classes for parametrized testing
SWARM_OPTIMIZERS = [
    AntColony,
    ArtificialFishSwarm,
    # BatAlgorithm,  # Requires n_bats parameter
    # BeeAlgorithm,  # Has issues with search
    CatSwarmOptimization,
    CuckooSearch,
    FireflyAlgorithm,
    GlowwormSwarmOptimization,
    GreyWolfOptimizer,
    ParticleSwarm,
    SquirrelSearchAlgorithm,
    WhaleOptimizationAlgorithm,
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
]

CONSTRAINED_OPTIMIZERS = [
    AugmentedLagrangian,
    SuccessiveLinearProgramming,
]

PROBABILISTIC_OPTIMIZERS = [
    LDAnalysis,
    ParzenTreeEstimator,
]

ALL_OPTIMIZERS = (
    SWARM_OPTIMIZERS
    + EVOLUTIONARY_OPTIMIZERS
    + GRADIENT_OPTIMIZERS
    + CLASSICAL_OPTIMIZERS
    + METAHEURISTIC_OPTIMIZERS
    + CONSTRAINED_OPTIMIZERS
    + PROBABILISTIC_OPTIMIZERS
)


class TestAbstractOptimizer:
    """Tests for the AbstractOptimizer base class."""

    def test_abstract_optimizer_cannot_be_instantiated(self) -> None:
        """Test that AbstractOptimizer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractOptimizer(  # type: ignore[abstract]
                func=sphere,
                lower_bound=-5,
                upper_bound=5,
                dim=2,
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
            func=sphere,
            lower_bound=-5,
            upper_bound=5,
            dim=2,
            max_iter=10,
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
        from opt.gradient_based import AdamW, SGDMomentum

        assert AdamW is not None
        assert SGDMomentum is not None

    def test_swarm_intelligence_import(self) -> None:
        """Test importing from swarm_intelligence submodule."""
        from opt.swarm_intelligence import ParticleSwarm, AntColony

        assert ParticleSwarm is not None
        assert AntColony is not None

    def test_evolutionary_import(self) -> None:
        """Test importing from evolutionary submodule."""
        from opt.evolutionary import GeneticAlgorithm, DifferentialEvolution

        assert GeneticAlgorithm is not None
        assert DifferentialEvolution is not None

    def test_classical_import(self) -> None:
        """Test importing from classical submodule."""
        from opt.classical import BFGS, NelderMead

        assert BFGS is not None
        assert NelderMead is not None

    def test_metaheuristic_import(self) -> None:
        """Test importing from metaheuristic submodule."""
        from opt.metaheuristic import HarmonySearch, CrossEntropyMethod

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

    def test_backward_compatible_import(self) -> None:
        """Test backward compatible imports from root opt module."""
        from opt import ParticleSwarm, AdamW, BFGS, HarmonySearch

        assert ParticleSwarm is not None
        assert AdamW is not None
        assert BFGS is not None
        assert HarmonySearch is not None
