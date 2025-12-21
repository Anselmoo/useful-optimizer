"""Benchmark tests for optimizer quality validation.

This module tests that optimizers find solutions within acceptable tolerance
of known optimal points. These tests verify the actual optimization quality,
not just that the optimizers run without errors.

Critical Test Cases:
- shifted_ackley: optimal at (1.0, 0.5) - solutions like (1.2, 0.7) are failures
- sphere: optimal at (0, 0) - should be very accurate
- rosenbrock: optimal at (1, 1) - harder, wider tolerance allowed
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from opt import BFGS
from opt import LBFGS
from opt import SGD
from opt import ADAGrad
from opt import ADAMOptimization
from opt import AMSGrad
from opt import AdaDelta
from opt import AdaMax
from opt import AdamW
from opt import AntColony
from opt import ArtificialFishSwarm
from opt import AugmentedLagrangian
from opt import BatAlgorithm
from opt import BeeAlgorithm
from opt import CMAESAlgorithm
from opt import CatSwarmOptimization
from opt import CollidingBodiesOptimization
from opt import ConjugateGradient
from opt import CrossEntropyMethod
from opt import CuckooSearch
from opt import CulturalAlgorithm
from opt import DifferentialEvolution
from opt import EagleStrategy
from opt import EstimationOfDistributionAlgorithm
from opt import FireflyAlgorithm
from opt import GeneticAlgorithm
from opt import GlowwormSwarmOptimization
from opt import GreyWolfOptimizer
from opt import HarmonySearch
from opt import HillClimbing
from opt import ImperialistCompetitiveAlgorithm
from opt import LDAnalysis
from opt import Nadam
from opt import NelderMead
from opt import NesterovAcceleratedGradient
from opt import ParticleFilter
from opt import ParticleSwarm
from opt import ParzenTreeEstimator
from opt import Powell
from opt import RMSprop
from opt import SGDMomentum
from opt import ShuffledFrogLeapingAlgorithm
from opt import SimulatedAnnealing
from opt import SineCosineAlgorithm
from opt import SquirrelSearchAlgorithm
from opt import StochasticDiffusionSearch
from opt import StochasticFractalSearch
from opt import SuccessiveLinearProgramming
from opt import TabuSearch
from opt import TrustRegion
from opt import VariableDepthSearch
from opt import VariableNeighborhoodSearch
from opt import VeryLargeScaleNeighborhood
from opt import WhaleOptimizationAlgorithm

from .conftest import calculate_solution_quality
from .conftest import is_near_any_optimum


if TYPE_CHECKING:
    from opt import AbstractOptimizer

    from .conftest import BenchmarkFunction


# =============================================================================
# Optimizer Categories for Benchmark Tests
# =============================================================================

# High-performing optimizers that should achieve tight tolerances
HIGH_PERFORMANCE_OPTIMIZERS = [
    ParticleSwarm,
    DifferentialEvolution,
    CMAESAlgorithm,
    Powell,
    WhaleOptimizationAlgorithm,
]

# Optimizers that struggle with multimodal functions like shifted_ackley
# They converge to local minima instead of the global optimum
LOCAL_MINIMA_PRONE_OPTIMIZERS = [
    pytest.param(
        BFGS,
        marks=pytest.mark.xfail(
            reason="BFGS converges to local minimum on multimodal shifted_ackley",
            strict=False,
        ),
    ),
    pytest.param(
        LBFGS,
        marks=pytest.mark.xfail(
            reason="LBFGS converges to local minimum on multimodal shifted_ackley",
            strict=False,
        ),
    ),
    pytest.param(
        NelderMead,
        marks=pytest.mark.xfail(
            reason="NelderMead converges to local minimum on multimodal shifted_ackley",
            strict=False,
        ),
    ),
    pytest.param(
        GreyWolfOptimizer,
        marks=pytest.mark.xfail(
            reason="GreyWolfOptimizer has convergence issues on shifted_ackley",
            strict=False,
        ),
    ),
]

# Medium performance - reasonable results expected
MEDIUM_PERFORMANCE_OPTIMIZERS = [
    GeneticAlgorithm,
    AntColony,
    FireflyAlgorithm,
    CuckooSearch,
    HarmonySearch,
    SimulatedAnnealing,
    BeeAlgorithm,
    CrossEntropyMethod,
    SineCosineAlgorithm,
]

# Optimizers that may need more iterations or have variable performance
VARIABLE_PERFORMANCE_OPTIMIZERS = [
    ArtificialFishSwarm,
    CatSwarmOptimization,
    GlowwormSwarmOptimization,
    SquirrelSearchAlgorithm,
    CollidingBodiesOptimization,
    EagleStrategy,
    CulturalAlgorithm,
    EstimationOfDistributionAlgorithm,
    ImperialistCompetitiveAlgorithm,
    ParticleFilter,
    ShuffledFrogLeapingAlgorithm,
    StochasticDiffusionSearch,
    StochasticFractalSearch,
    VariableDepthSearch,
    VariableNeighborhoodSearch,
    VeryLargeScaleNeighborhood,
]

# Gradient-based optimizers (may converge to local optima)
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
    ConjugateGradient,
    TrustRegion,
    HillClimbing,
    TabuSearch,
]

# Constrained/Probabilistic optimizers
SPECIALIZED_OPTIMIZERS = [
    AugmentedLagrangian,
    SuccessiveLinearProgramming,
    LDAnalysis,
    ParzenTreeEstimator,
]


# =============================================================================
# Critical Benchmark Tests - shifted_ackley
# =============================================================================


class TestShiftedAckleyBenchmark:
    """Critical tests for shifted_ackley function.

    The shifted_ackley function has its optimum at (1.0, 0.5) due to shift=(1, 0.5).
    Solutions like (1.2, 0.7) indicate the optimizer is NOT converging properly
    and should be flagged as critical failures.
    """

    OPTIMAL_POINT = np.array([1.0, 0.5])
    CRITICAL_TOLERANCE = 0.2  # Distance > 0.2 is a critical failure
    WARNING_TOLERANCE = 0.1  # Distance > 0.1 but <= 0.2 is a warning

    @pytest.mark.parametrize(
        "optimizer_class", HIGH_PERFORMANCE_OPTIMIZERS + LOCAL_MINIMA_PRONE_OPTIMIZERS
    )
    def test_high_performance_optimizers(
        self,
        optimizer_class: type[AbstractOptimizer],
        shifted_ackley_benchmark: BenchmarkFunction,
    ) -> None:
        """Test that high-performance optimizers find the shifted_ackley optimum."""
        optimizer = optimizer_class(
            func=shifted_ackley_benchmark.func,
            lower_bound=shifted_ackley_benchmark.lower_bound,
            upper_bound=shifted_ackley_benchmark.upper_bound,
            dim=shifted_ackley_benchmark.dim,
            max_iter=200,
        )
        solution, fitness = optimizer.search()

        # Calculate distance from known optimum
        distance = np.linalg.norm(solution - self.OPTIMAL_POINT)

        # Critical assertion - must not be too far from optimum
        assert distance <= self.CRITICAL_TOLERANCE, (
            f"{optimizer_class.__name__} CRITICAL FAILURE on shifted_ackley: "
            f"Solution {solution} is {distance:.4f} away from optimum {self.OPTIMAL_POINT}. "
            f"Fitness: {fitness:.6f}. Maximum allowed distance: {self.CRITICAL_TOLERANCE}"
        )

    @pytest.mark.parametrize("optimizer_class", MEDIUM_PERFORMANCE_OPTIMIZERS)
    def test_medium_performance_optimizers(
        self,
        optimizer_class: type[AbstractOptimizer],
        shifted_ackley_benchmark: BenchmarkFunction,
    ) -> None:
        """Test medium-performance optimizers on shifted_ackley with relaxed tolerance."""
        optimizer = optimizer_class(
            func=shifted_ackley_benchmark.func,
            lower_bound=shifted_ackley_benchmark.lower_bound,
            upper_bound=shifted_ackley_benchmark.upper_bound,
            dim=shifted_ackley_benchmark.dim,
            max_iter=300,
        )
        solution, _fitness = optimizer.search()

        distance = np.linalg.norm(solution - self.OPTIMAL_POINT)

        # More relaxed tolerance for medium performers
        relaxed_tolerance = 0.3
        assert distance <= relaxed_tolerance, (
            f"{optimizer_class.__name__} FAILURE on shifted_ackley: "
            f"Solution {solution} is {distance:.4f} away from optimum {self.OPTIMAL_POINT}. "
            f"Maximum allowed: {relaxed_tolerance}"
        )

    def test_bat_algorithm_shifted_ackley(
        self, shifted_ackley_benchmark: BenchmarkFunction
    ) -> None:
        """Test BatAlgorithm (requires special n_bats parameter)."""
        optimizer = BatAlgorithm(
            func=shifted_ackley_benchmark.func,
            lower_bound=shifted_ackley_benchmark.lower_bound,
            upper_bound=shifted_ackley_benchmark.upper_bound,
            dim=shifted_ackley_benchmark.dim,
            n_bats=30,
            max_iter=200,
        )
        solution, _fitness = optimizer.search()

        distance = np.linalg.norm(solution - self.OPTIMAL_POINT)
        assert distance <= 0.3, (
            f"BatAlgorithm solution {solution} too far from optimum. Distance: {distance:.4f}"
        )


# =============================================================================
# Sphere Function Benchmark Tests
# =============================================================================


class TestSphereBenchmark:
    """Tests for sphere function - should be easy for most optimizers."""

    OPTIMAL_POINT = np.array([0.0, 0.0])
    TIGHT_TOLERANCE = 0.1
    RELAXED_TOLERANCE = 0.5

    @pytest.mark.parametrize("optimizer_class", HIGH_PERFORMANCE_OPTIMIZERS)
    def test_high_performance_on_sphere(
        self,
        optimizer_class: type[AbstractOptimizer],
        sphere_benchmark: BenchmarkFunction,
    ) -> None:
        """High-performance optimizers should easily solve sphere."""
        optimizer = optimizer_class(
            func=sphere_benchmark.func,
            lower_bound=sphere_benchmark.lower_bound,
            upper_bound=sphere_benchmark.upper_bound,
            dim=sphere_benchmark.dim,
            max_iter=100,
        )
        solution, fitness = optimizer.search()

        distance = np.linalg.norm(solution - self.OPTIMAL_POINT)
        assert distance <= self.TIGHT_TOLERANCE, (
            f"{optimizer_class.__name__} failed on simple sphere function. "
            f"Distance: {distance:.4f}, Expected < {self.TIGHT_TOLERANCE}"
        )
        assert fitness <= 0.01, f"Fitness {fitness} too high for sphere"


# =============================================================================
# Comprehensive Quality Tests
# =============================================================================


class TestOptimizerQuality:
    """Comprehensive quality tests across multiple benchmark functions."""

    @pytest.mark.parametrize(
        "optimizer_class", HIGH_PERFORMANCE_OPTIMIZERS + MEDIUM_PERFORMANCE_OPTIMIZERS
    )
    def test_optimizer_on_easy_functions(
        self,
        optimizer_class: type[AbstractOptimizer],
        easy_benchmark: BenchmarkFunction,
    ) -> None:
        """Test optimizers on easy benchmark functions."""
        optimizer = optimizer_class(
            func=easy_benchmark.func,
            lower_bound=easy_benchmark.lower_bound,
            upper_bound=easy_benchmark.upper_bound,
            dim=easy_benchmark.dim,
            max_iter=200,
        )
        solution, fitness = optimizer.search()

        quality = calculate_solution_quality(solution, fitness, easy_benchmark)

        # Easy functions should be solved with reasonable accuracy
        assert quality["point_within_tolerance"] or quality["value_within_tolerance"], (
            f"{optimizer_class.__name__} failed on {easy_benchmark.name}: "
            f"Point distance: {quality['point_distance']:.4f} "
            f"(tolerance: {easy_benchmark.tolerance_point}), "
            f"Value error: {quality['value_error']:.4f} "
            f"(tolerance: {easy_benchmark.tolerance_value})"
        )

    @pytest.mark.parametrize("optimizer_class", HIGH_PERFORMANCE_OPTIMIZERS)
    def test_high_performers_on_medium_functions(
        self,
        optimizer_class: type[AbstractOptimizer],
        medium_benchmark: BenchmarkFunction,
    ) -> None:
        """Test high-performance optimizers on medium difficulty functions."""
        optimizer = optimizer_class(
            func=medium_benchmark.func,
            lower_bound=medium_benchmark.lower_bound,
            upper_bound=medium_benchmark.upper_bound,
            dim=medium_benchmark.dim,
            max_iter=500,
        )
        solution, fitness = optimizer.search()

        quality = calculate_solution_quality(solution, fitness, medium_benchmark)

        # At least one criterion should be met
        assert quality["point_within_tolerance"] or quality["value_within_tolerance"], (
            f"{optimizer_class.__name__} failed on {medium_benchmark.name}: "
            f"Solution: {solution}, Fitness: {fitness:.6f}"
        )


# =============================================================================
# Himmelblau Special Case (Multiple Optima)
# =============================================================================


class TestHimmelblauMultipleOptima:
    """Test Himmelblau function which has 4 identical global minima."""

    @pytest.mark.parametrize("optimizer_class", HIGH_PERFORMANCE_OPTIMIZERS)
    def test_finds_any_optimum(
        self,
        optimizer_class: type[AbstractOptimizer],
        himmelblau_optima: list[np.ndarray],
    ) -> None:
        """Test that optimizer finds one of Himmelblau's four minima."""
        from opt.benchmark.functions import himmelblau

        optimizer = optimizer_class(
            func=himmelblau, lower_bound=-5.0, upper_bound=5.0, dim=2, max_iter=300
        )
        solution, fitness = optimizer.search()

        # Should be near any of the 4 optima
        near_optimum = is_near_any_optimum(solution, himmelblau_optima, tolerance=0.5)
        assert near_optimum, (
            f"{optimizer_class.__name__} did not find any Himmelblau optimum. "
            f"Solution: {solution}, Fitness: {fitness:.6f}"
        )
        assert fitness <= 1.0, f"Fitness {fitness} too high for Himmelblau"


# =============================================================================
# Fitness Value Sanity Tests
# =============================================================================


class TestFitnessSanity:
    """Tests ensuring fitness values are reasonable."""

    @pytest.mark.parametrize(
        "optimizer_class", HIGH_PERFORMANCE_OPTIMIZERS + MEDIUM_PERFORMANCE_OPTIMIZERS
    )
    def test_fitness_improves_from_random(
        self,
        optimizer_class: type[AbstractOptimizer],
        sphere_benchmark: BenchmarkFunction,
    ) -> None:
        """Test that optimization actually improves over random initialization."""
        # Generate random fitness for comparison
        rng = np.random.default_rng(42)
        random_points = rng.uniform(
            sphere_benchmark.lower_bound,
            sphere_benchmark.upper_bound,
            (100, sphere_benchmark.dim),
        )
        random_fitnesses = [sphere_benchmark.func(p) for p in random_points]
        avg_random_fitness = np.mean(random_fitnesses)

        # Run optimizer
        optimizer = optimizer_class(
            func=sphere_benchmark.func,
            lower_bound=sphere_benchmark.lower_bound,
            upper_bound=sphere_benchmark.upper_bound,
            dim=sphere_benchmark.dim,
            max_iter=100,
        )
        _, fitness = optimizer.search()

        # Optimized fitness should be significantly better than random
        assert fitness < avg_random_fitness, (
            f"{optimizer_class.__name__} did not improve over random. "
            f"Optimized: {fitness:.4f}, Avg random: {avg_random_fitness:.4f}"
        )

    @pytest.mark.parametrize("optimizer_class", HIGH_PERFORMANCE_OPTIMIZERS)
    def test_fitness_is_finite(
        self,
        optimizer_class: type[AbstractOptimizer],
        quick_benchmark: BenchmarkFunction,
    ) -> None:
        """Test that fitness values are finite numbers."""
        optimizer = optimizer_class(
            func=quick_benchmark.func,
            lower_bound=quick_benchmark.lower_bound,
            upper_bound=quick_benchmark.upper_bound,
            dim=quick_benchmark.dim,
            max_iter=50,
        )
        solution, fitness = optimizer.search()

        assert np.isfinite(fitness), f"Non-finite fitness: {fitness}"
        assert np.all(np.isfinite(solution)), f"Non-finite solution: {solution}"


# =============================================================================
# Reproducibility Tests
# =============================================================================


class TestReproducibility:
    """Tests for optimizer reproducibility with seeds."""

    @pytest.mark.parametrize(
        "optimizer_class",
        [ParticleSwarm, DifferentialEvolution, GeneticAlgorithm, FireflyAlgorithm],
    )
    def test_seeded_reproducibility(
        self,
        optimizer_class: type[AbstractOptimizer],
        sphere_benchmark: BenchmarkFunction,
    ) -> None:
        """Test that seeded optimizers produce reproducible results."""
        results = []
        for _ in range(2):
            optimizer = optimizer_class(
                func=sphere_benchmark.func,
                lower_bound=sphere_benchmark.lower_bound,
                upper_bound=sphere_benchmark.upper_bound,
                dim=sphere_benchmark.dim,
                max_iter=50,
                seed=42,
            )
            solution, fitness = optimizer.search()
            results.append((solution.copy(), fitness))

        # Both runs should produce identical results
        np.testing.assert_array_almost_equal(
            results[0][0],
            results[1][0],
            decimal=10,
            err_msg=f"{optimizer_class.__name__} not reproducible with same seed",
        )
        assert results[0][1] == results[1][1], "Fitness values should be identical"


# =============================================================================
# Solution Bounds Tests
# =============================================================================


class TestSolutionBounds:
    """Tests ensuring solutions stay within specified bounds."""

    @pytest.mark.parametrize(
        "optimizer_class", HIGH_PERFORMANCE_OPTIMIZERS + MEDIUM_PERFORMANCE_OPTIMIZERS
    )
    def test_solution_within_bounds(
        self,
        optimizer_class: type[AbstractOptimizer],
        quick_benchmark: BenchmarkFunction,
    ) -> None:
        """Test that solutions respect the search space bounds."""
        optimizer = optimizer_class(
            func=quick_benchmark.func,
            lower_bound=quick_benchmark.lower_bound,
            upper_bound=quick_benchmark.upper_bound,
            dim=quick_benchmark.dim,
            max_iter=100,
        )
        solution, _ = optimizer.search()

        assert np.all(solution >= quick_benchmark.lower_bound - 1e-10), (
            f"{optimizer_class.__name__} returned solution below lower bound: "
            f"{solution} < {quick_benchmark.lower_bound}"
        )
        assert np.all(solution <= quick_benchmark.upper_bound + 1e-10), (
            f"{optimizer_class.__name__} returned solution above upper bound: "
            f"{solution} > {quick_benchmark.upper_bound}"
        )
