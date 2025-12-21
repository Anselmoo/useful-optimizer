"""Pytest configuration and shared fixtures for optimizer tests.

This module provides fixtures for benchmark functions with their known optimal
solutions, test configurations, and helper utilities for comprehensive optimizer
testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

from opt.benchmark.functions import ackley
from opt.benchmark.functions import beale
from opt.benchmark.functions import booth
from opt.benchmark.functions import easom
from opt.benchmark.functions import goldstein_price
from opt.benchmark.functions import griewank
from opt.benchmark.functions import himmelblau
from opt.benchmark.functions import levi
from opt.benchmark.functions import levi_n13
from opt.benchmark.functions import matyas
from opt.benchmark.functions import mccormick
from opt.benchmark.functions import rastrigin
from opt.benchmark.functions import rosenbrock
from opt.benchmark.functions import schwefel
from opt.benchmark.functions import shifted_ackley
from opt.benchmark.functions import sphere
from opt.benchmark.functions import three_hump_camel


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


@dataclass
class BenchmarkFunction:
    """Configuration for a benchmark function with known optimal solution.

    Attributes:
        name: Human-readable name of the function.
        func: The benchmark function callable.
        optimal_point: Known optimal point(s) as numpy array.
        optimal_value: Known optimal function value.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        tolerance_point: Acceptable tolerance for solution point (distance).
        tolerance_value: Acceptable tolerance for fitness value.
        difficulty: Estimated difficulty level ('easy', 'medium', 'hard').
    """

    name: str
    func: Callable[[ndarray], float]
    optimal_point: ndarray
    optimal_value: float
    lower_bound: float
    upper_bound: float
    dim: int
    tolerance_point: float = 0.5  # Euclidean distance tolerance
    tolerance_value: float = 1.0  # Fitness value tolerance
    difficulty: str = "medium"


# =============================================================================
# Benchmark Function Definitions with Known Optima
# =============================================================================

BENCHMARK_FUNCTIONS: dict[str, BenchmarkFunction] = {
    # Easy functions - unimodal, smooth
    "sphere": BenchmarkFunction(
        name="Sphere",
        func=sphere,
        optimal_point=np.array([0.0, 0.0]),
        optimal_value=0.0,
        lower_bound=-5.12,
        upper_bound=5.12,
        dim=2,
        tolerance_point=0.1,
        tolerance_value=0.01,
        difficulty="easy",
    ),
    "shifted_ackley": BenchmarkFunction(
        name="Shifted Ackley",
        func=shifted_ackley,
        optimal_point=np.array([1.0, 0.5]),  # shift=(1, 0.5) moves optimum
        optimal_value=0.0,
        lower_bound=-2.768,
        upper_bound=2.768,
        dim=2,
        tolerance_point=0.2,  # Critical: if 1.2 and 0.7 comes out, it's bad
        tolerance_value=0.5,
        difficulty="medium",
    ),
    "ackley": BenchmarkFunction(
        name="Ackley",
        func=ackley,
        optimal_point=np.array([0.0, 0.0]),
        optimal_value=0.0,
        lower_bound=-5.0,
        upper_bound=5.0,
        dim=2,
        tolerance_point=0.3,
        tolerance_value=1.0,
        difficulty="medium",
    ),
    "rosenbrock": BenchmarkFunction(
        name="Rosenbrock",
        func=rosenbrock,
        optimal_point=np.array([1.0, 1.0]),
        optimal_value=0.0,
        lower_bound=-5.0,
        upper_bound=10.0,
        dim=2,
        tolerance_point=0.5,
        tolerance_value=1.0,
        difficulty="hard",
    ),
    "rastrigin": BenchmarkFunction(
        name="Rastrigin",
        func=rastrigin,
        optimal_point=np.array([0.0, 0.0]),
        optimal_value=0.0,
        lower_bound=-5.12,
        upper_bound=5.12,
        dim=2,
        tolerance_point=0.5,
        tolerance_value=2.0,
        difficulty="hard",
    ),
    "griewank": BenchmarkFunction(
        name="Griewank",
        func=griewank,
        optimal_point=np.array([0.0, 0.0]),
        optimal_value=0.0,
        lower_bound=-600.0,
        upper_bound=600.0,
        dim=2,
        tolerance_point=1.0,
        tolerance_value=0.5,
        difficulty="medium",
    ),
    "schwefel": BenchmarkFunction(
        name="Schwefel",
        func=schwefel,
        optimal_point=np.array([420.9687, 420.9687]),
        optimal_value=0.0,
        lower_bound=-500.0,
        upper_bound=500.0,
        dim=2,
        tolerance_point=50.0,
        tolerance_value=100.0,
        difficulty="hard",
    ),
    "booth": BenchmarkFunction(
        name="Booth",
        func=booth,
        optimal_point=np.array([1.0, 3.0]),
        optimal_value=0.0,
        lower_bound=-10.0,
        upper_bound=10.0,
        dim=2,
        tolerance_point=0.3,
        tolerance_value=0.5,
        difficulty="easy",
    ),
    "matyas": BenchmarkFunction(
        name="Matyas",
        func=matyas,
        optimal_point=np.array([0.0, 0.0]),
        optimal_value=0.0,
        lower_bound=-10.0,
        upper_bound=10.0,
        dim=2,
        tolerance_point=0.2,
        tolerance_value=0.1,
        difficulty="easy",
    ),
    "himmelblau": BenchmarkFunction(
        name="Himmelblau",
        func=himmelblau,
        # Has 4 identical local minima; testing one of them
        optimal_point=np.array([3.0, 2.0]),
        optimal_value=0.0,
        lower_bound=-5.0,
        upper_bound=5.0,
        dim=2,
        tolerance_point=0.5,
        tolerance_value=1.0,
        difficulty="medium",
    ),
    "three_hump_camel": BenchmarkFunction(
        name="Three-Hump Camel",
        func=three_hump_camel,
        optimal_point=np.array([0.0, 0.0]),
        optimal_value=0.0,
        lower_bound=-5.0,
        upper_bound=5.0,
        dim=2,
        tolerance_point=0.3,
        tolerance_value=0.5,
        difficulty="easy",
    ),
    "beale": BenchmarkFunction(
        name="Beale",
        func=beale,
        optimal_point=np.array([3.0, 0.5]),
        optimal_value=0.0,
        lower_bound=-4.5,
        upper_bound=4.5,
        dim=2,
        tolerance_point=0.5,
        tolerance_value=1.0,
        difficulty="medium",
    ),
    "goldstein_price": BenchmarkFunction(
        name="Goldstein-Price",
        func=goldstein_price,
        optimal_point=np.array([0.0, -1.0]),
        optimal_value=3.0,
        lower_bound=-2.0,
        upper_bound=2.0,
        dim=2,
        tolerance_point=0.3,
        tolerance_value=5.0,
        difficulty="medium",
    ),
    "levi": BenchmarkFunction(
        name="Levi",
        func=levi,
        optimal_point=np.array([1.0, 1.0]),
        optimal_value=0.0,
        lower_bound=-10.0,
        upper_bound=10.0,
        dim=2,
        tolerance_point=0.3,
        tolerance_value=0.5,
        difficulty="medium",
    ),
    "levi_n13": BenchmarkFunction(
        name="Levi N.13",
        func=levi_n13,
        optimal_point=np.array([1.0, 1.0]),
        optimal_value=0.0,
        lower_bound=-10.0,
        upper_bound=10.0,
        dim=2,
        tolerance_point=0.3,
        tolerance_value=0.5,
        difficulty="medium",
    ),
    "easom": BenchmarkFunction(
        name="Easom",
        func=easom,
        optimal_point=np.array([np.pi, np.pi]),
        optimal_value=-1.0,
        lower_bound=-100.0,
        upper_bound=100.0,
        dim=2,
        tolerance_point=0.5,
        tolerance_value=0.5,
        difficulty="hard",
    ),
    "mccormick": BenchmarkFunction(
        name="McCormick",
        func=mccormick,
        optimal_point=np.array([-0.54719, -1.54719]),
        optimal_value=-1.9133,
        lower_bound=-1.5,
        upper_bound=4.0,
        dim=2,
        tolerance_point=0.3,
        tolerance_value=0.5,
        difficulty="easy",
    ),
}

# Functions suitable for quick tests (easy/medium difficulty)
QUICK_TEST_FUNCTIONS = [
    "sphere",
    "shifted_ackley",
    "booth",
    "matyas",
    "three_hump_camel",
]

# Functions for comprehensive tests (all difficulties)
COMPREHENSIVE_TEST_FUNCTIONS = list(BENCHMARK_FUNCTIONS.keys())

# Functions by difficulty
EASY_FUNCTIONS = [k for k, v in BENCHMARK_FUNCTIONS.items() if v.difficulty == "easy"]
MEDIUM_FUNCTIONS = [
    k for k, v in BENCHMARK_FUNCTIONS.items() if v.difficulty == "medium"
]
HARD_FUNCTIONS = [k for k, v in BENCHMARK_FUNCTIONS.items() if v.difficulty == "hard"]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sphere_benchmark() -> BenchmarkFunction:
    """Fixture for sphere function benchmark."""
    return BENCHMARK_FUNCTIONS["sphere"]


@pytest.fixture
def shifted_ackley_benchmark() -> BenchmarkFunction:
    """Fixture for shifted_ackley function benchmark."""
    return BENCHMARK_FUNCTIONS["shifted_ackley"]


@pytest.fixture
def rosenbrock_benchmark() -> BenchmarkFunction:
    """Fixture for rosenbrock function benchmark."""
    return BENCHMARK_FUNCTIONS["rosenbrock"]


@pytest.fixture(params=QUICK_TEST_FUNCTIONS)
def quick_benchmark(request: pytest.FixtureRequest) -> BenchmarkFunction:
    """Parametrized fixture for quick benchmark tests."""
    return BENCHMARK_FUNCTIONS[request.param]


@pytest.fixture(params=COMPREHENSIVE_TEST_FUNCTIONS)
def all_benchmarks(request: pytest.FixtureRequest) -> BenchmarkFunction:
    """Parametrized fixture for all benchmark functions."""
    return BENCHMARK_FUNCTIONS[request.param]


@pytest.fixture(params=EASY_FUNCTIONS)
def easy_benchmark(request: pytest.FixtureRequest) -> BenchmarkFunction:
    """Parametrized fixture for easy benchmark functions."""
    return BENCHMARK_FUNCTIONS[request.param]


@pytest.fixture(params=MEDIUM_FUNCTIONS)
def medium_benchmark(request: pytest.FixtureRequest) -> BenchmarkFunction:
    """Parametrized fixture for medium difficulty benchmark functions."""
    return BENCHMARK_FUNCTIONS[request.param]


@pytest.fixture(params=HARD_FUNCTIONS)
def hard_benchmark(request: pytest.FixtureRequest) -> BenchmarkFunction:
    """Parametrized fixture for hard benchmark functions."""
    return BENCHMARK_FUNCTIONS[request.param]


# =============================================================================
# Optimizer Test Configuration
# =============================================================================


@dataclass
class OptimizerTestConfig:
    """Configuration for optimizer testing.

    Attributes:
        max_iter_quick: Maximum iterations for quick tests.
        max_iter_full: Maximum iterations for full benchmark tests.
        population_size: Default population size for population-based methods.
        seed: Random seed for reproducibility.
        strict_tolerance: Whether to use strict tolerances.
    """

    max_iter_quick: int = 50
    max_iter_full: int = 500
    population_size: int = 30
    seed: int = 42
    strict_tolerance: bool = False


# Alias for external use
TestConfig = OptimizerTestConfig


@pytest.fixture
def test_config() -> OptimizerTestConfig:
    """Fixture providing test configuration."""
    return OptimizerTestConfig()


@pytest.fixture
def strict_test_config() -> TestConfig:
    """Fixture providing strict test configuration."""
    return TestConfig(max_iter_quick=100, max_iter_full=1000, strict_tolerance=True)


# =============================================================================
# Helper Functions
# =============================================================================


def calculate_solution_quality(
    solution: ndarray, fitness: float, benchmark: BenchmarkFunction
) -> dict[str, float | bool]:
    """Calculate quality metrics for an optimization solution.

    Args:
        solution: The solution found by the optimizer.
        fitness: The fitness value of the solution.
        benchmark: The benchmark function configuration.

    Returns:
        Dictionary with quality metrics:
            - point_distance: Euclidean distance from optimal point
            - value_error: Absolute error in fitness value
            - point_within_tolerance: Whether point is within tolerance
            - value_within_tolerance: Whether value is within tolerance
            - overall_pass: Whether both tolerances are met
    """
    point_distance = float(np.linalg.norm(solution - benchmark.optimal_point))
    value_error = abs(fitness - benchmark.optimal_value)

    point_ok = point_distance <= benchmark.tolerance_point
    value_ok = value_error <= benchmark.tolerance_value

    return {
        "point_distance": point_distance,
        "value_error": value_error,
        "point_within_tolerance": point_ok,
        "value_within_tolerance": value_ok,
        "overall_pass": point_ok and value_ok,
    }


def is_near_any_optimum(
    solution: ndarray, optima: list[ndarray], tolerance: float
) -> bool:
    """Check if solution is near any of multiple optimal points.

    Some functions like Himmelblau have multiple global optima.

    Args:
        solution: The solution found by the optimizer.
        optima: List of known optimal points.
        tolerance: Distance tolerance.

    Returns:
        True if solution is within tolerance of any optimum.
    """
    return any(np.linalg.norm(solution - optimum) <= tolerance for optimum in optima)


# Himmelblau has 4 identical minima
HIMMELBLAU_OPTIMA = [
    np.array([3.0, 2.0]),
    np.array([-2.805118, 3.131312]),
    np.array([-3.779310, -3.283186]),
    np.array([3.584428, -1.848126]),
]


@pytest.fixture
def himmelblau_optima() -> list[ndarray]:
    """Fixture providing all Himmelblau function optima."""
    return HIMMELBLAU_OPTIMA
