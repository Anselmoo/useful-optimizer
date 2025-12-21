"""Tests for the constants module.

This module verifies that constants are properly defined and maintain
expected relationships (e.g., tolerances are smaller than bounds).
"""

from __future__ import annotations

from opt.constants import ACKLEY_BOUND
from opt.constants import ADAMW_LEARNING_RATE
from opt.constants import ADAMW_WEIGHT_DECAY
from opt.constants import ADAM_BETA1
from opt.constants import ADAM_BETA2
from opt.constants import ADAM_EPSILON
from opt.constants import BARRIER_METHOD_MIN_MU
from opt.constants import DEFAULT_CONVERGENCE_THRESHOLD
from opt.constants import DEFAULT_MAX_ITERATIONS
from opt.constants import DEFAULT_POPULATION_SIZE
from opt.constants import DEFAULT_SEED
from opt.constants import DEFAULT_TOLERANCE
from opt.constants import ELITE_FRACTION
from opt.constants import EPSILON_STABILITY
from opt.constants import GOLDEN_RATIO
from opt.constants import NADAM_LEARNING_RATE
from opt.constants import PENALTY_METHOD_TOLERANCE
from opt.constants import POWER_THIRTY_TWO
from opt.constants import PSO_COGNITIVE_COEFFICIENT
from opt.constants import PSO_INERTIA_WEIGHT
from opt.constants import PSO_SOCIAL_COEFFICIENT
from opt.constants import ROSENBROCK_BOUND
from opt.constants import SHIFTED_ACKLEY_BOUND
from opt.constants import SPHERE_BOUND


class TestDefaultConstants:
    """Test default parameter constants."""

    def test_default_population_size(self) -> None:
        """Test default population size is positive."""
        assert DEFAULT_POPULATION_SIZE > 0
        assert isinstance(DEFAULT_POPULATION_SIZE, int)

    def test_default_max_iterations(self) -> None:
        """Test default max iterations is positive."""
        assert DEFAULT_MAX_ITERATIONS > 0
        assert isinstance(DEFAULT_MAX_ITERATIONS, int)

    def test_default_seed(self) -> None:
        """Test default seed is non-negative."""
        assert DEFAULT_SEED >= 0
        assert isinstance(DEFAULT_SEED, int)


class TestToleranceConstants:
    """Test convergence and tolerance threshold constants."""

    def test_tolerance_positive(self) -> None:
        """Test that tolerance values are positive."""
        assert DEFAULT_TOLERANCE > 0
        assert DEFAULT_CONVERGENCE_THRESHOLD > 0
        assert BARRIER_METHOD_MIN_MU > 0
        assert PENALTY_METHOD_TOLERANCE > 0

    def test_tolerance_ordering(self) -> None:
        """Test that stricter tolerances have smaller values."""
        assert BARRIER_METHOD_MIN_MU < DEFAULT_CONVERGENCE_THRESHOLD
        assert DEFAULT_CONVERGENCE_THRESHOLD < DEFAULT_TOLERANCE


class TestNumericalStabilityConstants:
    """Test numerical stability constants."""

    def test_epsilon_values(self) -> None:
        """Test epsilon values are appropriately small."""
        assert EPSILON_STABILITY > 0
        assert ADAM_EPSILON > 0
        assert EPSILON_STABILITY < 1e-6

    def test_epsilon_ordering(self) -> None:
        """Test epsilon values maintain expected relationships."""
        assert EPSILON_STABILITY == ADAM_EPSILON


class TestPSOConstants:
    """Test Particle Swarm Optimization constants."""

    def test_pso_coefficients_positive(self) -> None:
        """Test PSO coefficients are positive."""
        assert PSO_INERTIA_WEIGHT > 0
        assert PSO_COGNITIVE_COEFFICIENT > 0
        assert PSO_SOCIAL_COEFFICIENT > 0

    def test_pso_coefficients_reasonable(self) -> None:
        """Test PSO coefficients are in reasonable ranges."""
        assert 0 < PSO_INERTIA_WEIGHT < 1
        assert 0 < PSO_COGNITIVE_COEFFICIENT < 5
        assert 0 < PSO_SOCIAL_COEFFICIENT < 5


class TestAdamConstants:
    """Test Adam-family optimizer constants."""

    def test_adam_beta_ranges(self) -> None:
        """Test Adam beta values are in valid ranges."""
        assert 0 < ADAM_BETA1 < 1
        assert 0 < ADAM_BETA2 < 1
        assert ADAM_BETA1 < ADAM_BETA2

    def test_learning_rates_positive(self) -> None:
        """Test learning rates are positive."""
        assert ADAMW_LEARNING_RATE > 0
        assert NADAM_LEARNING_RATE > 0

    def test_learning_rates_reasonable(self) -> None:
        """Test learning rates are in reasonable ranges."""
        assert ADAMW_LEARNING_RATE < 1
        assert NADAM_LEARNING_RATE < 1

    def test_weight_decay_reasonable(self) -> None:
        """Test weight decay is in reasonable range."""
        assert 0 < ADAMW_WEIGHT_DECAY < 1


class TestAlgorithmSpecificConstants:
    """Test algorithm-specific constants."""

    def test_golden_ratio(self) -> None:
        """Test golden ratio approximation."""
        expected_golden_ratio = (1 + 5**0.5) / 2
        assert abs(GOLDEN_RATIO - expected_golden_ratio) < 1e-10

    def test_fraction_ranges(self) -> None:
        """Test fraction constants are in valid ranges."""
        assert 0 < ELITE_FRACTION < 1


class TestBoundConstants:
    """Test bound constants for benchmark functions."""

    def test_bounds_positive(self) -> None:
        """Test bound constants are positive."""
        assert SHIFTED_ACKLEY_BOUND > 0
        assert ACKLEY_BOUND > 0
        assert SPHERE_BOUND > 0
        assert ROSENBROCK_BOUND > 0

    def test_ackley_bounds_relationship(self) -> None:
        """Test relationship between Ackley bound variants."""
        assert SHIFTED_ACKLEY_BOUND < ACKLEY_BOUND


class TestPowerConstants:
    """Test power and exponent constants."""

    def test_power_values(self) -> None:
        """Test power constants are correct."""
        assert POWER_THIRTY_TWO == 32
        assert isinstance(POWER_THIRTY_TWO, int)


class TestConstantConsistency:
    """Test consistency across related constants."""

    def test_no_zero_constants(self) -> None:
        """Test that no constants are exactly zero (except where intentional)."""
        constants_to_check = [
            DEFAULT_POPULATION_SIZE,
            DEFAULT_MAX_ITERATIONS,
            DEFAULT_TOLERANCE,
            PSO_INERTIA_WEIGHT,
            ADAM_BETA1,
            ADAM_BETA2,
        ]
        for constant in constants_to_check:
            assert constant != 0

    def test_constants_have_expected_types(self) -> None:
        """Test constants have expected types."""
        # Integer constants
        assert isinstance(DEFAULT_POPULATION_SIZE, int)
        assert isinstance(DEFAULT_MAX_ITERATIONS, int)
        assert isinstance(DEFAULT_SEED, int)
        assert isinstance(POWER_THIRTY_TWO, int)

        # Float constants
        assert isinstance(PSO_INERTIA_WEIGHT, float)
        assert isinstance(ADAM_BETA1, float)
        assert isinstance(DEFAULT_TOLERANCE, float)
