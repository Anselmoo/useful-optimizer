"""Constants used across optimization algorithms.

This module centralizes magic numbers and default values used throughout the
optimization library, making them explicit, documented, and maintainable.

Constants are organized into categories:
    - Population and iteration defaults
    - Convergence and tolerance thresholds
    - Numerical stability constants
    - Algorithm-specific parameters (PSO, gradient-based, etc.)
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# Population and Iteration Defaults
# =============================================================================

DEFAULT_POPULATION_SIZE = 100
"""Default number of individuals in population-based algorithms."""

DEFAULT_MAX_ITERATIONS = 1000
"""Default maximum number of optimization iterations."""

DEFAULT_SEED = 42
"""Default random seed for reproducibility."""

# =============================================================================
# Convergence and Tolerance Thresholds
# =============================================================================

DEFAULT_TOLERANCE = 1e-6
"""Default convergence tolerance for optimization algorithms."""

DEFAULT_CONVERGENCE_THRESHOLD = 1e-8
"""Default threshold for determining convergence."""

BARRIER_METHOD_MIN_MU = 1e-10
"""Minimum barrier parameter for barrier method termination."""

PENALTY_METHOD_TOLERANCE = 1e-6
"""Tolerance for penalty method constraint satisfaction."""

# =============================================================================
# Numerical Stability Constants
# =============================================================================

EPSILON_STABILITY = 1e-8
"""Small constant for numerical stability in gradient-based methods."""

EPSILON_GRADIENT = 1e-10
"""Epsilon for gradient computation numerical stability."""

# =============================================================================
# Particle Swarm Optimization (PSO) Constants
# =============================================================================

PSO_INERTIA_WEIGHT = 0.5
"""Default inertia weight for PSO velocity updates."""

PSO_COGNITIVE_COEFFICIENT = 1.5
"""Default cognitive coefficient (c1) for PSO."""

PSO_SOCIAL_COEFFICIENT = 1.5
"""Default social coefficient (c2) for PSO."""

PSO_CONSTRICTION_COEFFICIENT = 0.7298
"""Constriction coefficient for PSO (Clerc and Kennedy, 2002)."""

PSO_ACCELERATION_COEFFICIENT = 1.49618
"""Acceleration coefficient for PSO (common in literature)."""

# =============================================================================
# Gradient-Based Optimizer Constants
# =============================================================================

ADAM_BETA1 = 0.9
"""Default beta1 (first moment decay) for Adam-family optimizers."""

ADAM_BETA2 = 0.999
"""Default beta2 (second moment decay) for Adam-family optimizers."""

ADAM_EPSILON = 1e-8
"""Default epsilon for numerical stability in Adam-family optimizers."""

ADAMW_LEARNING_RATE = 0.001
"""Default learning rate for AdamW optimizer."""

ADAMW_WEIGHT_DECAY = 0.01
"""Default weight decay coefficient for AdamW optimizer."""

NADAM_LEARNING_RATE = 0.002
"""Default learning rate for Nadam optimizer."""

SGD_MOMENTUM = 0.9
"""Default momentum coefficient for SGD with momentum."""

RMSPROP_DECAY_RATE = 0.9
"""Default decay rate for RMSprop optimizer."""

# =============================================================================
# Algorithm-Specific Constants
# =============================================================================

GOLDEN_RATIO = 1.618033988749895
"""The golden ratio, used in various optimization algorithms."""

ELITE_FRACTION = 0.2
"""Default fraction of elite samples in cross-entropy method."""

TRAINING_PROBABILITY = 0.2
"""Default probability for training phase in certain algorithms."""

SOCIAL_COEFFICIENT = 0.2
"""Default self-introspection coefficient for social algorithms."""

BANDWIDTH_DEFAULT = 0.2
"""Default bandwidth for kernel density estimation."""

GAMMA_DEFAULT = 0.15
"""Default gamma parameter for various algorithms."""

BETA_ATTRACTION = 0.2
"""Default beta multiplier for attraction/repulsion forces."""

ACKLEY_A = 20.0
"""Ackley function constant 'a'."""

ACKLEY_B = 0.2
"""Ackley function constant 'b'."""

ACKLEY_C = 2.0 * np.pi
"""Ackley function constant 'c' (typically 2*pi)."""

MATYAS_COEFFICIENT_A = 0.26
"""Matyas function coefficient for squared terms."""

MATYAS_COEFFICIENT_B = 0.48
"""Matyas function coefficient for cross term."""

# =============================================================================
# Multi-Objective Constants
# =============================================================================

WEIGHT_ADJUSTMENT_MIN = 1e-6
"""Minimum weight value for numerical stability in multi-objective algorithms."""

# =============================================================================
# Probability and Fraction Constants
# =============================================================================

PROBABILITY_HALF = 0.5
"""Probability value of 0.5 for binary decisions."""

FRACTION_QUARTER = 0.25
"""Fraction value of 0.25 (25%)."""

FRACTION_THIRD = 0.33
"""Fraction value of 0.33 (approximately 1/3)."""

FRACTION_TWO_THIRDS = 0.67
"""Fraction value of 0.67 (approximately 2/3)."""

# =============================================================================
# Dimension and Bound Defaults
# =============================================================================

DEFAULT_DIM = 2
"""Default dimensionality for test problems."""

SHIFTED_ACKLEY_BOUND = 2.768
"""Typical bound for shifted Ackley function."""

ACKLEY_BOUND = 32.768
"""Typical bound for standard Ackley function."""

SPHERE_BOUND = 5.0
"""Typical bound for sphere function."""

ROSENBROCK_BOUND = 5.0
"""Typical bound for Rosenbrock function."""

# =============================================================================
# Power and Exponent Constants
# =============================================================================

POWER_TWO = 2
"""Exponent value of 2 for squaring operations."""

POWER_THIRTY_TWO = 32
"""Exponent value of 32 for random seed generation (2^32)."""
