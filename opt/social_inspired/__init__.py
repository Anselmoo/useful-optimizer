"""Social-inspired optimization algorithms.

This module provides implementations of optimization algorithms inspired by
social behaviors of humans and other social species.

Available Algorithms:
    - TeachingLearningOptimizer: Teaching-Learning Based Optimization (TLBO)

References:
    Rao, R. V., Savsani, V. J., & Vakharia, D. P. (2011). Teaching-learning-based
    optimization: A novel method for constrained mechanical design optimization
    problems. Computer-Aided Design, 43(3), 303-315.
"""

from __future__ import annotations

from opt.social_inspired.teaching_learning import TeachingLearningOptimizer


__all__: list[str] = ["TeachingLearningOptimizer"]
