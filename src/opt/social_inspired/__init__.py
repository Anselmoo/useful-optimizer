"""Social-inspired optimization algorithms.

This module provides implementations of optimization algorithms inspired by
social behaviors of humans and other social species.

Available Algorithms:
    - TeachingLearningOptimizer: Teaching-Learning Based Optimization (TLBO)
    - PoliticalOptimizer: Political Optimizer based on election processes
    - SocialGroupOptimizer: Social Group Optimization (SGO)
    - SoccerLeagueOptimizer: Soccer League Competition Algorithm

References:
    Rao, R. V., Savsani, V. J., & Vakharia, D. P. (2011). Teaching-learning-based
    optimization: A novel method for constrained mechanical design optimization
    problems. Computer-Aided Design, 43(3), 303-315.
"""

from __future__ import annotations

from opt.social_inspired.political_optimizer import PoliticalOptimizer
from opt.social_inspired.soccer_league_optimizer import SoccerLeagueOptimizer
from opt.social_inspired.social_group_optimizer import SocialGroupOptimizer
from opt.social_inspired.teaching_learning import TeachingLearningOptimizer


__all__: list[str] = [
    "PoliticalOptimizer",
    "SoccerLeagueOptimizer",
    "SocialGroupOptimizer",
    "TeachingLearningOptimizer",
]
