"""Swarm intelligence optimization algorithms.

This module contains nature-inspired optimizers based on collective behavior of
decentralized, self-organized systems. Includes: Ant Colony, Artificial Fish Swarm,
Bat, Bee, Cat Swarm, Cuckoo Search, Firefly, Glowworm Swarm, Grey Wolf, Particle Swarm,
Squirrel Search, and Whale Optimization algorithms.
"""

from __future__ import annotations

from opt.swarm_intelligence.ant_colony import AntColony
from opt.swarm_intelligence.artificial_fish_swarm_algorithm import ArtificialFishSwarm
from opt.swarm_intelligence.bat_algorithm import BatAlgorithm
from opt.swarm_intelligence.bee_algorithm import BeeAlgorithm
from opt.swarm_intelligence.cat_swarm_optimization import CatSwarmOptimization
from opt.swarm_intelligence.cuckoo_search import CuckooSearch
from opt.swarm_intelligence.firefly_algorithm import FireflyAlgorithm
from opt.swarm_intelligence.glowworm_swarm_optimization import GlowwormSwarmOptimization
from opt.swarm_intelligence.grey_wolf_optimizer import GreyWolfOptimizer
from opt.swarm_intelligence.harris_hawks_optimization import HarrisHawksOptimizer
from opt.swarm_intelligence.particle_swarm import ParticleSwarm
from opt.swarm_intelligence.squirrel_search import SquirrelSearchAlgorithm
from opt.swarm_intelligence.whale_optimization_algorithm import (
    WhaleOptimizationAlgorithm,
)


__all__: list[str] = [
    "AntColony",
    "ArtificialFishSwarm",
    "BatAlgorithm",
    "BeeAlgorithm",
    "CatSwarmOptimization",
    "CuckooSearch",
    "FireflyAlgorithm",
    "GlowwormSwarmOptimization",
    "GreyWolfOptimizer",
    "HarrisHawksOptimizer",
    "ParticleSwarm",
    "SquirrelSearchAlgorithm",
    "WhaleOptimizationAlgorithm",
]
