"""Swarm intelligence optimization algorithms.

This module contains nature-inspired optimizers based on collective behavior of
decentralized, self-organized systems. Includes: Ant Colony, Artificial Fish Swarm,
Bat, Bee, Cat Swarm, Cuckoo Search, Firefly, Glowworm Swarm, Grey Wolf, Particle Swarm,
Squirrel Search, and Whale Optimization algorithms.
"""

from __future__ import annotations

from opt.swarm_intelligence.african_vultures_optimizer import AfricanVulturesOptimizer
from opt.swarm_intelligence.ant_colony import AntColony
from opt.swarm_intelligence.ant_lion_optimizer import AntLionOptimizer
from opt.swarm_intelligence.aquila_optimizer import AquilaOptimizer
from opt.swarm_intelligence.artificial_fish_swarm_algorithm import ArtificialFishSwarm
from opt.swarm_intelligence.artificial_gorilla_troops import (
    ArtificialGorillaTroopsOptimizer,
)
from opt.swarm_intelligence.bat_algorithm import BatAlgorithm
from opt.swarm_intelligence.bee_algorithm import BeeAlgorithm
from opt.swarm_intelligence.cat_swarm_optimization import CatSwarmOptimization
from opt.swarm_intelligence.chimp_optimization import ChimpOptimizationAlgorithm
from opt.swarm_intelligence.cuckoo_search import CuckooSearch
from opt.swarm_intelligence.dragonfly_algorithm import DragonflyOptimizer
from opt.swarm_intelligence.emperor_penguin import EmperorPenguinOptimizer
from opt.swarm_intelligence.firefly_algorithm import FireflyAlgorithm
from opt.swarm_intelligence.flower_pollination import FlowerPollinationAlgorithm
from opt.swarm_intelligence.glowworm_swarm_optimization import GlowwormSwarmOptimization
from opt.swarm_intelligence.golden_eagle import GoldenEagleOptimizer
from opt.swarm_intelligence.grasshopper_optimization import GrasshopperOptimizer
from opt.swarm_intelligence.grey_wolf_optimizer import GreyWolfOptimizer
from opt.swarm_intelligence.harris_hawks_optimization import HarrisHawksOptimizer
from opt.swarm_intelligence.manta_ray import MantaRayForagingOptimization
from opt.swarm_intelligence.marine_predators_algorithm import MarinePredatorsOptimizer
from opt.swarm_intelligence.moth_flame_optimization import MothFlameOptimizer
from opt.swarm_intelligence.particle_swarm import ParticleSwarm
from opt.swarm_intelligence.pathfinder import PathfinderAlgorithm
from opt.swarm_intelligence.reptile_search import ReptileSearchAlgorithm
from opt.swarm_intelligence.salp_swarm_algorithm import SalpSwarmOptimizer
from opt.swarm_intelligence.seagull_optimization import SeagullOptimizationAlgorithm
from opt.swarm_intelligence.slime_mould import SlimeMouldAlgorithm
from opt.swarm_intelligence.spotted_hyena import SpottedHyenaOptimizer
from opt.swarm_intelligence.squirrel_search import SquirrelSearchAlgorithm
from opt.swarm_intelligence.tunicate_swarm import TunicateSwarmAlgorithm
from opt.swarm_intelligence.whale_optimization_algorithm import (
    WhaleOptimizationAlgorithm,
)


__all__: list[str] = [
    "AfricanVulturesOptimizer",
    "AntColony",
    "AntLionOptimizer",
    "AquilaOptimizer",
    "ArtificialFishSwarm",
    "ArtificialGorillaTroopsOptimizer",
    "BatAlgorithm",
    "BeeAlgorithm",
    "CatSwarmOptimization",
    "ChimpOptimizationAlgorithm",
    "CuckooSearch",
    "DragonflyOptimizer",
    "EmperorPenguinOptimizer",
    "FireflyAlgorithm",
    "FlowerPollinationAlgorithm",
    "GlowwormSwarmOptimization",
    "GoldenEagleOptimizer",
    "GrasshopperOptimizer",
    "GreyWolfOptimizer",
    "HarrisHawksOptimizer",
    "MantaRayForagingOptimization",
    "MarinePredatorsOptimizer",
    "MothFlameOptimizer",
    "ParticleSwarm",
    "PathfinderAlgorithm",
    "ReptileSearchAlgorithm",
    "SalpSwarmOptimizer",
    "SeagullOptimizationAlgorithm",
    "SlimeMouldAlgorithm",
    "SpottedHyenaOptimizer",
    "SquirrelSearchAlgorithm",
    "TunicateSwarmAlgorithm",
    "WhaleOptimizationAlgorithm",
]
