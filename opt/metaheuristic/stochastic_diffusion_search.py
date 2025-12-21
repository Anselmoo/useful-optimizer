"""Stochastic Diffusion Search optimizer.

This module implements the Stochastic Diffusion Search optimizer, which is an
optimization algorithm that uses a population of agents to explore the search space and
find the optimal solution for a given objective function.

The main class in this module is `StochasticDiffusionSearch`, which represents the
optimizer. It takes the objective function, lower and upper bounds of the search space,
dimensionality of the search space, population size, maximum number of iterations,
and seed for the random number generator as input parameters.

The optimizer works by initializing a population of agents, where each agent has a
position in the search space and a score based on the objective function. The algorithm
then iteratively performs a test phase and a diffusion phase to update the positions of
the agents. After the specified number of iterations, the algorithm returns the best
solution found and its corresponding score.

Example usage:
    optimizer = StochasticDiffusionSearch(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=+32.768,
        population_size=100,
        max_iter=1000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class Agent:
    """Represents an agent in the stochastic diffusion search algorithm.

    An agent is a member of the population in the optimizer. It has a position in the search space,
    a score based on the objective function, and an active status indicating whether it is currently
    participating in the diffusion process.

    Attributes:
        position (ndarray): The position of the agent in the search space.
        score (float): The score of the agent's current position.
        active (bool): Indicates whether the agent is active or not.

    Methods:
        __init__: Initializes the Agent class.

    """

    def __init__(
        self,
        dim: int,
        lower_bound: float,
        upper_bound: float,
        func: Callable[[ndarray], float],
    ) -> None:
        """Initialize the Agent class.

        Args:
            dim (int): The dimensionality of the search space.
            lower_bound (float): The lower bound of the search space.
            upper_bound (float): The upper bound of the search space.
            func (Callable[[ndarray], float]): The objective function to be optimized.

        """
        self.position = np.random.default_rng(1).uniform(lower_bound, upper_bound, dim)
        self.score = func(self.position)
        self.active = False


class StochasticDiffusionSearch(AbstractOptimizer):
    """Stochastic Diffusion Search optimizer.

    This optimizer uses the Stochastic Diffusion Search algorithm to find the optimal solution
    for a given function within the specified bounds.

    Args:
        func (Callable[[ndarray], float]): The objective function to be optimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        population_size (int, optional): The size of the population. Defaults to 100.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        seed (int | None, optional): The seed for the random number generator. Defaults to None.

    Attributes:
        population (List[Agent]): The population of agents.

    Returns:
        tuple[np.ndarray, float]: The best solution found and its corresponding score.
    """

    def search(self) -> tuple[np.ndarray, float]:
        """Run the Stochastic Diffusion Search algorithm.

        Returns:
            tuple[np.ndarray, float]: The best solution found and its corresponding score.
        """
        self.population = [
            Agent(self.dim, self.lower_bound, self.upper_bound, self.func)
            for _ in range(self.population_size)
        ]

        for _ in range(self.max_iter):
            self.seed += 1
            for agent in self.population:
                # Test phase
                if self.func(agent.position) < np.random.default_rng(self.seed).uniform(
                    self.lower_bound, self.upper_bound
                ):
                    agent.active = True
                else:
                    agent.active = False

                # Diffusion phase
                if not agent.active:
                    other_agent = np.random.default_rng(self.seed).choice(
                        np.array(self.population)
                    )
                    if other_agent.active:
                        agent.position = other_agent.position
                    else:
                        agent.position = np.random.default_rng(self.seed).uniform(
                            self.lower_bound, self.upper_bound, self.dim
                        )
                    agent.score = self.func(agent.position)

        best_agent = min(self.population, key=lambda agent: agent.score)
        return best_agent.position, best_agent.score


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(StochasticDiffusionSearch)
