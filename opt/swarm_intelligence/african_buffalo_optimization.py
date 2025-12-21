"""African Buffalo Optimization Algorithm.

Implementation based on:
Odili, J.B., Kahar, M.N.M. & Anwar, S. (2015).
African Buffalo Optimization: A Swarm-Intelligence Technique.
Procedia Computer Science, 76, 443-448.

The algorithm mimics the migratory and herding behavior of African buffalos,
using two key equations: the buffalo's movement toward the best location and
its tendency to explore new areas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Learning parameters
_LP1 = 0.6  # Learning parameter 1 (exploitation)
_LP2 = 0.4  # Learning parameter 2 (exploration)


class AfricanBuffaloOptimizer(AbstractOptimizer):
    """African Buffalo Optimization algorithm.

    Simulates the cooperative behavior of African buffalo herds in finding
    optimal grazing locations. The algorithm uses two types of calls:
    - Warning call (waaa): Guides buffalos toward the best location
    - Movement call (maaa): Encourages exploration of new locations

    Args:
        func: Objective function to minimize.
        lower_bound: Lower bound for the search space.
        upper_bound: Upper bound for the search space.
        dim: Dimensionality of the search space.
        max_iter: Maximum number of iterations.
        population_size: Number of buffalos in the herd.
        lp1: Learning parameter 1 for exploitation (waaa). Default 0.6.
        lp2: Learning parameter 2 for exploration (maaa). Default 0.4.


    Example:
        >>> from opt.swarm_intelligence.african_buffalo_optimization import AfricanBuffaloOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = AfricanBuffaloOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = AfricanBuffaloOptimizer(
        ...     func=shifted_ackley, dim=2,
        ...     lower_bound=-2.768, upper_bound=2.768,
        ...     max_iter=10, seed=42
        ... )
        >>> _, fitness = optimizer.search()
        >>> isinstance(float(fitness), float)
        True
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int,
        population_size: int = 30,
        lp1: float = _LP1,
        lp2: float = _LP2,
    ) -> None:
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size
        self.lp1 = lp1
        self.lp2 = lp2

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the African Buffalo Optimization algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize buffalo positions
        positions = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Initialize fitness and best positions
        fitness = np.array([self.func(pos) for pos in positions])
        personal_best = positions.copy()
        personal_best_fitness = fitness.copy()

        # Global best
        best_idx = np.argmin(fitness)
        global_best = positions[best_idx].copy()
        global_best_fitness = fitness[best_idx]

        # Initialize exploration memory (maaa)
        exploration_memory = np.zeros((self.population_size, self.dim))

        for iteration in range(self.max_iter):
            for i in range(self.population_size):
                # Update exploration memory (maaa equation)
                r1, r2 = np.random.rand(2)
                exploration_memory[i] = (
                    exploration_memory[i]
                    + self.lp1 * r1 * (global_best - positions[i])
                    + self.lp2 * r2 * (personal_best[i] - positions[i])
                )

                # Update position (waaa equation)
                positions[i] = (positions[i] + exploration_memory[i]) / 2.0

                # Boundary handling
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

                # Evaluate new position
                new_fitness = self.func(positions[i])
                fitness[i] = new_fitness

                # Update personal best
                if new_fitness < personal_best_fitness[i]:
                    personal_best[i] = positions[i].copy()
                    personal_best_fitness[i] = new_fitness

                    # Update global best
                    if new_fitness < global_best_fitness:
                        global_best = positions[i].copy()
                        global_best_fitness = new_fitness

            # Adaptive restart for stagnant buffalos
            stagnation_threshold = 0.3
            if iteration > 0 and iteration % 50 == 0:
                for i in range(self.population_size):
                    if np.random.rand() < stagnation_threshold:
                        # Random restart
                        positions[i] = np.random.uniform(
                            self.lower_bound, self.upper_bound, self.dim
                        )
                        exploration_memory[i] = np.zeros(self.dim)

        return global_best, global_best_fitness


if __name__ == "__main__":
    from opt.benchmark.functions import shifted_ackley

    optimizer = AfricanBuffaloOptimizer(
        func=shifted_ackley,
        lower_bound=-2.768,
        upper_bound=2.768,
        dim=2,
        max_iter=100,
        population_size=30,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
