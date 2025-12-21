"""Black Widow Optimization Algorithm.

Implementation based on:
Hayyolalam, V. & Kazem, A.A.P. (2020).
Black Widow Optimization Algorithm: A novel meta-heuristic approach
for solving engineering optimization problems.
Engineering Applications of Artificial Intelligence, 87, 103249.

The algorithm mimics the mating behavior of black widow spiders, including
cannibalistic behaviors where females may eat males after mating.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm parameters
_PP = 0.6  # Procreation probability
_CR = 0.44  # Cannibalism rate
_PM = 0.4  # Mutation probability


class BlackWidowOptimizer(AbstractOptimizer):
    """Black Widow Optimization Algorithm.

    Simulates the mating and cannibalistic behavior of black widow spiders.
    The algorithm includes:
    - Procreation: offspring generation from parent pairs
    - Cannibalism: elimination of weak solutions
    - Mutation: random exploration

    Args:
        func: Objective function to minimize.
        lower_bound: Lower bound for the search space.
        upper_bound: Upper bound for the search space.
        dim: Dimensionality of the search space.
        max_iter: Maximum number of iterations.
        population_size: Number of spiders in the population.
        pp: Procreation probability. Default 0.6.
        cr: Cannibalism rate. Default 0.44.
        pm: Mutation probability. Default 0.4.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int,
        population_size: int = 30,
        pp: float = _PP,
        cr: float = _CR,
        pm: float = _PM,
    ) -> None:
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size
        self.pp = pp
        self.cr = cr
        self.pm = pm

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Black Widow Optimization Algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population
        population = np.random.uniform(
            self.lower_bound,
            self.upper_bound,
            (self.population_size, self.dim),
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(ind) for ind in population])

        # Find best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for _ in range(self.max_iter):
            # Sort population by fitness
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]

            # Procreation phase
            offspring = []
            n_pairs = int(self.population_size * self.pp) // 2

            for i in range(n_pairs):
                # Select parents (adjacent in sorted list = similar fitness)
                parent1 = population[2 * i]
                parent2 = population[2 * i + 1]

                # Generate offspring using crossover
                alpha = np.random.rand(self.dim)
                child1 = alpha * parent1 + (1 - alpha) * parent2
                child2 = (1 - alpha) * parent1 + alpha * parent2

                offspring.extend([child1, child2])

            offspring = np.array(offspring) if offspring else np.array([]).reshape(0, self.dim)

            # Apply boundary constraints to offspring
            if len(offspring) > 0:
                offspring = np.clip(offspring, self.lower_bound, self.upper_bound)
                offspring_fitness = np.array([self.func(ind) for ind in offspring])

                # Sexual cannibalism - mother eats father if she's fitter
                # (implicitly done through selection later)

                # Sibling cannibalism - keep only best offspring per pair
                if len(offspring) >= 2:
                    filtered_offspring = []
                    filtered_fitness = []
                    for i in range(0, len(offspring) - 1, 2):
                        if offspring_fitness[i] < offspring_fitness[i + 1]:
                            filtered_offspring.append(offspring[i])
                            filtered_fitness.append(offspring_fitness[i])
                        else:
                            filtered_offspring.append(offspring[i + 1])
                            filtered_fitness.append(offspring_fitness[i + 1])

                    offspring = np.array(filtered_offspring)
                    offspring_fitness = np.array(filtered_fitness)

            # Combine population with offspring
            if len(offspring) > 0:
                combined_pop = np.vstack([population, offspring])
                combined_fitness = np.concatenate([fitness, offspring_fitness])
            else:
                combined_pop = population
                combined_fitness = fitness

            # Cannibalism - keep only best solutions
            n_survivors = int(self.population_size * (1 - self.cr))
            n_survivors = max(n_survivors, 5)  # Keep at least 5

            sorted_idx = np.argsort(combined_fitness)[:n_survivors]
            survivors = combined_pop[sorted_idx]
            survivor_fitness = combined_fitness[sorted_idx]

            # Mutation phase - add mutants to fill population
            n_mutants = self.population_size - n_survivors
            mutants = []
            mutant_fitness_list = []

            for _ in range(n_mutants):
                # Select a random survivor and mutate
                idx = np.random.randint(n_survivors)
                mutant = survivors[idx].copy()

                # Apply mutation
                if np.random.rand() < self.pm:
                    # Gaussian mutation
                    sigma = (self.upper_bound - self.lower_bound) / 6
                    mutant += np.random.randn(self.dim) * sigma
                else:
                    # Random reinitialization
                    mutant = np.random.uniform(
                        self.lower_bound, self.upper_bound, self.dim
                    )

                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                mutants.append(mutant)
                mutant_fitness_list.append(self.func(mutant))

            mutants = np.array(mutants)
            mutant_fitness = np.array(mutant_fitness_list)

            # Form new population
            population = np.vstack([survivors, mutants])
            fitness = np.concatenate([survivor_fitness, mutant_fitness])

            # Update global best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_solution = population[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.benchmark.functions import shifted_ackley

    optimizer = BlackWidowOptimizer(
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
