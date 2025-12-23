from __future__ import annotations

from typing import Callable
import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


class BrainStormOptimizer(AbstractOptimizer):
    """
    Brain Storm Optimization (BSO).

    Social-inspired population-based optimizer that generates new solutions
    by perturbing or combining elite ideas (cluster representatives).

    This implementation uses top-k individuals as cluster centers instead of
    explicit clustering, which is a common simplified variant.

    Reference:
    Y. Shi, "Brain Storm Optimization Algorithm",
    Advances in Swarm Intelligence, 2011.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        dim: int,
        lower_bound: float,
        upper_bound: float,
        max_iter: int = 300,
        population_size: int = 30,
        n_clusters: int = 5,
        step_size: float = 0.15,
        seed: int | None = None,
    ):
        super().__init__(
            func=func,
            dim=dim,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            max_iter=max_iter,
        )

        self.population_size = population_size
        self.n_clusters = n_clusters
        self.step_size = step_size

        self.seed = 0 if seed is None else seed
        self.rng = np.random.default_rng(self.seed)

    def search(self) -> tuple[np.ndarray, float]:
        dim = self.dim
        lower = np.full(dim, self.lower_bound)
        upper = np.full(dim, self.upper_bound)

        pop = self.rng.uniform(lower, upper, size=(self.population_size, dim))
        fitness = np.apply_along_axis(self.func, 1, pop)

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        for t in range(self.max_iter):
            order = np.argsort(fitness)
            pop = pop[order]
            fitness = fitness[order]

            centers = pop[: self.n_clusters]

            noise_scale = self.step_size * (1.0 - t / self.max_iter)

            for i in range(self.population_size):
                if self.rng.random() < 0.5:
                    center = centers[self.rng.integers(self.n_clusters)]
                    candidate = center + noise_scale * self.rng.normal(size=dim)
                else:
                    c1, c2 = centers[
                        self.rng.choice(self.n_clusters, size=2, replace=False)
                    ]
                    alpha = self.rng.random()
                    candidate = alpha * c1 + (1.0 - alpha) * c2
                    candidate += noise_scale * self.rng.normal(size=dim)

                candidate = np.clip(candidate, lower, upper)
                cand_f = self.func(candidate)

                if cand_f < fitness[i]:
                    pop[i] = candidate
                    fitness[i] = cand_f

                    if cand_f < best_f:
                        best_f = cand_f
                        best_x = candidate.copy()

        return best_x, best_f
