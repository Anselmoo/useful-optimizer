"""Random Search optimizer.

This module implements pure random search: every candidate is drawn independently
and uniformly from the box-shaped search space, and only the best point seen so
far is kept. The method uses no gradient, no model and no memory of earlier
samples, which makes it the reference baseline against which other optimizers are
compared in benchmark statistics.

The RandomSearch class is the main class that implements the optimizer. It takes
the objective function, the lower and upper bounds of the search space, the
dimensionality and the sampling budget as input. The search method returns the
best solution found and its objective value.

Example usage:
    optimizer = RandomSearch(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=+32.768,
        max_iter=5000,
        seed=42,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

Attributes:
    population_size (int): The number of points sampled per iteration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class RandomSearch(AbstractOptimizer):
    r"""Random Search, the uniform sampling baseline for black-box optimization.

    Algorithm Metadata:
        | Property          | Value                                    |
        |-------------------|------------------------------------------|
        | Algorithm Name    | Random Search                            |
        | Acronym           | RS                                       |
        | Year Introduced   | 1958                                     |
        | Authors           | Brooks, S. H.                            |
        | Algorithm Class   | Classical                                |
        | Complexity        | $O(\text{population\_size} \times \text{max\_iter} \times \text{dim})$ |
        | Properties        | Derivative-free, Stochastic, Global search |
        | Implementation    | Python 3.10+                             |
        | COCO Compatible   | Yes                                      |

    Mathematical Formulation:
        Every candidate is an independent uniform sample from the search box
        $\Omega = [l, u]^d$:

            $$
            X_k \sim \mathcal{U}(\Omega), \quad k = 1, \dots, n
            $$

        where $n = \text{population\_size} \times \text{iterations}$ is the number of
        function evaluations. The returned solution is the best sample:

            $$
            \hat{x}_n = \arg\min_{k \le n} f(X_k)
            $$

        Probability of improvement. Let $p(y) = \lambda(\{x \in \Omega : f(x) < y\}) / \lambda(\Omega)$
        be the fraction of the box (with Lebesgue measure $\lambda$) that is better than a level $y$:

            $$
            P\left(\min_{k \le n} f(X_k) < y\right) = 1 - (1 - p(y))^n
            $$

        Expected best-of-$n$ value, with $F(y) = P(f(X) \le y)$ and $f_{\min}$ the
        minimum of $f$ on $\Omega$:

            $$
            \mathbb{E}\left[\min_{k \le n} f(X_k)\right]
            = f_{\min} + \int_{f_{\min}}^{\infty} (1 - F(y))^n \, dy
            $$

        where:
            - $d$ is the problem dimension (`dim`)
            - $l, u$ are `lower_bound` and `upper_bound`
            - $n$ is the total number of evaluations

        Constraint handling:
            - **Boundary conditions**: Samples are drawn inside the bounds, so no clipping is needed
            - **Feasibility enforcement**: Every candidate is feasible by construction

    Hyperparameters:
        | Parameter              | Default | BBOB Recommended | Description                    |
        |------------------------|---------|------------------|--------------------------------|
        | max_iter               | 1000    | 10000            | Number of sampling rounds      |
        | population_size        | 1       | 1-100            | Points sampled per round       |

        **Sensitivity Analysis**:
            - `max_iter` and `population_size` only matter through their product, the
              total evaluation budget. Splitting the budget differently changes the
              recorded history, not the distribution of the final result.
            - Recommended tuning: none, the method has no tunable behavior

    COCO/BBOB Benchmark Settings:
        **Search Space**:
            - Dimensions tested: `2, 3, 5, 10, 20, 40`
            - Bounds: Function-specific (typically `[-5, 5]` or `[-100, 100]`)
            - Instances: **15** per function (BBOB standard)

        **Evaluation Budget**:
            - Budget: $\text{dim} \times 10000$ function evaluations
            - Independent runs: **15** (for statistical significance)
            - Seeds: `0-14` (reproducibility requirement)

        **Performance Metrics**:
            - Target precision: `1e-8` (BBOB default)
            - Success rate at precision thresholds: `[1e-8, 1e-6, 1e-4, 1e-2]`
            - Expected Running Time (ERT) tracking

    Example:
        Basic usage with BBOB benchmark function:

        >>> from opt.classical.random_search import RandomSearch
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = RandomSearch(
        ...     func=shifted_ackley,
        ...     lower_bound=-32.768,
        ...     upper_bound=32.768,
        ...     dim=2,
        ...     max_iter=50,
        ...     seed=42,
        ... )
        >>> solution, fitness = optimizer.search()
        >>> isinstance(fitness, float)
        True
        >>> len(solution) == 2
        True
        >>> bool(np.all(solution >= -32.768) and np.all(solution <= 32.768))
        True
        >>> optimizer.n_evaluations == 50
        True

        The same seed reproduces the same result:

        >>> again = RandomSearch(
        ...     func=shifted_ackley,
        ...     lower_bound=-32.768,
        ...     upper_bound=32.768,
        ...     dim=2,
        ...     max_iter=50,
        ...     seed=42,
        ... )
        >>> solution2, fitness2 = again.search()
        >>> bool(np.array_equal(solution, solution2) and fitness == fitness2)
        True

        For COCO/BBOB benchmarking with full statistical analysis,
        see `benchmarks/run_benchmark_suite.py`.

    Args:
        func (Callable[[ndarray], float]): Objective function to minimize. Must accept
            numpy array and return scalar. BBOB functions available in
            `opt.benchmark.functions`.
        lower_bound (float): Lower bound of search space. BBOB typical: -5
            (most functions).
        upper_bound (float): Upper bound of search space. BBOB typical: 5
            (most functions).
        dim (int): Problem dimensionality. BBOB standard dimensions: 2, 3, 5, 10, 20, 40.
        max_iter (int, optional): Maximum number of sampling rounds. Must be at least 1.
            BBOB recommendation: 10000. Defaults to 1000.
        population_size (int, optional): Number of points drawn and evaluated per round.
            Must be at least 1. The total budget is `population_size * max_iter`
            evaluations. Defaults to 1.
        seed (int | None, optional): Random seed for reproducibility. BBOB requires
            seeds 0-14 for 15 runs. If None, generates random seed. Defaults to None.
        track_history (bool, optional): Whether to record the convergence history.
            Defaults to False.
        target_precision (float, optional): Target precision for early stopping, used
            together with `f_opt`. BBOB standard: 1e-8. Defaults to 1e-8.
        f_opt (float | None, optional): Known optimal value. When set, the search stops
            once `|best_fitness - f_opt| < target_precision`. If None, the whole budget
            is used. Defaults to None.

    Attributes:
        func (Callable[[ndarray], float]): The objective function being optimized.
        lower_bound (float): Lower search space boundary.
        upper_bound (float): Upper search space boundary.
        dim (int): Problem dimensionality.
        max_iter (int): Maximum number of sampling rounds.
        population_size (int): Number of points sampled per round.
        seed (int): **REQUIRED** Random seed for reproducibility (BBOB compliance).
        n_evaluations (int): Number of objective function evaluations performed.
        history (dict[str, list]): Convergence history when `track_history` is True.

    Methods:
        search() -> tuple[np.ndarray, float]:
            Execute optimization algorithm.

    Returns:
        tuple[np.ndarray, float]:
        Best solution found and its fitness value

    Raises:
        ValueError: If `max_iter` or `population_size` is smaller than 1.

    Notes:
        - Modifies self.history if track_history=True
        - Uses self.seed for all random number generation
        - BBOB: Returns the best sampled point after max_iter rounds or convergence

    References:
        [1] Brooks, S. H. (1958). "A discussion of random methods for seeking maxima."
            _Operations Research_, 6(2), 244-251.

        [2] Solis, F. J., & Wets, R. J.-B. (1981). "Minimization by random search techniques."
            _Mathematics of Operations Research_, 6(1), 19-30.

        [3] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
            "COCO: A platform for comparing continuous optimizers in a black-box setting."
            _Optimization Methods and Software_, 36(1), 114-144.
            https://doi.org/10.1080/10556788.2020.1808977

        **COCO Data Archive**:
            - Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
            - Algorithm data: Baseline for statistical comparison
            - Code repository: https://github.com/Anselmoo/useful-optimizer

        **Implementation**:
            - Original concept: Uniform random sampling for maximum seeking (Brooks, 1958)
            - This implementation: Batched uniform sampling with a single seeded generator

    See Also:
        HillClimbing: Greedy local search that exploits the best point found
            BBOB Comparison: HC faster on unimodal, RS unaffected by local optima

        SimulatedAnnealing: Local search that accepts worse points with decreasing probability
            BBOB Comparison: SA exploits structure, RS ignores it

        AbstractOptimizer: Base class for all optimizers
        opt.benchmark.functions: BBOB-compatible test functions

        Related BBOB Algorithm Classes:
            - Classical: HillClimbing, SimulatedAnnealing
            - Baseline: Random search is the reference for statistical tests

    Notes:
        **Computational Complexity**:
        - Time per iteration: $O(p \times d)$ for $p$ samples of dimension $d$, plus the
          cost of $p$ function evaluations
        - Space complexity: $O(p \times d)$ for the current batch
        - BBOB budget usage: _Uses exactly `population_size * iterations` evaluations_

        **BBOB Performance Characteristics**:
            - **Best function classes**: Low-dimensional, Highly multimodal with a wide basin
            - **Weak function classes**: High-dimensional, Ill-conditioned, Narrow optimum
            - Typical success rate at 1e-8 precision: **~0%** beyond dim=2
            - Expected Running Time (ERT): Grows exponentially with the dimension

        **Convergence Properties**:
            - Convergence rate: Sub-linear, the best value improves slowly with the budget
            - Local vs Global: Converges to the global optimum in probability
            - Premature convergence risk: **None** (samples are independent)

        **Reproducibility**:
            - **Deterministic**: Yes (given same seed)
            - **BBOB compliance**: seed parameter required for 15 independent runs
            - Initialization: Not applicable, every iteration samples afresh
            - RNG usage: One `numpy.random.default_rng(self.seed)` created before the loop

        **Implementation Details**:
            - Parallelization: Not used, although the samples are independent
            - Constraint handling: Samples lie inside the bounds
            - Numerical stability: No arithmetic beyond the scaling of uniform samples

        **Known Limitations**:
            - No use of earlier samples, so no exploitation of structure
            - Curse of dimensionality: the hit probability of a target region shrinks
              exponentially with the dimension
            - Intended as a baseline, not as a competitive optimizer

        **Version History**:
            - v0.1.0: Initial implementation as the benchmark baseline
    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = 1000,
        population_size: int = 1,
        seed: int | None = None,
        track_history: bool = False,  # noqa: FBT001, FBT002
        target_precision: float = 1e-8,
        f_opt: float | None = None,
    ) -> None:
        """Initialize the RandomSearch class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
            track_history=track_history,
            target_precision=target_precision,
            f_opt=f_opt,
        )

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the random search.

        Returns:
            tuple[np.ndarray, float]: The best solution found and its fitness value.

        Raises:
            ValueError: If `max_iter` or `population_size` is smaller than 1.
        """
        if self.max_iter < 1:
            msg = "max_iter must be at least 1"
            raise ValueError(msg)
        if self.population_size < 1:
            msg = "population_size must be at least 1"
            raise ValueError(msg)

        rng = np.random.default_rng(self.seed)

        best_solution = np.empty(self.dim)
        best_fitness = np.inf
        population = np.empty((self.population_size, self.dim))
        population_fitness = np.empty(self.population_size)

        for _ in range(self.max_iter):
            population = rng.uniform(
                self.lower_bound, self.upper_bound, (self.population_size, self.dim)
            )
            population_fitness = np.array([self.func(x) for x in population])

            index = int(np.argmin(population_fitness))
            if population_fitness[index] < best_fitness or not np.isfinite(
                best_fitness
            ):
                best_fitness = float(population_fitness[index])
                best_solution = population[index].copy()

            if self.track_history:
                self._record_history(
                    best_fitness=best_fitness,
                    best_solution=best_solution,
                    population_fitness=population_fitness,
                    population=population,
                )

            if self._check_convergence(best_fitness):
                break

        # Track final state
        if self.track_history:
            self._record_history(
                best_fitness=best_fitness,
                best_solution=best_solution,
                population_fitness=population_fitness,
                population=population,
            )
            self._finalize_history()

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(RandomSearch)
