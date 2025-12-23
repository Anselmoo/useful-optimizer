"""Covariance Matrix Adaptation Evolution Strategy.

This module implements the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) algorithm,
which is a derivative-free optimization method that uses an evolutionary strategy to search for
the optimal solution. It adapts the covariance matrix of the multivariate Gaussian distribution
to guide the search towards promising regions of the search space.

The CMA-ES algorithm is implemented in the `CMAESAlgorithm` class, which inherits from the
`AbstractOptimizer` class. The `CMAESAlgorithm` class provides a `search` method that runs the
CMA-ES algorithm to search for the optimal solution.

Example usage:
    optimizer = CMAESAlgorithm(
        func=shifted_ackley,
        dim=2,
        lower_bound=-12.768,
        upper_bound=12.768,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.linalg import sqrtm

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class CMAESAlgorithm(AbstractOptimizer):
    r"""FIXME: [Algorithm Full Name] ([ACRONYM]) optimization algorithm.

    Algorithm Metadata:
        | Property          | Value                                    |
        |-------------------|------------------------------------------|
        | Algorithm Name    | FIXME: [Full algorithm name]             |
        | Acronym           | FIXME: [SHORT]                           |
        | Year Introduced   | FIXME: [YYYY]                            |
        | Authors           | FIXME: [Last, First; ...]                |
        | Algorithm Class   | Evolutionary |
        | Complexity        | FIXME: O([expression])                   |
        | Properties        | FIXME: [Population-based, ...]           |
        | Implementation    | Python 3.10+                             |
        | COCO Compatible   | Yes                                      |

    Mathematical Formulation:
        FIXME: Core update equation:

            $$
            x_{t+1} = x_t + v_t
            $$

        where:
            - $x_t$ is the position at iteration $t$
            - $v_t$ is the velocity/step at iteration $t$
            - FIXME: Additional variable definitions...

        Constraint handling:
            - **Boundary conditions**: FIXME: [clamping/reflection/periodic]
            - **Feasibility enforcement**: FIXME: [description]

    Hyperparameters:
        | Parameter              | Default | BBOB Recommended | Description                    |
        |------------------------|---------|------------------|--------------------------------|
        | population_size        | 100     | 10*dim           | Number of individuals          |
        | max_iter               | 1000    | 10000            | Maximum iterations             |
        | FIXME: [param_name]    | [val]   | [bbob_val]       | [description]                  |

        **Sensitivity Analysis**:
            - FIXME: `[param_name]`: **[High/Medium/Low]** impact on convergence
            - Recommended tuning ranges: FIXME: $\text{[param]} \in [\text{min}, \text{max}]$

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

        >>> from opt.evolutionary.cma_es import CMAESAlgorithm
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = CMAESAlgorithm(
        ...     func=shifted_ackley,
        ...     lower_bound=-2.768,
        ...     upper_bound=2.768,
        ...     dim=2,
        ...     max_iter=100,
        ...     seed=42,  # Required for reproducibility
        ... )
        >>> solution, fitness = optimizer.search()
        >>> isinstance(fitness, float) and fitness >= 0
        True

        COCO benchmark example:

        >>> from opt.benchmark.functions import sphere
        >>> optimizer = CMAESAlgorithm(
        ...     func=sphere, lower_bound=-5, upper_bound=5, dim=10, max_iter=10000, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> len(solution) == 10
        True

    Args:
        FIXME: Document all parameters with BBOB guidance.
        Detected parameters from __init__ signature: func, dim, lower_bound, upper_bound, population_size, max_iter, sigma_init, epsilon, seed

        Common parameters (adjust based on actual signature):
        func (Callable[[ndarray], float]): Objective function to minimize. Must accept
            numpy array and return scalar. BBOB functions available in
            `opt.benchmark.functions`.
        lower_bound (float): Lower bound of search space. BBOB typical: -5
            (most functions).
        upper_bound (float): Upper bound of search space. BBOB typical: 5
            (most functions).
        dim (int): Problem dimensionality. BBOB standard dimensions: 2, 3, 5, 10, 20, 40.
        max_iter (int, optional): Maximum iterations. BBOB recommendation: 10000 for
            complete evaluation. Defaults to 1000.
        seed (int | None, optional): Random seed for reproducibility. BBOB requires
            seeds 0-14 for 15 runs. If None, generates random seed. Defaults to None.
        population_size (int, optional): Population size. BBOB recommendation: 10*dim
            for population-based methods. Defaults to 100. (Only for population-based
            algorithms)
        track_history (bool, optional): Enable convergence history tracking for BBOB
            post-processing. Defaults to False.
        FIXME: [algorithm_specific_params] ([type], optional): FIXME: Document any
            algorithm-specific parameters not listed above. Defaults to [value].

    Attributes:
        func (Callable[[ndarray], float]): The objective function being optimized.
        lower_bound (float): Lower search space boundary.
        upper_bound (float): Upper search space boundary.
        dim (int): Problem dimensionality.
        max_iter (int): Maximum number of iterations.
        seed (int): **REQUIRED** Random seed for reproducibility (BBOB compliance).
        population_size (int): Number of individuals in population.
        track_history (bool): Whether convergence history is tracked.
        history (dict[str, list]): Optimization history if track_history=True. Contains:
            - 'best_fitness': list[float] - Best fitness per iteration
            - 'best_solution': list[ndarray] - Best solution per iteration
            - 'population_fitness': list[ndarray] - All fitness values
            - 'population': list[ndarray] - All solutions
        FIXME: [algorithm_specific_attrs] ([type]): FIXME: [Description]

    Methods:
        search() -> tuple[np.ndarray, float]:
            Execute optimization algorithm.

    Returns:
                tuple[np.ndarray, float]:
                    Best solution found and its fitness value

    Raises:
                ValueError:
                    If search space is invalid or function evaluation fails.

    Notes:
                - Modifies self.history if track_history=True
                - Uses self.seed for all random number generation
                - BBOB: Returns final best solution after max_iter or convergence

    References:
        FIXME: [1] Author1, A., Author2, B. (YEAR). "Algorithm Name: Description."
            _Journal Name_, Volume(Issue), Pages.
            https://doi.org/10.xxxx/xxxxx

        [2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
            "COCO: A platform for comparing continuous optimizers in a black-box setting."
            _Optimization Methods and Software_, 36(1), 114-144.
            https://doi.org/10.1080/10556788.2020.1808977

        **COCO Data Archive**:
            - Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
            - FIXME: Algorithm data: [URL to algorithm-specific COCO results if available]
            - Code repository: https://github.com/Anselmoo/useful-optimizer

        **Implementation**:
            - FIXME: Original paper code: [URL if different from this implementation]
            - This implementation: Based on [1] with modifications for BBOB compliance

    See Also:
        FIXME: [RelatedAlgorithm1]: Similar algorithm with [key difference]
            BBOB Comparison: [Brief performance notes on sphere/rosenbrock/ackley]

        FIXME: [RelatedAlgorithm2]: [Relationship description]
            BBOB Comparison: Generally [faster/slower/more robust] on [function classes]

        AbstractOptimizer: Base class for all optimizers
        opt.benchmark.functions: BBOB-compatible test functions

        Related BBOB Algorithm Classes:
            - Evolutionary: GeneticAlgorithm, DifferentialEvolution
            - Swarm: ParticleSwarm, AntColony
            - Gradient: AdamW, SGDMomentum

    Notes:
        **Computational Complexity**:
            - Time per iteration: FIXME: $O(\text{[expression]})$
            - Space complexity: FIXME: $O(\text{[expression]})$
            - BBOB budget usage: FIXME: _[Typical percentage of dim*10000 budget needed]_

        **BBOB Performance Characteristics**:
            - **Best function classes**: FIXME: [Unimodal/Multimodal/Ill-conditioned/...]
            - **Weak function classes**: FIXME: [Function types where algorithm struggles]
            - Typical success rate at 1e-8 precision: FIXME: **[X]%** (dim=5)
            - Expected Running Time (ERT): FIXME: [Comparative notes vs other algorithms]

        **Convergence Properties**:
            - Convergence rate: FIXME: [Linear/Quadratic/Exponential]
            - Local vs Global: FIXME: [Tendency for local/global optima]
            - Premature convergence risk: FIXME: **[High/Medium/Low]**

        **Reproducibility**:
            - **Deterministic**: FIXME: [Yes/No] - Same seed guarantees same results
            - **BBOB compliance**: seed parameter required for 15 independent runs
            - Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
            - RNG usage: `numpy.random.default_rng(self.seed)` throughout

        **Implementation Details**:
            - Parallelization: FIXME: [Not supported/Supported via `[method]`]
            - Constraint handling: FIXME: [Clamping to bounds/Penalty/Repair]
            - Numerical stability: FIXME: [Considerations for floating-point arithmetic]

        **Known Limitations**:
            - FIXME: [Any known issues or limitations specific to this implementation]
            - FIXME: BBOB known issues: [Any BBOB-specific challenges]

        **Version History**:
            - v0.1.0: Initial implementation
            - FIXME: [vX.X.X]: [Changes relevant to BBOB compliance]
    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        dim: int,
        lower_bound: float,
        upper_bound: float,
        population_size: int = 100,
        max_iter: int = 1000,
        sigma_init: float = 0.5,
        epsilon: float = 1e-9,
        seed: int | None = None,
    ) -> None:
        """Initialize the CMAESAlgorithm class."""
        super().__init__(
            func=func,
            dim=dim,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            population_size=population_size,
            max_iter=max_iter,
            seed=seed,
        )
        self.sigma = sigma_init
        self.epsilon = epsilon

    def search(self) -> tuple[np.ndarray, float]:
        """Run the CMA-ES algorithm to search for the optimal solution.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the best solution found and its corresponding fitness value.
        """
        # Initialize mean and covariance matrix
        rng = np.random.default_rng(self.seed)
        mean = rng.uniform(self.lower_bound, self.upper_bound, self.dim)
        cov = np.eye(self.dim)

        # Initialize evolution paths
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)

        # Other parameters
        mu = self.population_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        cc = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        cs = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
        cmu = min(
            1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2) ** 2 + mu_eff)
        )
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + cs

        h_sigma_threshold = 1.4
        regularization = 1e-8  # Small regularization for numerical stability

        for iteration in range(self.max_iter):
            # Sample new solutions
            try:
                # Add regularization to ensure positive definite covariance
                cov_regularized = cov + regularization * np.eye(self.dim)
                solutions = rng.multivariate_normal(
                    mean, self.sigma**2 * cov_regularized, self.population_size
                )
            except np.linalg.LinAlgError:
                # If sampling fails, reinitialize covariance matrix
                cov = np.eye(self.dim)
                solutions = rng.multivariate_normal(
                    mean, self.sigma**2 * cov, self.population_size
                )

            # Evaluate solutions
            fitness = np.apply_along_axis(self.func, 1, solutions)

            # Sort by fitness and compute weighted mean into center
            indices = np.argsort(fitness)
            mean_old = mean
            mean = np.dot(weights, solutions[indices[:mu]])

            # Update evolution paths
            try:
                cov_sqrt_inv = np.linalg.inv(sqrtm(cov_regularized))
            except np.linalg.LinAlgError:
                # Fallback: use regularized inverse
                cov_sqrt_inv = np.linalg.inv(
                    sqrtm(cov + regularization * 10 * np.eye(self.dim))
                )

            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * np.dot(
                cov_sqrt_inv, (mean - mean_old) / self.sigma
            )
            h_sigma = (
                np.linalg.norm(p_sigma)
                / np.sqrt(1 - (1 - cs) ** (2 * (iteration + 1)))
                / np.sqrt(self.dim)
                < h_sigma_threshold
            )
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * (
                mean - mean_old
            ) / self.sigma

            # Adapt covariance matrix
            artmp = (1 / self.sigma) * (solutions[indices[:mu]] - mean_old)
            cov = (
                (1 - c1 - cmu) * cov
                + c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * cc * (2 - cc) * cov)
                + cmu * np.dot(artmp.T, np.dot(np.diag(weights), artmp))
            )

            # Ensure covariance matrix remains symmetric
            cov = (cov + cov.T) / 2

            # Adapt step size
            self.sigma *= np.exp(
                (cs / damps) * (np.linalg.norm(p_sigma) / np.sqrt(self.dim) - 1)
            )

            # Prevent sigma from becoming too small
            self.sigma = max(self.sigma, self.epsilon)

        best_solution = mean
        best_fitness = self.func(best_solution)
        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(CMAESAlgorithm)
