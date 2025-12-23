"""Parzen Tree Estimator optimizer.

The Parzen Tree Estimator optimizer is an algorithm that uses the Parzen Tree Estimator
technique to search for the optimal solution of a given function within a specified
search space. It is particularly useful for optimization problems where the objective
function is expensive to evaluate.

The Parzen Tree Estimator algorithm works by maintaining a population of
hyperparameters and their corresponding scores. It segments the population into two
distributions based on the scores and fits Gaussian kernel density estimators to each
distribution. It then samples hyperparameters from the low score distribution and
selects the hyperparameters with the highest score difference or ratio between the
low and high score distributions. This process is iteratively repeated to search
for the optimal solution.

This implementation of the Parzen Tree Estimator optimizer provides a flexible and
customizable framework for solving optimization problems. It allows users to specify
the objective function, search space, population size, maximum number of iterations,
selection strategy, and other parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sklearn.neighbors import KernelDensity

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class ParzenTreeEstimator(AbstractOptimizer):
    r"""FIXME: [Algorithm Full Name] ([ACRONYM]) optimization algorithm.

    Algorithm Metadata:
        | Property          | Value                                    |
        |-------------------|------------------------------------------|
        | Algorithm Name    | FIXME: [Full algorithm name]             |
        | Acronym           | FIXME: [SHORT]                           |
        | Year Introduced   | FIXME: [YYYY]                            |
        | Authors           | FIXME: [Last, First; ...]                |
        | Algorithm Class   | Probabilistic |
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

        >>> from opt.probabilistic.parzen_tree_stimator import ParzenTreeEstimator
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = ParzenTreeEstimator(
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
        >>> optimizer = ParzenTreeEstimator(
        ...     func=sphere, lower_bound=-5, upper_bound=5, dim=10, max_iter=10000, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> len(solution) == 10
        True

    Args:
        FIXME: Document all parameters with BBOB guidance.
        Detected parameters from __init__ signature: func, dim, lower_bound, upper_bound, population_size, max_iter, gamma, bandwidth, n_samples, selection_strategy, seed

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
        gamma: float = 0.15,
        bandwidth: float = 0.2,
        n_samples: int | None = None,
        selection_strategy: str = "difference",
        seed: int | None = None,
    ) -> None:
        """Initialize the ParzenTreeEstimator class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        self.gamma = gamma
        self.bandwidth = bandwidth
        if n_samples is None:
            n_samples = population_size
        self.n_samples = n_samples
        self.population = np.empty((population_size, dim))
        self.scores = np.inf * np.ones(population_size)
        if selection_strategy == "difference":
            self.sample_select = lambda l_score, g_score: np.argmax(l_score - g_score)
        elif selection_strategy == "ratio":
            self.sample_select = lambda l_score, g_score: np.argmax(g_score / l_score)
        else:
            msg = f"Invalid selection strategy: {selection_strategy}"
            raise ValueError(msg)

    def initialize_population(self) -> None:
        """Initializes the population of hyperparameters.

        This method generates a random population of hyperparameters within the specified search space.
        """
        self.population = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        self.scores = np.array(
            [self.func(individual) for individual in self.population]
        )

    def segment_distributions(self) -> tuple[KernelDensity, KernelDensity]:
        """Segments the distributions based on the scores.

        This method segments the population into two distributions based on the scores of the hyperparameters.
        It fits a Gaussian kernel density estimator to each distribution.

        Returns:
            Tuple[KernelDensity, KernelDensity]: The fitted kernel density estimators for the low and high score distributions.
        """
        cut = np.quantile(self.scores, self.gamma)
        l_x = self.population[self.scores < cut]
        g_x = self.population[self.scores >= cut]
        l_kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth).fit(l_x)
        g_kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth).fit(g_x)
        return l_kde, g_kde

    def choose_next_hps(self, l_kde: KernelDensity, g_kde: KernelDensity) -> np.ndarray:
        """Choose the next set of hyperparameters using the KDE-based strategy.

        Args:
            l_kde (KernelDensity): Kernel density estimator for the low-score distribution.
            g_kde (KernelDensity): Kernel density estimator for the high-score distribution.

        Returns:
            np.ndarray: Selected set of hyperparameters.
        """
        samples = l_kde.sample(self.n_samples)
        l_score = l_kde.score_samples(samples)
        g_score = g_kde.score_samples(samples)
        return samples[self.sample_select(l_score, g_score)]

    def search(self) -> tuple[np.ndarray, float]:
        """Executes the Parzen Tree Estimator algorithm to find the optimal solution.

        This method iteratively performs the Parzen Tree Estimator algorithm to search for the optimal solution.
        It updates the population of hyperparameters based on the scores and selects the best solution.

        Returns:
            Tuple[np.ndarray, float]: The best set of hyperparameters and its corresponding score.
        """
        self.initialize_population()
        for _ in range(self.max_iter):
            self.seed += 1
            l_kde, g_kde = self.segment_distributions()
            hps = self.choose_next_hps(l_kde, g_kde)
            score = self.func(hps)
            worst_index = np.argmax(self.scores)
            self.population[worst_index] = hps
            self.scores[worst_index] = score
        best_index = np.argmin(self.scores)
        return self.population[best_index], self.scores[best_index]


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(ParzenTreeEstimator)
