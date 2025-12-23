"""Bat Algorithm optimization algorithm.

This module implements the Bat Algorithm optimization algorithm. The Bat Algorithm is a
metaheuristic algorithm inspired by the echolocation behavior of bats. It is commonly
used for solving optimization problems.

The BatAlgorithm class provides an implementation of the Bat Algorithm optimization
algorithm. It takes an objective function, the dimensionality of the problem, the
search space bounds, the number of bats in the population, and other optional
parameters. The search method runs the Bat Algorithm optimization and returns the
best solution found.

Example:
    import numpy as np
    from opt.benchmark.functions import shifted_ackley
    from opt.bat_algorithm import BatAlgorithm

    # Define the objective function
    def objective_function(x):
        return np.sum(x ** 2)

    # Create an instance of the BatAlgorithm class
    optimizer = BatAlgorithm(
        func=objective_function,
        dim=2,
        lower_bound=-5.0,
        upper_bound=5.0,
        n_bats=10,
        max_iter=1000,
        loudness=0.5,
        pulse_rate=0.9,
        freq_min=0,
        freq_max=2
    )

    # Run the Bat Algorithm optimization
    best_solution, best_fitness = optimizer.search()

    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")

Attributes:
    freq_min (float): The minimum frequency of the bats.
    freq_max (float): The maximum frequency of the bats.
    positions (ndarray): The current positions of the bats.
    velocities (ndarray): The velocities of the bats.
    frequencies (ndarray): The frequencies of the bats.
    loudnesses (ndarray): The loudnesses of the bats.
    best_positions (ndarray): The best positions found by each bat.
    best_fitnesses (ndarray): The fitness values corresponding to the best positions found by each bat.
    alpha (float): The pulse rate of the bats.
    gamma (float): The loudness of the bats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class BatAlgorithm(AbstractOptimizer):
    r"""FIXME: [Algorithm Full Name] ([ACRONYM]) optimization algorithm.

    Algorithm Metadata:
        | Property          | Value                                    |
        |-------------------|------------------------------------------|
        | Algorithm Name    | FIXME: [Full algorithm name]             |
        | Acronym           | FIXME: [SHORT]                           |
        | Year Introduced   | FIXME: [YYYY]                            |
        | Authors           | FIXME: [Last, First; ...]                |
        | Algorithm Class   | Swarm Intelligence |
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

        >>> from opt.swarm_intelligence.bat_algorithm import BatAlgorithm
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = BatAlgorithm(
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
        >>> optimizer = BatAlgorithm(
        ...     func=sphere, lower_bound=-5, upper_bound=5, dim=10, max_iter=10000, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> len(solution) == 10
        True

    Args:
        FIXME: Document all parameters with BBOB guidance.
        Detected parameters from __init__ signature: func, dim, lower_bound, upper_bound, n_bats, max_iter, loudness, pulse_rate, freq_min, freq_max, seed

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
        n_bats: int,
        max_iter: int = 1000,
        loudness: float = 0.5,
        pulse_rate: float = 0.9,
        freq_min: float = 0,
        freq_max: float = 2,
        seed: int | None = None,
    ) -> None:
        """Initialize the BatAlgorithm class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=n_bats,
        )

        self.freq_min = freq_min
        self.freq_max = freq_max
        self.positions = np.random.default_rng(self.seed).uniform(
            lower_bound, upper_bound, (self.population_size, dim)
        )
        self.velocities = np.zeros((self.population_size, dim))
        self.frequencies = np.random.default_rng(self.seed).uniform(
            freq_min, freq_max, self.population_size
        )
        self.loudnesses = np.full(self.population_size, loudness)
        self.best_positions = self.positions.copy()
        self.best_fitnesses = np.full(self.population_size, np.inf)
        self.alpha = pulse_rate
        self.gamma = loudness

    def search(self) -> tuple[np.ndarray, float]:
        """Run the Bat Algorithm optimization.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the best solution found (position) and its fitness value.

        """
        best_solution_idx = None
        for _ in range(self.max_iter):
            self.seed += 1
            for i in range(self.population_size):
                self.seed += 1
                self.positions[i] += self.velocities[i]
                fitness = self.func(self.positions[i])
                if fitness < self.best_fitnesses[i]:
                    self.best_positions[i] = self.positions[i].copy()
                    self.best_fitnesses[i] = fitness
                if (
                    best_solution_idx is None
                    or fitness < self.best_fitnesses[best_solution_idx]
                ):
                    best_solution_idx = i
                self.velocities[i] += (
                    self.best_positions[best_solution_idx] - self.positions[i]
                ) * self.loudnesses[i]
                self.frequencies[i] = (
                    self.freq_min
                    + (self.freq_max - self.freq_min)
                    * np.random.default_rng(self.seed).random()
                )
                if np.random.default_rng(self.seed + 1).random() > self.loudnesses[i]:
                    self.seed += 1
                    self.positions[i] = self.best_positions[
                        best_solution_idx
                    ] + self.alpha * np.random.default_rng(self.seed).normal(
                        0, 1, self.dim
                    )
            self.loudnesses *= self.gamma
        return self.best_positions[best_solution_idx], self.best_fitnesses[
            best_solution_idx
        ]


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(BatAlgorithm)
