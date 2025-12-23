"""Imperialist Competitive Algorithm optimizer.

This module implements the Imperialist Competitive Algorithm (ICA) for solving
optimization problems. The ICA is a population-based algorithm that simulates the
competition between empires and colonies. It starts with a random population and
iteratively improves the solutions by assimilation, revolution, position exchange,
and imperialistic competition.

Example:
    To use this optimizer, create an instance of the `ImperialistCompetitiveAlgorithm` class and call
    the `search` method to run the optimization.

        from opt.imperialist_competitive_algorithm import ImperialistCompetitiveAlgorithm
        from opt.benchmark.functions import shifted_ackley

        # Define the objective function
        def objective_function(x):
            return shifted_ackley(x)

        # Create an instance of the optimizer
        optimizer = ImperialistCompetitiveAlgorithm(
            func=objective_function,
            dim=2,
            lower_bound=-32.768,
            upper_bound=32.768,
            num_empires=15,
            population_size=100,
            max_iter=1000,
        )

        # Run the optimization
        best_solution, best_fitness = optimizer.search()

        print(f"Best solution found: {best_solution}")
        print(f"Best fitness value: {best_fitness}")

Attributes:
    num_empires (int): The number of empires in the algorithm.
    revolution_rate (float): The rate of revolution, which determines the probability of a revolution occurring.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class ImperialistCompetitiveAlgorithm(AbstractOptimizer):
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

        >>> from opt.evolutionary.imperialist_competitive_algorithm import (
        ...     ImperialistCompetitiveAlgorithm,
        ... )
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = ImperialistCompetitiveAlgorithm(
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
        >>> optimizer = ImperialistCompetitiveAlgorithm(
        ...     func=sphere, lower_bound=-5, upper_bound=5, dim=10, max_iter=10000, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> len(solution) == 10
        True

    Args:
        FIXME: Document all parameters with BBOB guidance.
        Detected parameters from __init__ signature: func, dim, lower_bound, upper_bound, num_empires, population_size, max_iter, revolution_rate, seed

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
        num_empires: int = 15,
        population_size: int = 100,
        max_iter: int = 1000,
        revolution_rate: float = 0.3,
        seed: int | None = None,
    ) -> None:
        """Initialize the ImperialistCompetitiveAlgorithm class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )

        self.num_empires = num_empires
        self.revolution_rate = revolution_rate

    def search(self) -> tuple[np.ndarray, float]:
        """Run the Imperialist Competitive Algorithm optimization.

        Returns:
            tuple[np.ndarray, float]: The best solution found and its fitness value.

        """
        # Initialize population
        population = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.apply_along_axis(self.func, 1, population)

        # Create empires
        empires = []
        for i in range(self.num_empires):
            empire = {
                "imperialist": i,
                "colonies": list(range(self.num_empires, self.num_empires + i + 1)),
                "total_cost": fitness[i]
                + fitness[self.num_empires + i : self.num_empires + i + 1].sum(),
            }
            empires.append(empire)

        for _ in range(self.max_iter):
            self.seed += 1
            # Assimilation
            for empire in empires:
                self.seed += 1
                population[empire["colonies"]] += np.random.default_rng(
                    self.seed
                ).random((len(empire["colonies"]), self.dim)) * (
                    population[empire["imperialist"]] - population[empire["colonies"]]
                )

            # Revolution
            revolution_indices = (
                np.random.default_rng(self.seed).random(len(population))
                < self.revolution_rate
            )
            population[revolution_indices] = np.random.default_rng(self.seed).uniform(
                self.lower_bound, self.upper_bound, (revolution_indices.sum(), self.dim)
            )

            # Position exchange between a colony and Imperialist
            for empire in empires:
                self.seed += 1
                colonies_fitness = np.apply_along_axis(
                    self.func, 1, population[empire["colonies"]]
                )
                if colonies_fitness.min() < fitness[empire["imperialist"]]:
                    best_colony_index = colonies_fitness.argmin()
                    (
                        population[empire["imperialist"]],
                        population[empire["colonies"][best_colony_index]],
                    ) = (
                        population[empire["colonies"][best_colony_index]],
                        population[empire["imperialist"]],
                    )

            # Imperialistic competition
            total_power = sum([1 / empire["total_cost"] for empire in empires])
            for i in range(len(empires)):
                self.seed += 1
                for j in range(i + 1, len(empires)):
                    self.seed += 1
                    if (
                        np.random.default_rng(self.seed).random()
                        < (1 / empires[i]["total_cost"]) / total_power
                    ) and len(empires[j]["colonies"]) > 0:
                        lost_colony = empires[j]["colonies"].pop()
                        empires[i]["colonies"].append(lost_colony)

            # Eliminate the powerless empires
            empires = [empire for empire in empires if len(empire["colonies"]) > 0]

        best_solution = min(
            empires, key=lambda empire: self.func(population[empire["imperialist"]])
        )
        best_fitness = self.func(population[best_solution["imperialist"]])
        return population[best_solution["imperialist"]], best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(ImperialistCompetitiveAlgorithm)
