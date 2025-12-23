"""Adaptive Moment Estimation (Adam) Optimizer.

This module implements the Adam optimization algorithm. Adam is a gradient-based
optimization algorithm that computes adaptive learning rates for each parameter. It
combines the advantages of two other extensions of stochastic gradient descent:

    - AdaGrad
    - RMSProp

Adam works well in practice and compares favorably to other adaptive learning-method
algorithms as it converges fast and the learning speed of the Model is quite fast and
efficient. It is straightforward to implement, is computationally efficient, has little
memory requirements, is invariant to diagonal rescaling of the gradients, and is well
suited for problems that are large in terms of data and/or parameters.

Example:
    optimizer = Adam(func=objective_function, learning_rate=0.01, initial_guess=[0, 0])
    best_solution, best_fitness = optimizer.optimize()

Attributes:
    func (Callable): The objective function to optimize.
    learning_rate (float): The learning rate for the optimization.
    initial_guess (List[float]): The starting point for the optimization.

Methods:
    optimize(): Perform the Adam optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.optimize import approx_fprime

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class ADAMOptimization(AbstractOptimizer):
    r"""FIXME: [Algorithm Full Name] ([ACRONYM]) optimization algorithm.

    Algorithm Metadata:
        | Property          | Value                                    |
        |-------------------|------------------------------------------|
        | Algorithm Name    | FIXME: [Full algorithm name]             |
        | Acronym           | FIXME: [SHORT]                           |
        | Year Introduced   | FIXME: [YYYY]                            |
        | Authors           | FIXME: [Last, First; ...]                |
        | Algorithm Class   | Gradient Based |
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

        >>> from opt.gradient_based.adaptive_moment_estimation import ADAMOptimization
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = ADAMOptimization(
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
        >>> optimizer = ADAMOptimization(
        ...     func=sphere, lower_bound=-5, upper_bound=5, dim=10, max_iter=10000, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> len(solution) == 10
        True

    Args:
        FIXME: Document all parameters with BBOB guidance.
        Detected parameters from __init__ signature: func, lower_bound, upper_bound, dim, max_iter, alpha, beta1, beta2, epsilon, seed

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
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = 1000,
        alpha: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-13,
        seed: int | None = None,
    ) -> None:
        """Initialize the ADAM optimization algorithm."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
        )
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the ADAM optimization search.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the best solution found and its fitness value.
        """
        m = np.zeros(self.dim)
        v = np.zeros(self.dim)
        best_solution = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, self.dim
        )
        best_fitness = self.func(best_solution)

        for t in range(1, self.max_iter + 1):
            grad = self._compute_gradient(best_solution)
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * np.square(grad)
            m_hat = m / (1 - np.power(self.beta1, t))
            v_hat = v / (1 - np.power(self.beta2, t))

            best_solution = best_solution - self.alpha * m_hat / (
                np.sqrt(v_hat) + self.epsilon
            )
            best_solution = np.clip(best_solution, self.lower_bound, self.upper_bound)

            fitness = self.func(best_solution)
            best_fitness = min(best_fitness, fitness)

        return best_solution, best_fitness

    def _compute_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the objective function at a given point.

        Args:
            x (np.ndarray): The point at which to compute the gradient.

        Returns:
            np.ndarray: The gradient vector.
        """
        epsilon = np.sqrt(np.finfo(float).eps)
        return approx_fprime(x, self.func, epsilon)


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(ADAMOptimization)
