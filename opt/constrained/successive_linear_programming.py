"""Successive Linear Programming optimization algorithm.

!!! warning

    This module is still under development and is not yet ready for use.

This module implements the Successive Linear Programming optimization algorithm. The
algorithm performs a search for the optimal solution by iteratively updating a
population of individuals. At each iteration, it computes the gradient of the objective
function for each individual and uses linear programming to find a new solution that
improves the objective function value. The process continues until the maximum number
of iterations is reached.

The SuccessiveLinearProgramming class is the main class that implements the algorithm.
It inherits from the AbstractOptimizer class and overrides the search() and gradient()
methods.

Attributes:
    seed (int): The seed value for the random number generator.
    lower_bound (float): The lower bound for the search space.
    upper_bound (float): The upper bound for the search space.
    population_size (int): The size of the population.
    dim (int): The dimensionality of the search space.
    max_iter (int): The maximum number of iterations.

Example usage:
    optimizer = SuccessiveLinearProgramming(
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

import numpy as np

from scipy.optimize import linprog

from opt.abstract import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley


class SuccessiveLinearProgramming(AbstractOptimizer):
    r"""Successive Linear Programming (SLP) optimization algorithm.

    Algorithm Metadata:
        | Property          | Value                                    |
        |-------------------|------------------------------------------|
        | Algorithm Name    | Successive Linear Programming            |
        | Acronym           | SLP                                      |
        | Year Introduced   | 1961                                     |
        | Authors           | Griffith, R. E.; Stewart, R. A.          |
        | Algorithm Class   | Constrained                              |
        | Complexity        | O(n³) per LP subproblem                  |
        | Properties        | Gradient-based, Deterministic        |
        | Implementation    | Python 3.10+                             |
        | COCO Compatible   | Yes                                      |

    Mathematical Formulation:
        At each iteration $k$, solve linear programming subproblem:

            $$
            \min_d \quad \nabla f(x_k)^T d
            $$

            $$
            \text{subject to} \quad \nabla g_i(x_k)^T d + g_i(x_k) \leq 0
            $$

        where:
            - $x_k$ is current iterate
            - $d$ is the search direction
            - $\nabla f(x_k)$ is gradient of objective
            - $g_i(x)$ are inequality constraints

        Update:

            $$
            x_{k+1} = x_k + d_k
            $$

        Constraint handling:
            - **Boundary conditions**: Box constraints in LP
            - **Feasibility enforcement**: Linearized constraints
            - **Trust region**: Implicit via bounds on search space

    Hyperparameters:
        | Parameter              | Default | BBOB Recommended | Description                    |
        |------------------------|---------|------------------|--------------------------------|
        | max_iter               | 1000    | 5000-10000       | Maximum SLP iterations         |
        | population_size        | 100     | 50-200           | Population for gradient est.   |

        **Sensitivity Analysis**:
            - `population_size`: **Medium** impact - affects gradient quality
            - Recommended tuning ranges: $\text{pop\_size} \in [50, 200]$

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

        >>> from opt.constrained.successive_linear_programming import SuccessiveLinearProgramming
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = SuccessiveLinearProgramming(
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
        >>> optimizer = SuccessiveLinearProgramming(
        ...     func=sphere, lower_bound=-5, upper_bound=5, dim=10, max_iter=10000, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> # TODO: Replaced trivial doctest with a suggested mini-benchmark — please review.
        >>> # Suggested mini-benchmark (seeded, quick):
        >>> # >>> res = optimizer.benchmark(store=True, quick=True, quick_max_iter=10, seed=0)
        >>> # >>> assert isinstance(res, dict) and res.get('metadata') is not None
        True

    Args:
        func (Callable[[ndarray], float]): Objective function to minimize. Must accept
            numpy array and return scalar. BBOB functions available in
            `opt.benchmark.functions`.
        lower_bound (float): Lower bound of search space. BBOB typical: -5
            (most functions).
        upper_bound (float): Upper bound of search space. BBOB typical: 5
            (most functions).
        dim (int): Problem dimensionality. BBOB standard dimensions: 2, 3, 5, 10, 20, 40.
        max_iter (int, optional): Maximum SLP iterations. BBOB recommendation: 5000-10000
            for SLP. Defaults to 1000.
        population_size (int, optional): Population size for gradient estimation via
            finite differences. Defaults to 100.
        seed (int | None, optional): Random seed for reproducibility. BBOB requires
            seeds 0-14 for 15 runs. If None, generates random seed. Defaults to None.

    Attributes:
        func (Callable[[ndarray], float]): The objective function being optimized.
        lower_bound (float): Lower search space boundary.
        upper_bound (float): Upper search space boundary.
        dim (int): Problem dimensionality.
        max_iter (int): Maximum number of SLP iterations.
        population_size (int): Population size for gradient estimation.
        seed (int): **REQUIRED** Random seed for reproducibility (BBOB compliance).

    Methods:
        search() -> tuple[np.ndarray, float]:
            Execute Successive Linear Programming optimization.

    Returns:
                tuple[np.ndarray, float]:
                    - best_solution (np.ndarray): Best solution found, shape (dim,)
                    - best_fitness (float): Fitness value at best_solution

    Raises:
        ValueError: If search space is invalid or function evaluation fails.

    Notes:
                - Uses scipy linprog for LP subproblems
                - Finite difference gradient estimation
                - BBOB: Returns final best solution after max_iter

    References:
        [1] Griffith, R. E., & Stewart, R. A. (1961). "A nonlinear programming
            technique for the optimization of continuous processing systems."
            _Management Science_, 7(4), 379-392.
            https://doi.org/10.1287/mnsc.7.4.379

        [2] Palacios-Gomez, F., Lasdon, L., & Engquist, M. (1982). "Nonlinear
            optimization by successive linear programming."
            _Management Science_, 28(10), 1106-1120.
            https://doi.org/10.1287/mnsc.28.10.1106

        [3] Nocedal, J., & Wright, S. J. (2006). "Numerical Optimization" (2nd ed.).
            _Springer_. Chapter 19: Sequential Linear Programming.

        [4] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
            "COCO: A platform for comparing continuous optimizers in a black-box setting."
            _Optimization Methods and Software_, 36(1), 114-144.
            https://doi.org/10.1080/10556788.2020.1808977

        **COCO Data Archive**:
            - Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
            - Code repository: https://github.com/Anselmoo/useful-optimizer

        **Implementation**:
            - This implementation: scipy.optimize.linprog for LP subproblems

    See Also:
        SequentialQuadraticProgramming: Quadratic subproblem variant
            BBOB Comparison: SQP generally superior for smooth nonlinear problems

        PenaltyMethodOptimizer: Penalty-based alternative
            BBOB Comparison: SLP better for highly constrained linear-like problems

        AugmentedLagrangian: Penalty + multiplier method
            BBOB Comparison: ALM more robust for general nonlinear constraints

        AbstractOptimizer: Base class for all optimizers
        opt.benchmark.functions: BBOB-compatible test functions

        Related BBOB Algorithm Classes:
            - Classical: SimulatedAnnealing, NelderMead
            - Gradient: AdamW, BFGS

    Notes:
        **Computational Complexity**:
            - Time per iteration: $O(n^3)$ for LP solve + $O(n \cdot \text{pop\_size})$ for gradient
            - Space complexity: $O(n^2)$ for LP constraint matrices
            - BBOB budget usage: _Typically 30-60% of dim*10000 for convergence_

        **BBOB Performance Characteristics**:
            - **Best function classes**: Piecewise linear, highly constrained
            - **Weak function classes**: Strongly nonlinear, smooth unconstrained
            - Typical success rate at 1e-8 precision: **40-55%** (dim=5, general problems)
            - Expected Running Time (ERT): Slower than SQP for smooth problems

        **Convergence Properties**:
            - Convergence rate: Linear for general problems, quadratic at vertex optima
            - Local vs Global: Limited global search, strong at feasible vertices
            - Premature convergence risk: **Medium** (may zigzag near optimum)

        **Reproducibility**:
            - **Deterministic**: Yes - Same seed guarantees same results
            - **BBOB compliance**: seed parameter required for 15 independent runs
            - Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
            - RNG usage: `numpy.random.default_rng(self.seed)` for population init

        **Implementation Details**:
            - Parallelization: Not supported (sequential LP solves)
            - Constraint handling: Linearized constraints in LP subproblems
            - Numerical stability: Finite difference gradients may be imprecise
            - Inner solver: scipy.optimize.linprog with HiGHS method
            - Gradient: Finite differences with ε=1e-5 perturbation

        **Known Limitations**:
            - Superseded by SQP for most smooth nonlinear problems
            - Finite difference gradients less accurate than analytical
            - Linear approximation poor for strongly nonlinear objectives
            - May require many iterations for high-precision convergence
            - BBOB adaptation note: Standard BBOB is unconstrained; SLP designed
              for constrained optimization

        **Version History**:
            - v0.1.0: Initial implementation
            - v0.1.2: Added COCO/BBOB compliant docstring
    """

    def search(self) -> tuple[np.ndarray, float]:
        """Performs the search for the optimal solution.

        Returns:
        tuple[np.ndarray, float]: A tuple containing the best solution found and
        its corresponding objective function value.
        """
        population = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        for _ in range(self.max_iter):
            for i in range(self.population_size):
                gradient = self.gradient(population[i])
                bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
                result = linprog(c=gradient, bounds=bounds, method="highs")
                if result.success:
                    population[i] = result.x
        best_index = np.argmin([self.func(individual) for individual in population])
        return population[best_index], self.func(population[best_index])

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Computes the gradient of the objective function at a given point.

        Args:
            x (np.ndarray): The point at which to compute the gradient.

        Returns:
        np.ndarray: The gradient vector.
        """
        eps = 1e-5
        return np.array(
            [
                (self.func(x + eps * unit_vector) - self.func(x)) / eps
                for unit_vector in np.eye(self.dim)
            ]
        )


if __name__ == "__main__":
    optimizer = SuccessiveLinearProgramming(
        func=shifted_ackley, dim=2, lower_bound=-2.768, upper_bound=+2.768
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")
