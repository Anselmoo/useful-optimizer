"""Bayesian Optimization.

This module implements Bayesian Optimization, a probabilistic optimization
technique using Gaussian Process surrogate models.

The algorithm builds a probabilistic model of the objective function and
uses it to select promising points to evaluate.

Reference:
    Snoek, J., Larochelle, H., & Adams, R. P. (2012).
    Practical Bayesian Optimization of Machine Learning Algorithms.
    Advances in Neural Information Processing Systems 25 (NIPS 2012).

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = BayesianOptimizer(
    ...     func=shifted_ackley,
    ...     lower_bound=-2.768,
    ...     upper_bound=2.768,
    ...     dim=2,
    ...     n_initial=10,
    ...     max_iter=50,
    ... )
    >>> best_solution, best_fitness = optimizer.search()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.optimize import minimize
from scipy.stats import norm

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable


class BayesianOptimizer(AbstractOptimizer):
    """Bayesian Optimization using Gaussian Process surrogate.

    This algorithm uses:
    1. Gaussian Process regression to model the objective
    2. Expected Improvement acquisition function
    3. Sequential design to select evaluation points

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        n_initial: Number of initial random samples.
        max_iter: Maximum number of iterations.
        xi: Exploration-exploitation trade-off parameter.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        n_initial: int = 10,
        max_iter: int = 50,
        xi: float = 0.01,
    ) -> None:
        """Initialize Bayesian Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            n_initial: Number of initial random samples. Defaults to 10.
            max_iter: Maximum iterations. Defaults to 50.
            xi: Exploration parameter. Defaults to 0.01.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.n_initial = n_initial
        self.xi = xi

    def _kernel(
        self, X1: np.ndarray, X2: np.ndarray, length_scale: float = 1.0
    ) -> np.ndarray:
        """Compute RBF (squared exponential) kernel.

        Args:
            X1: First set of points.
            X2: Second set of points.
            length_scale: Kernel length scale.

        Returns:
            Kernel matrix.
        """
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        dists = (
            np.sum(X1**2, axis=1).reshape(-1, 1)
            + np.sum(X2**2, axis=1)
            - 2 * np.dot(X1, X2.T)
        )
        return np.exp(-0.5 * dists / length_scale**2)

    def _gp_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        noise: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gaussian Process prediction.

        Args:
            X_train: Training points.
            y_train: Training values.
            X_test: Test points.
            noise: Observation noise variance.

        Returns:
            Tuple of (mean predictions, standard deviations).
        """
        K = self._kernel(X_train, X_train) + noise * np.eye(len(X_train))
        K_s = self._kernel(X_train, X_test)
        K_ss = self._kernel(X_test, X_test)

        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            mu = K_s.T.dot(alpha)
            v = np.linalg.solve(L, K_s)
            var = np.diag(K_ss) - np.sum(v**2, axis=0)
            std = np.sqrt(np.maximum(var, 1e-10))
        except np.linalg.LinAlgError:
            mu = np.full(len(X_test), np.mean(y_train))
            std = np.full(len(X_test), np.std(y_train))

        return mu, std

    def _expected_improvement(
        self, X: np.ndarray, X_train: np.ndarray, y_train: np.ndarray
    ) -> float:
        """Compute Expected Improvement acquisition function.

        Args:
            X: Point to evaluate.
            X_train: Training points.
            y_train: Training values.

        Returns:
            Expected improvement value (negated for minimization).
        """
        X = np.atleast_2d(X)
        mu, std = self._gp_predict(X_train, y_train, X)

        f_best = np.min(y_train)
        z = (f_best - mu - self.xi) / (std + 1e-10)
        ei = (f_best - mu - self.xi) * norm.cdf(z) + std * norm.pdf(z)

        return -ei[0]  # Negative for minimization

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Bayesian Optimization algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initial random samples
        X_samples = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.n_initial, self.dim)
        )
        y_samples = np.array([self.func(x) for x in X_samples])

        best_idx = np.argmin(y_samples)
        best_solution = X_samples[best_idx].copy()
        best_fitness = y_samples[best_idx]

        bounds = [(self.lower_bound, self.upper_bound)] * self.dim

        for _ in range(self.max_iter):
            # Find next point by maximizing expected improvement
            best_ei = np.inf
            best_x = None

            # Multi-start optimization of acquisition function
            for _ in range(10):
                x0 = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                result = minimize(
                    lambda x: self._expected_improvement(x, X_samples, y_samples),
                    x0,
                    bounds=bounds,
                    method="L-BFGS-B",
                )
                if result.fun < best_ei:
                    best_ei = result.fun
                    best_x = result.x

            if best_x is None:
                best_x = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

            # Evaluate new point
            new_y = self.func(best_x)

            # Update samples
            X_samples = np.vstack([X_samples, best_x])
            y_samples = np.append(y_samples, new_y)

            if new_y < best_fitness:
                best_solution = best_x.copy()
                best_fitness = new_y

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(BayesianOptimizer)
