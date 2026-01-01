"""Known optimal values for benchmark functions.

This module provides a mapping of benchmark function names to their
known optimal values (global minima). These values are used for:
- Early stopping (target precision)
- Expected Running Time (ERT) calculation
- Success rate computation

All values are for minimization problems.
"""

from __future__ import annotations

import numpy as np

# Global optima for benchmark functions
# Format: "function_name": optimal_value
FUNCTION_OPTIMA: dict[str, float] = {
    # Simple unimodal functions
    "sphere": 0.0,  # Global minimum at x = [0, 0, ...]
    "rosenbrock": 0.0,  # Global minimum at x = [1, 1, ...]
    # Multimodal functions
    "ackley": 0.0,  # Global minimum at x = [0, 0, ...]
    "shifted_ackley": 0.0,  # Global minimum at shifted position
    "rastrigin": 0.0,  # Global minimum at x = [0, 0, ...]
    "griewank": 0.0,  # Global minimum at x = [0, 0, ...]
    "schwefel": 0.0,  # Global minimum at x = [420.9687, 420.9687, ...]
    # Other classic functions
    "levi": 0.0,  # Global minimum at x = [1, 1]
    "himmelblau": 0.0,  # Global minimum at 4 locations
    "eggholder": -959.6407,  # Global minimum at x ≈ [512, 404.2319]
    "beale": 0.0,  # Global minimum at x = [3, 0.5]
    "goldstein_price": 3.0,  # Global minimum at x = [0, -1]
    "booth": 0.0,  # Global minimum at x = [1, 3]
    "bukin": 0.0,  # Global minimum at x = [-10, 1]
    "matyas": 0.0,  # Global minimum at x = [0, 0]
    "levi_n13": 0.0,  # Global minimum at x = [1, 1]
    "three_hump_camel": 0.0,  # Global minimum at x = [0, 0]
    "easom": -1.0,  # Global minimum at x = [π, π]
    "cross_in_tray": -2.06261,  # Global minimum at 4 locations
    "hold_table": -19.2085,  # Global minimum at 4 locations
    "mccormick": -1.9133,  # Global minimum at x ≈ [-0.54719, -1.54719]
}


def get_optimum(func_name: str) -> float:
    """Get the optimal value for a benchmark function.

    Args:
        func_name: Name of the benchmark function (e.g., 'sphere', 'rosenbrock').

    Returns:
        The known optimal value for the function.

    Raises:
        KeyError: If the function name is not in the known optima mapping.

    Example:
        >>> from opt.benchmark.optima import get_optimum
        >>> get_optimum('sphere')
        0.0
        >>> get_optimum('eggholder')
        -959.6407
    """
    if func_name not in FUNCTION_OPTIMA:
        msg = (
            f"Unknown function '{func_name}'. "
            f"Available functions: {sorted(FUNCTION_OPTIMA.keys())}"
        )
        raise KeyError(msg)
    return FUNCTION_OPTIMA[func_name]


def get_optimum_safe(func_name: str, default: float | None = None) -> float | None:
    """Get the optimal value for a benchmark function with a fallback.

    Args:
        func_name: Name of the benchmark function.
        default: Default value to return if function is unknown. Defaults to None.

    Returns:
        The known optimal value for the function, or the default if unknown.

    Example:
        >>> from opt.benchmark.optima import get_optimum_safe
        >>> get_optimum_safe('sphere')
        0.0
        >>> get_optimum_safe('unknown_function', default=0.0)
        0.0
        >>> get_optimum_safe('unknown_function') is None
        True
    """
    return FUNCTION_OPTIMA.get(func_name, default)


def is_converged(
    current_fitness: float,
    func_name: str,
    target_precision: float = 1e-8,
) -> bool:
    """Check if optimization has converged to target precision.

    Args:
        current_fitness: Current best fitness value.
        func_name: Name of the benchmark function.
        target_precision: Target precision threshold. Defaults to 1e-8.

    Returns:
        True if |f(x) - f_opt| < target_precision, False otherwise.

    Example:
        >>> from opt.benchmark.optima import is_converged
        >>> is_converged(0.0, 'sphere', target_precision=1e-8)
        True
        >>> is_converged(1e-6, 'sphere', target_precision=1e-8)
        False
        >>> is_converged(1e-10, 'rosenbrock', target_precision=1e-8)
        True
    """
    f_opt = get_optimum_safe(func_name)
    if f_opt is None:
        return False
    return abs(current_fitness - f_opt) < target_precision
