"""Stability testing framework for optimization algorithms.

This module provides tools for running optimization algorithms multiple times
with different random seeds to assess their stability and performance consistency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from matplotlib.figure import Figure

    from opt.abstract_optimizer import AbstractOptimizer


class StabilityResults:
    """Results from stability testing of an optimization algorithm.

    This class stores and analyzes results from multiple runs of an optimizer
    with different random seeds.

    Args:
        optimizer_name (str): Name of the optimizer class.
        function_name (str): Name of the objective function.
        solutions (list[np.ndarray]): List of best solutions from each run.
        fitness_values (list[float]): List of best fitness values from each run.
        seeds (list[int]): List of random seeds used for each run.

    Attributes:
        optimizer_name (str): Name of the optimizer class.
        function_name (str): Name of the objective function.
        solutions (list[np.ndarray]): List of best solutions from each run.
        fitness_values (np.ndarray): Array of best fitness values from each run.
        seeds (list[int]): List of random seeds used for each run.
    """

    def __init__(
        self,
        optimizer_name: str,
        function_name: str,
        solutions: list[np.ndarray],
        fitness_values: list[float],
        seeds: list[int],
    ) -> None:
        """Initialize StabilityResults."""
        self.optimizer_name = optimizer_name
        self.function_name = function_name
        self.solutions = solutions
        self.fitness_values = np.array(fitness_values)
        self.seeds = seeds

    def summary(self) -> dict[str, float]:
        """Generate statistical summary of the results.

        Returns:
            dict[str, float]: Dictionary containing mean, std, min, max, and median fitness values.
            {'mean': 0.123, 'std': 0.045, 'min': 0.001, 'max': 0.234, 'median': 0.112}
        """
        return {
            "mean": float(np.mean(self.fitness_values)),
            "std": float(np.std(self.fitness_values)),
            "min": float(np.min(self.fitness_values)),
            "max": float(np.max(self.fitness_values)),
            "median": float(np.median(self.fitness_values)),
            "q25": float(np.percentile(self.fitness_values, 25)),
            "q75": float(np.percentile(self.fitness_values, 75)),
        }

    def print_summary(self) -> None:
        """Print a formatted summary of the results.
        Stability Test Results for ParticleSwarm on shifted_ackley
        ============================================================
        Number of runs: 10
        ...
        """
        stats = self.summary()
        print(
            f"\nStability Test Results for {self.optimizer_name} on {self.function_name}"
        )
        print("=" * 60)
        print(f"Number of runs: {len(self.fitness_values)}")
        print(f"Mean fitness:   {stats['mean']:.6f}")
        print(f"Std deviation:  {stats['std']:.6f}")
        print(f"Min fitness:    {stats['min']:.6f}")
        print(f"Max fitness:    {stats['max']:.6f}")
        print(f"Median fitness: {stats['median']:.6f}")
        print(f"Q25 fitness:    {stats['q25']:.6f}")
        print(f"Q75 fitness:    {stats['q75']:.6f}")
        if stats["mean"] != 0:
            cv = stats["std"] / stats["mean"]
            print(f"CV (std/mean):  {cv:.4f}")
        else:
            print("CV: N/A")
        print("=" * 60)

    def plot_boxplot(self, show: bool = True, save_path: str | None = None) -> Figure:
        """Generate box plot of fitness values across runs.

        Args:
            show (bool, optional): Whether to display the plot. Defaults to True.
            save_path (str | None, optional): Path to save the figure. If None, doesn't save.

        Returns:
            Figure: The matplotlib figure object.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))

        # Create box plot
        bp = ax.boxplot(
            [self.fitness_values],
            tick_labels=[self.optimizer_name],
            patch_artist=True,
            widths=0.6,
        )

        # Customize box plot colors
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)

        for whisker in bp["whiskers"]:
            whisker.set(linewidth=1.5)

        for median in bp["medians"]:
            median.set(color="red", linewidth=2)

        # Add individual points
        y_points = self.fitness_values
        rng = np.random.default_rng(42)
        x_points = rng.normal(1, 0.04, size=len(y_points))
        ax.scatter(x_points, y_points, alpha=0.5, color="darkblue", s=30, zorder=3)

        ax.set_ylabel("Best Fitness Value", fontsize=12)
        ax.set_title(
            f"Stability Analysis: {self.optimizer_name} on {self.function_name}\n"
            f"(n={len(self.fitness_values)} runs)",
            fontsize=13,
        )
        ax.grid(True, alpha=0.3, axis="y")

        # Add statistics text
        stats = self.summary()
        stats_text = (
            f"Mean: {stats['mean']:.4f}\n"
            f"Std: {stats['std']:.4f}\n"
            f"Min: {stats['min']:.4f}\n"
            f"Max: {stats['max']:.4f}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Box plot saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_histogram(
        self, bins: int = 20, show: bool = True, save_path: str | None = None
    ) -> Figure:
        """Generate histogram of fitness values across runs.

        Args:
            bins (int, optional): Number of bins for histogram. Defaults to 20.
            show (bool, optional): Whether to display the plot. Defaults to True.
            save_path (str | None, optional): Path to save the figure. If None, doesn't save.

        Returns:
            Figure: The matplotlib figure object.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create histogram
        ax.hist(
            self.fitness_values,
            bins=bins,
            color="skyblue",
            edgecolor="black",
            alpha=0.7,
        )

        # Add vertical lines for statistics
        stats = self.summary()
        ax.axvline(
            stats["mean"], color="red", linestyle="--", linewidth=2, label="Mean"
        )
        ax.axvline(
            stats["median"], color="green", linestyle="--", linewidth=2, label="Median"
        )

        ax.set_xlabel("Best Fitness Value", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            f"Fitness Distribution: {self.optimizer_name} on {self.function_name}\n"
            f"(n={len(self.fitness_values)} runs)",
            fontsize=13,
        )
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Histogram saved to {save_path}")

        if show:
            plt.show()

        return fig


def run_stability_test(
    optimizer_class: type[AbstractOptimizer],
    func: Callable[[np.ndarray], float],
    lower_bound: float,
    upper_bound: float,
    dim: int,
    max_iter: int = 100,
    seeds: Sequence[int] | None = None,
    n_runs: int = 10,
    verbose: bool = True,
    **optimizer_kwargs,
) -> StabilityResults:
    """Run stability test for an optimization algorithm.

    Runs the optimizer multiple times with different random seeds to assess
    performance stability and consistency.

    Args:
        optimizer_class (type[AbstractOptimizer]): The optimizer class to test.
        func (Callable): The objective function to optimize.
        lower_bound (float): Lower bound of the search space.
        upper_bound (float): Upper bound of the search space.
        dim (int): Dimensionality of the search space.
        max_iter (int, optional): Maximum iterations per run. Defaults to 100.
        seeds (Sequence[int] | None, optional): Specific seeds to use. If None, generates random seeds.
        n_runs (int, optional): Number of runs if seeds not specified. Defaults to 10.
        verbose (bool, optional): Whether to print progress. Defaults to True.
        **optimizer_kwargs: Additional keyword arguments for the optimizer.

    Returns:
        StabilityResults: Object containing results from all runs.

    Example:
        >>> from opt.swarm_intelligence.particle_swarm import ParticleSwarm
        >>> from opt.benchmark.functions import sphere
        >>> from opt.visualization import run_stability_test
        >>> results = run_stability_test(
        ...     optimizer_class=ParticleSwarm,
        ...     func=sphere,
        ...     lower_bound=-5,
        ...     upper_bound=5,
        ...     dim=2,
        ...     max_iter=10,
        ...     seeds=[42, 123],
        ...     verbose=False,
        ... )
        >>> len(results.fitness_values) == 2
        True
    """
    # Determine seeds to use
    if seeds is None:
        rng = np.random.default_rng(42)
        test_seeds = rng.integers(0, 2**31, size=n_runs).tolist()
    else:
        test_seeds = list(seeds)

    if verbose:
        print(f"\nRunning stability test for {optimizer_class.__name__}")
        print(f"Function: {func.__name__}")
        print(f"Number of runs: {len(test_seeds)}")
        print(f"Max iterations per run: {max_iter}")
        print("-" * 60)

    solutions = []
    fitness_values = []

    for i, seed in enumerate(test_seeds):
        if verbose:
            print(f"Run {i + 1}/{len(test_seeds)} (seed={seed})...", end=" ")

        optimizer = optimizer_class(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            **optimizer_kwargs,
        )

        solution, fitness = optimizer.search()
        solutions.append(solution)
        fitness_values.append(fitness)

        if verbose:
            print(f"Fitness: {fitness:.6f}")

    results = StabilityResults(
        optimizer_name=optimizer_class.__name__,
        function_name=func.__name__,
        solutions=solutions,
        fitness_values=fitness_values,
        seeds=test_seeds,
    )

    if verbose:
        results.print_summary()

    return results


def compare_optimizers_stability(
    optimizer_classes: list[type[AbstractOptimizer]],
    func: Callable[[np.ndarray], float],
    lower_bound: float,
    upper_bound: float,
    dim: int,
    max_iter: int = 100,
    n_runs: int = 10,
    show: bool = True,
    save_path: str | None = None,
) -> tuple[dict[str, StabilityResults], Figure]:
    """Compare stability of multiple optimizers.

    Runs multiple optimizers on the same function and compares their stability
    using box plots.

    Args:
        optimizer_classes (list[type[AbstractOptimizer]]): List of optimizer classes to compare.
        func (Callable): The objective function to optimize.
        lower_bound (float): Lower bound of the search space.
        upper_bound (float): Upper bound of the search space.
        dim (int): Dimensionality of the search space.
        max_iter (int, optional): Maximum iterations per run. Defaults to 100.
        n_runs (int, optional): Number of runs per optimizer. Defaults to 10.
        show (bool, optional): Whether to display the plot. Defaults to True.
        save_path (str | None, optional): Path to save the figure. If None, doesn't save.

    Returns:
        tuple[dict[str, StabilityResults], Figure]: Dictionary of results and comparison figure.

    Example:
        >>> from opt.swarm_intelligence.particle_swarm import ParticleSwarm
        >>> from opt.evolutionary.genetic_algorithm import GeneticAlgorithm
        >>> from opt.benchmark.functions import sphere
        >>> results, fig = compare_optimizers_stability(
        ...     optimizer_classes=[ParticleSwarm, GeneticAlgorithm],
        ...     func=sphere,
        ...     lower_bound=-5,
        ...     upper_bound=5,
        ...     dim=2,
        ...     max_iter=10,
        ...     n_runs=2,
        ...     show=False,
        ... )  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    all_results = {}

    # Run stability tests for each optimizer
    for optimizer_class in optimizer_classes:
        results = run_stability_test(
            optimizer_class=optimizer_class,
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            n_runs=n_runs,
            verbose=False,
        )
        all_results[optimizer_class.__name__] = results

    # Create comparison box plot
    fig, ax = plt.subplots(figsize=(max(10, len(optimizer_classes) * 2), 6))

    data = [results.fitness_values for results in all_results.values()]
    labels = list(all_results.keys())

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)

    # Customize colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(optimizer_classes)))
    # Python 3.10+ supports strict parameter, but we ensure equal lengths
    assert len(bp["boxes"]) == len(colors)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for median in bp["medians"]:
        median.set(color="red", linewidth=2)

    # Add individual points
    for i, (_name, results) in enumerate(all_results.items(), 1):
        y_points = results.fitness_values
        rng = np.random.default_rng(42)
        x_points = rng.normal(i, 0.04, size=len(y_points))
        ax.scatter(x_points, y_points, alpha=0.5, s=30, zorder=3, color="darkblue")

    ax.set_ylabel("Best Fitness Value", fontsize=12)
    ax.set_xlabel("Optimizer", fontsize=12)
    ax.set_title(
        f"Optimizer Stability Comparison on {func.__name__}\n(n={n_runs} runs each)",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to {save_path}")

    if show:
        plt.show()

    return all_results, fig
