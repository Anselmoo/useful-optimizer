"""Visualization module for optimization algorithms.

This module provides visualization capabilities for optimization algorithms,
including convergence curves, trajectory plots, and average fitness tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from opt.abstract_optimizer import AbstractOptimizer


class Visualizer:
    """Visualizer for optimization algorithms.

    This class provides various visualization methods for optimization algorithms,
    including convergence curves, trajectory plots, and population fitness tracking.

    Args:
        optimizer (AbstractOptimizer): The optimizer instance to visualize.
            Must have been run with track_history=True.

    Raises:
        ValueError: If optimizer doesn't have history tracked.
    """

    def __init__(self, optimizer: AbstractOptimizer) -> None:
        """Initialize the Visualizer.

        Args:
            optimizer (AbstractOptimizer): The optimizer instance to visualize.

        Raises:
        ValueError: If optimizer doesn't have history tracked.
        """
        if not optimizer.track_history or not optimizer.history:
            msg = (
                "Optimizer must be run with track_history=True to use visualization. "
                "Re-run the optimizer with track_history=True."
            )
            raise ValueError(msg)

        self.optimizer = optimizer
        self.history = optimizer.history

    def plot_convergence(
        self, log_scale: bool = False, show: bool = True, ax: Axes | None = None
    ) -> Figure:
        """Plot convergence curve showing best fitness over iterations.

        Args:
            log_scale (bool, optional): Whether to use log scale for y-axis. Defaults to False.
            show (bool, optional): Whether to display the plot. Defaults to True.
            ax (Axes | None, optional): Matplotlib axes to plot on. If None, creates new figure.

        Returns:
        Figure: The matplotlib figure object.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()

        iterations = range(len(self.history["best_fitness"]))
        best_fitness = self.history["best_fitness"]

        ax.plot(
            iterations, best_fitness, linewidth=2, color="blue", label="Best Fitness"
        )
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Best Fitness Value", fontsize=12)
        ax.set_title(
            f"Convergence Curve - {self.optimizer.__class__.__name__}", fontsize=14
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        if log_scale:
            ax.set_yscale("log")

        if show:
            plt.tight_layout()
            plt.show()

        return fig

    def plot_trajectory(
        self, show: bool = True, ax: Axes | None = None, max_points: int = 1000
    ) -> Figure:
        """Plot 2D trajectory of the best solution through the search space.

        This visualization shows how the best solution moves through the search space
        over iterations. Only works for 2D problems.

        Args:
            show (bool, optional): Whether to display the plot. Defaults to True.
            ax (Axes | None, optional): Matplotlib axes to plot on. If None, creates new figure.
            max_points (int, optional): Maximum number of points to plot. Defaults to 1000.

        Returns:
        Figure: The matplotlib figure object.

        Raises:
        ValueError: If optimizer dimensionality is not 2.
        """
        import matplotlib.pyplot as plt

        if self.optimizer.dim != 2:
            msg = "Trajectory plotting only works for 2D problems (dim=2)"
            raise ValueError(msg)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.get_figure()

        # Extract trajectory
        best_solutions = np.array(self.history["best_solution"])

        # Subsample if too many points
        if len(best_solutions) > max_points:
            indices = np.linspace(0, len(best_solutions) - 1, max_points, dtype=int)
            best_solutions = best_solutions[indices]

        x_coords = best_solutions[:, 0]
        y_coords = best_solutions[:, 1]

        # Plot trajectory with color gradient
        scatter = ax.scatter(
            x_coords,
            y_coords,
            c=range(len(x_coords)),
            cmap="viridis",
            s=50,
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
        )

        # Plot start and end points
        ax.plot(
            x_coords[0],
            y_coords[0],
            "go",
            markersize=12,
            label="Start",
            markeredgecolor="black",
            markeredgewidth=2,
        )
        ax.plot(
            x_coords[-1],
            y_coords[-1],
            "r*",
            markersize=15,
            label="End",
            markeredgecolor="black",
            markeredgewidth=2,
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Iteration", fontsize=11)

        ax.set_xlabel("Dimension 1", fontsize=12)
        ax.set_ylabel("Dimension 2", fontsize=12)
        ax.set_title(
            f"Search Trajectory - {self.optimizer.__class__.__name__}", fontsize=14
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        if show:
            plt.tight_layout()
            plt.show()

        return fig

    def plot_average_fitness(
        self, show_std: bool = True, show: bool = True, ax: Axes | None = None
    ) -> Figure:
        """Plot average fitness of population over iterations with standard deviation.

        This visualization shows the mean fitness of the entire population over time,
        with optional standard deviation bands to show population diversity.

        Args:
            show_std (bool, optional): Whether to show standard deviation bands. Defaults to True.
            show (bool, optional): Whether to display the plot. Defaults to True.
            ax (Axes | None, optional): Matplotlib axes to plot on. If None, creates new figure.

        Returns:
        Figure: The matplotlib figure object.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()

        iterations = range(len(self.history["population_fitness"]))
        population_fitness = self.history["population_fitness"]

        # Calculate mean and std
        mean_fitness = [np.mean(f) for f in population_fitness]
        std_fitness = [np.std(f) for f in population_fitness]

        # Plot mean fitness
        ax.plot(
            iterations, mean_fitness, linewidth=2, color="green", label="Mean Fitness"
        )

        # Plot best fitness for comparison
        best_fitness = self.history["best_fitness"]
        ax.plot(
            iterations,
            best_fitness,
            linewidth=2,
            color="blue",
            label="Best Fitness",
            linestyle="--",
        )

        # Add standard deviation bands
        if show_std:
            mean_arr = np.array(mean_fitness)
            std_arr = np.array(std_fitness)
            ax.fill_between(
                iterations,
                mean_arr - std_arr,
                mean_arr + std_arr,
                alpha=0.2,
                color="green",
                label="±1 Std Dev",
            )

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Fitness Value", fontsize=12)
        ax.set_title(
            f"Population Fitness Over Time - {self.optimizer.__class__.__name__}",
            fontsize=14,
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        if show:
            plt.tight_layout()
            plt.show()

        return fig

    def plot_all(self, save_path: str | None = None) -> None:
        """Plot all available visualizations in a single figure.

        Creates a comprehensive visualization with convergence, trajectory (if 2D),
        and average fitness plots.

        Args:
            save_path (str | None, optional): Path to save the figure. If None, displays instead.
        """
        import matplotlib.pyplot as plt

        if self.optimizer.dim == 2:
            _fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()

            # Convergence plot
            self.plot_convergence(show=False, ax=axes[0])

            # Convergence plot (log scale)
            self.plot_convergence(log_scale=True, show=False, ax=axes[1])
            axes[1].set_title(
                f"Convergence Curve (Log Scale) - {self.optimizer.__class__.__name__}",
                fontsize=14,
            )

            # Trajectory plot
            self.plot_trajectory(show=False, ax=axes[2])

            # Average fitness plot
            self.plot_average_fitness(show=False, ax=axes[3])
        else:
            _fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Convergence plot
            self.plot_convergence(show=False, ax=axes[0])

            # Convergence plot (log scale)
            self.plot_convergence(log_scale=True, show=False, ax=axes[1])
            axes[1].set_title(
                f"Convergence Curve (Log Scale) - {self.optimizer.__class__.__name__}",
                fontsize=14,
            )

            # Average fitness plot
            self.plot_average_fitness(show=False, ax=axes[2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
