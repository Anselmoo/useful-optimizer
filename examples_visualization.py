"""Example usage of the visualization module for optimization algorithms.

This script demonstrates how to use the visualization capabilities
including convergence plots, trajectory plots, average fitness tracking,
and stability testing.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from opt.benchmark.functions import rosenbrock
from opt.benchmark.functions import shifted_ackley
from opt.benchmark.functions import sphere
from opt.evolutionary.genetic_algorithm import GeneticAlgorithm
from opt.swarm_intelligence.particle_swarm import ParticleSwarm
from opt.visualization import Visualizer
from opt.visualization import compare_optimizers_stability
from opt.visualization import run_stability_test


def example_basic_visualization() -> None:
    """Example 1: Basic visualization with ParticleSwarm."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Visualization with Particle Swarm Optimization")
    print("=" * 70)

    # Create optimizer with history tracking enabled
    pso = ParticleSwarm(
        func=shifted_ackley,
        lower_bound=-5,
        upper_bound=5,
        dim=2,
        max_iter=100,
        track_history=True,  # Enable history tracking
        population_size=30,
        seed=42,
    )

    # Run optimization
    best_solution, best_fitness = pso.search()
    print("\nOptimization completed!")
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness:.6f}")

    # Create visualizer
    viz = Visualizer(pso)

    # Plot convergence curve
    print("\nGenerating convergence plot...")
    viz.plot_convergence(show=False)
    plt.savefig("example_convergence.png", dpi=300, bbox_inches="tight")
    print("✓ Saved to example_convergence.png")

    # Plot trajectory (2D only)
    print("Generating trajectory plot...")
    viz.plot_trajectory(show=False)
    plt.savefig("example_trajectory.png", dpi=300, bbox_inches="tight")
    print("✓ Saved to example_trajectory.png")

    # Plot average fitness
    print("Generating average fitness plot...")
    viz.plot_average_fitness(show=False)
    plt.savefig("example_average_fitness.png", dpi=300, bbox_inches="tight")
    print("✓ Saved to example_average_fitness.png")

    # Plot all visualizations in one figure
    print("Generating comprehensive plot...")
    viz.plot_all(save_path="example_all_plots.png")
    print("✓ Saved to example_all_plots.png")

    plt.close("all")


def example_stability_testing() -> None:
    """Example 2: Stability testing with multiple seeds."""
    print("\n" + "=" * 70)
    print("Example 2: Stability Testing with Multiple Seeds")
    print("=" * 70)

    # Run stability test with specific seeds
    results = run_stability_test(
        optimizer_class=ParticleSwarm,
        func=shifted_ackley,
        lower_bound=-5,
        upper_bound=5,
        dim=2,
        max_iter=100,
        seeds=[42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066],
        verbose=True,
        population_size=30,
    )

    # Get statistical summary
    print("\nGenerating statistical summary...")
    summary = results.summary()
    print(f"Mean fitness: {summary['mean']:.6f}")
    print(f"Std deviation: {summary['std']:.6f}")
    print(f"Coefficient of variation: {summary['std'] / summary['mean']:.4f}")

    # Generate box plot
    print("\nGenerating box plot...")
    results.plot_boxplot(show=False, save_path="example_stability_boxplot.png")
    print("✓ Saved to example_stability_boxplot.png")

    # Generate histogram
    print("Generating histogram...")
    results.plot_histogram(show=False, save_path="example_stability_histogram.png")
    print("✓ Saved to example_stability_histogram.png")

    plt.close("all")


def example_optimizer_comparison() -> None:
    """Example 3: Compare stability of multiple optimizers."""
    print("\n" + "=" * 70)
    print("Example 3: Compare Stability of Multiple Optimizers")
    print("=" * 70)

    # Compare ParticleSwarm and GeneticAlgorithm
    results_dict, _fig = compare_optimizers_stability(
        optimizer_classes=[ParticleSwarm, GeneticAlgorithm],
        func=sphere,
        lower_bound=-10,
        upper_bound=10,
        dim=2,
        max_iter=100,
        n_runs=10,
        show=False,
        save_path="example_optimizer_comparison.png",
    )

    print("\nComparison Results:")
    print("-" * 70)
    for name, results in results_dict.items():
        summary = results.summary()
        print(f"\n{name}:")
        print(f"  Mean: {summary['mean']:.6f} ± {summary['std']:.6f}")
        print(f"  Min:  {summary['min']:.6f}")
        print(f"  Max:  {summary['max']:.6f}")

    print("\n✓ Saved to example_optimizer_comparison.png")
    plt.close("all")


def example_convergence_log_scale() -> None:
    """Example 4: Convergence plot with log scale."""
    print("\n" + "=" * 70)
    print("Example 4: Convergence Plot with Log Scale")
    print("=" * 70)

    pso = ParticleSwarm(
        func=rosenbrock,
        lower_bound=-5,
        upper_bound=5,
        dim=2,
        max_iter=200,
        track_history=True,
        population_size=50,
        seed=42,
    )

    _best_solution, best_fitness = pso.search()
    print(f"\nBest fitness: {best_fitness:.6f}")

    viz = Visualizer(pso)

    # Create side-by-side comparison
    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Regular scale
    viz.plot_convergence(show=False, ax=axes[0])
    axes[0].set_title("Convergence Curve - Linear Scale", fontsize=14)

    # Log scale
    viz.plot_convergence(log_scale=True, show=False, ax=axes[1])
    axes[1].set_title("Convergence Curve - Log Scale", fontsize=14)

    plt.tight_layout()
    plt.savefig("example_log_scale_comparison.png", dpi=300, bbox_inches="tight")
    print("✓ Saved to example_log_scale_comparison.png")

    plt.close("all")


def example_custom_visualization() -> None:
    """Example 5: Custom visualization with matplotlib integration."""
    print("\n" + "=" * 70)
    print("Example 5: Custom Visualization with Matplotlib Integration")
    print("=" * 70)

    # Run optimizer with history
    pso = ParticleSwarm(
        func=shifted_ackley,
        lower_bound=-5,
        upper_bound=5,
        dim=2,
        max_iter=100,
        track_history=True,
        population_size=30,
        seed=42,
    )
    pso.search()

    # Create custom multi-panel figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Add visualizations to custom grid
    viz = Visualizer(pso)

    ax1 = fig.add_subplot(gs[0, :])
    viz.plot_convergence(show=False, ax=ax1)

    ax2 = fig.add_subplot(gs[1, 0])
    viz.plot_trajectory(show=False, ax=ax2)

    ax3 = fig.add_subplot(gs[1, 1])
    viz.plot_average_fitness(show=False, ax=ax3)

    ax4 = fig.add_subplot(gs[2, :])
    viz.plot_convergence(log_scale=True, show=False, ax=ax4)
    ax4.set_title("Convergence Curve (Log Scale)", fontsize=14)

    plt.suptitle(
        "Custom Visualization Dashboard - Particle Swarm Optimization",
        fontsize=16,
        fontweight="bold",
    )

    plt.savefig("example_custom_dashboard.png", dpi=300, bbox_inches="tight")
    print("✓ Saved to example_custom_dashboard.png")

    plt.close("all")


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION VISUALIZATION EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates the visualization capabilities")
    print("of the useful-optimizer package.")

    # Set matplotlib to non-interactive backend
    mpl.use("Agg")

    # Run all examples
    example_basic_visualization()
    example_stability_testing()
    example_optimizer_comparison()
    example_convergence_log_scale()
    example_custom_visualization()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - example_convergence.png")
    print("  - example_trajectory.png")
    print("  - example_average_fitness.png")
    print("  - example_all_plots.png")
    print("  - example_stability_boxplot.png")
    print("  - example_stability_histogram.png")
    print("  - example_optimizer_comparison.png")
    print("  - example_log_scale_comparison.png")
    print("  - example_custom_dashboard.png")
    print("\nFeel free to open and examine these visualizations!")


if __name__ == "__main__":
    main()
