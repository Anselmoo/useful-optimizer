"""Unit tests for visualization module."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from opt.benchmark.functions import shifted_ackley
from opt.swarm_intelligence.particle_swarm import ParticleSwarm
from opt.visualization import StabilityResults
from opt.visualization import Visualizer
from opt.visualization import run_stability_test


# Use non-interactive backend for testing
matplotlib.use("Agg")


class TestVisualizer:
    """Tests for the Visualizer class."""

    @pytest.fixture
    def optimizer_with_history(self):
        """Create an optimizer with history tracking."""
        pso = ParticleSwarm(
            func=shifted_ackley,
            lower_bound=-5,
            upper_bound=5,
            dim=2,
            max_iter=20,
            track_history=True,
            population_size=20,
            seed=42,
        )
        pso.search()
        return pso

    @pytest.fixture
    def optimizer_without_history(self):
        """Create an optimizer without history tracking."""
        pso = ParticleSwarm(
            func=shifted_ackley,
            lower_bound=-5,
            upper_bound=5,
            dim=2,
            max_iter=10,
            track_history=False,
            seed=42,
        )
        pso.search()
        return pso

    def test_visualizer_initialization(self, optimizer_with_history):
        """Test that Visualizer initializes correctly with history."""
        viz = Visualizer(optimizer_with_history)
        assert viz.optimizer == optimizer_with_history
        assert viz.history == optimizer_with_history.history
        assert len(viz.history["best_fitness"]) > 0

    def test_visualizer_without_history_raises_error(self, optimizer_without_history):
        """Test that Visualizer raises error when history is not tracked."""
        with pytest.raises(ValueError, match="track_history=True"):
            Visualizer(optimizer_without_history)

    def test_plot_convergence(self, optimizer_with_history):
        """Test convergence plot generation."""
        viz = Visualizer(optimizer_with_history)
        fig = viz.plot_convergence(show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_convergence_log_scale(self, optimizer_with_history):
        """Test convergence plot with log scale."""
        viz = Visualizer(optimizer_with_history)
        fig = viz.plot_convergence(log_scale=True, show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_trajectory(self, optimizer_with_history):
        """Test trajectory plot generation for 2D problems."""
        viz = Visualizer(optimizer_with_history)
        fig = viz.plot_trajectory(show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_trajectory_non_2d_raises_error(self):
        """Test that trajectory plot raises error for non-2D problems."""
        from opt.benchmark.functions import sphere

        pso = ParticleSwarm(
            func=sphere,
            lower_bound=-5,
            upper_bound=5,
            dim=3,  # 3D problem
            max_iter=10,
            track_history=True,
            seed=42,
        )
        pso.search()
        viz = Visualizer(pso)

        with pytest.raises(ValueError, match="2D problems"):
            viz.plot_trajectory(show=False)

    def test_plot_average_fitness(self, optimizer_with_history):
        """Test average fitness plot generation."""
        viz = Visualizer(optimizer_with_history)
        fig = viz.plot_average_fitness(show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_average_fitness_without_std(self, optimizer_with_history):
        """Test average fitness plot without std deviation bands."""
        viz = Visualizer(optimizer_with_history)
        fig = viz.plot_average_fitness(show_std=False, show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_all_2d(self, optimizer_with_history):
        """Test plot_all for 2D problems."""
        viz = Visualizer(optimizer_with_history)
        viz.plot_all(save_path="/tmp/test_plot_all_2d.png")
        plt.close("all")

    def test_plot_all_3d(self):
        """Test plot_all for non-2D problems."""
        from opt.benchmark.functions import sphere

        pso = ParticleSwarm(
            func=sphere,
            lower_bound=-5,
            upper_bound=5,
            dim=3,
            max_iter=10,
            track_history=True,
            seed=42,
        )
        pso.search()
        viz = Visualizer(pso)
        viz.plot_all(save_path="/tmp/test_plot_all_3d.png")
        plt.close("all")


class TestStabilityResults:
    """Tests for the StabilityResults class."""

    @pytest.fixture
    def sample_results(self):
        """Create sample stability results."""
        solutions = [
            np.array([1.0, 0.5]),
            np.array([1.1, 0.6]),
            np.array([0.9, 0.4]),
        ]
        fitness_values = [0.01, 0.02, 0.015]
        seeds = [42, 123, 456]
        return StabilityResults(
            optimizer_name="ParticleSwarm",
            function_name="shifted_ackley",
            solutions=solutions,
            fitness_values=fitness_values,
            seeds=seeds,
        )

    def test_stability_results_initialization(self, sample_results):
        """Test StabilityResults initialization."""
        assert sample_results.optimizer_name == "ParticleSwarm"
        assert sample_results.function_name == "shifted_ackley"
        assert len(sample_results.solutions) == 3
        assert len(sample_results.fitness_values) == 3
        assert len(sample_results.seeds) == 3

    def test_summary(self, sample_results):
        """Test summary statistics generation."""
        summary = sample_results.summary()
        assert "mean" in summary
        assert "std" in summary
        assert "min" in summary
        assert "max" in summary
        assert "median" in summary
        assert "q25" in summary
        assert "q75" in summary
        assert summary["min"] == 0.01
        assert summary["max"] == 0.02

    def test_print_summary(self, sample_results, capsys):
        """Test print_summary output."""
        sample_results.print_summary()
        captured = capsys.readouterr()
        assert "Stability Test Results" in captured.out
        assert "ParticleSwarm" in captured.out
        assert "shifted_ackley" in captured.out

    def test_plot_boxplot(self, sample_results):
        """Test box plot generation."""
        fig = sample_results.plot_boxplot(show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_histogram(self, sample_results):
        """Test histogram generation."""
        fig = sample_results.plot_histogram(show=False)
        assert fig is not None
        plt.close(fig)


class TestStabilityTesting:
    """Tests for stability testing functions."""

    def test_run_stability_test_with_seeds(self):
        """Test stability test with specific seeds."""
        results = run_stability_test(
            optimizer_class=ParticleSwarm,
            func=shifted_ackley,
            lower_bound=-5,
            upper_bound=5,
            dim=2,
            max_iter=10,
            seeds=[42, 123],
            verbose=False,
            population_size=10,
        )

        assert isinstance(results, StabilityResults)
        assert len(results.fitness_values) == 2
        assert len(results.solutions) == 2
        assert results.seeds == [42, 123]

    def test_run_stability_test_with_n_runs(self):
        """Test stability test with n_runs parameter."""
        results = run_stability_test(
            optimizer_class=ParticleSwarm,
            func=shifted_ackley,
            lower_bound=-5,
            upper_bound=5,
            dim=2,
            max_iter=10,
            n_runs=3,
            verbose=False,
            population_size=10,
        )

        assert isinstance(results, StabilityResults)
        assert len(results.fitness_values) == 3
        assert len(results.solutions) == 3

    def test_run_stability_test_reproducibility(self):
        """Test that same seeds produce same results."""
        results1 = run_stability_test(
            optimizer_class=ParticleSwarm,
            func=shifted_ackley,
            lower_bound=-5,
            upper_bound=5,
            dim=2,
            max_iter=10,
            seeds=[42],
            verbose=False,
            population_size=10,
        )

        results2 = run_stability_test(
            optimizer_class=ParticleSwarm,
            func=shifted_ackley,
            lower_bound=-5,
            upper_bound=5,
            dim=2,
            max_iter=10,
            seeds=[42],
            verbose=False,
            population_size=10,
        )

        assert results1.fitness_values[0] == results2.fitness_values[0]
        np.testing.assert_array_equal(results1.solutions[0], results2.solutions[0])


class TestHistoryTracking:
    """Tests for history tracking in optimizers."""

    def test_particle_swarm_tracks_history(self):
        """Test that ParticleSwarm correctly tracks history."""
        pso = ParticleSwarm(
            func=shifted_ackley,
            lower_bound=-5,
            upper_bound=5,
            dim=2,
            max_iter=10,
            track_history=True,
            seed=42,
        )

        pso.search()

        assert pso.track_history is True
        assert len(pso.history["best_fitness"]) == 11  # max_iter + 1
        assert len(pso.history["best_solution"]) == 11
        assert len(pso.history["population_fitness"]) == 11
        assert len(pso.history["population"]) == 11

    def test_particle_swarm_no_history_by_default(self):
        """Test that ParticleSwarm doesn't track history by default."""
        pso = ParticleSwarm(
            func=shifted_ackley,
            lower_bound=-5,
            upper_bound=5,
            dim=2,
            max_iter=10,
            seed=42,
        )

        pso.search()

        assert pso.track_history is False
        assert pso.history == {}
