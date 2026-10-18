"""Tests for the RandomSearch baseline optimizer."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from opt.benchmark.functions import shifted_ackley
from opt.benchmark.functions import sphere
from opt.classical.random_search import RandomSearch


def _make(**overrides: object) -> RandomSearch:
    params: dict[str, object] = {
        "func": shifted_ackley,
        "lower_bound": -5.0,
        "upper_bound": 5.0,
        "dim": 3,
        "max_iter": 40,
        "seed": 7,
    }
    params.update(overrides)
    return RandomSearch(**params)  # type: ignore[arg-type]


def test_same_seed_gives_identical_result() -> None:
    sol_a, fit_a = _make(population_size=3).search()
    sol_b, fit_b = _make(population_size=3).search()
    assert np.array_equal(sol_a, sol_b)
    assert fit_a == fit_b


def test_different_seeds_give_different_results() -> None:
    sol_a, fit_a = _make(seed=1).search()
    sol_b, fit_b = _make(seed=2).search()
    assert not np.array_equal(sol_a, sol_b)
    assert fit_a != fit_b


def test_solution_within_bounds_and_types() -> None:
    solution, fitness = _make(lower_bound=-2.0, upper_bound=3.0).search()
    assert isinstance(solution, np.ndarray)
    assert solution.shape == (3,)
    assert isinstance(fitness, float)
    assert bool(np.all(solution >= -2.0))
    assert bool(np.all(solution <= 3.0))


def test_fitness_matches_returned_solution() -> None:
    solution, fitness = _make().search()
    assert fitness == pytest.approx(shifted_ackley(solution))


def test_exact_evaluation_count() -> None:
    optimizer = _make(max_iter=25)
    optimizer.search()
    assert optimizer.n_evaluations == 25


@pytest.mark.parametrize("population_size", [2, 5, 10])
def test_population_size_scales_evaluations(population_size: int) -> None:
    optimizer = _make(max_iter=12, population_size=population_size)
    optimizer.search()
    assert optimizer.n_evaluations == population_size * 12


def test_history_has_one_finite_entry_per_iteration_plus_final() -> None:
    optimizer = _make(max_iter=15, population_size=4, track_history=True)
    optimizer.search()
    best = optimizer.history["best_fitness"]
    assert len(best) == 15 + 1
    assert all(np.isfinite(v) for v in best)
    # best-so-far is monotonically non-increasing
    assert all(b <= a for a, b in pairwise(best))
    assert len(optimizer.history["best_solution"]) == len(best)
    assert np.asarray(optimizer.history["population"]).shape == (16, 4, 3)


def test_history_empty_without_tracking() -> None:
    optimizer = _make()
    optimizer.search()
    assert optimizer.history == {}


def test_get_close_to_zero_on_sphere_with_large_budget() -> None:
    optimizer = RandomSearch(
        func=sphere,
        lower_bound=-5.0,
        upper_bound=5.0,
        dim=2,
        max_iter=2000,
        population_size=5,
        seed=0,
    )
    _, fitness = optimizer.search()
    assert fitness < 0.05


def test_early_stop_when_target_reached() -> None:
    optimizer = RandomSearch(
        func=lambda _x: 0.0,
        lower_bound=-1.0,
        upper_bound=1.0,
        dim=2,
        max_iter=100,
        population_size=3,
        seed=0,
        f_opt=0.0,
        target_precision=1e-8,
    )
    _, fitness = optimizer.search()
    assert fitness == 0.0
    assert optimizer.converged
    assert optimizer.n_evaluations == 3  # a single round
    assert optimizer.evaluations_to_target == 3


def test_early_stop_history_stays_finite() -> None:
    optimizer = RandomSearch(
        func=lambda _x: 0.0,
        lower_bound=-1.0,
        upper_bound=1.0,
        dim=2,
        max_iter=100,
        seed=0,
        f_opt=0.0,
        track_history=True,
    )
    optimizer.search()
    best = optimizer.history["best_fitness"]
    assert 1 <= len(best) <= 2
    assert all(np.isfinite(v) for v in best)


def test_no_early_stop_without_f_opt() -> None:
    optimizer = RandomSearch(
        func=lambda _x: 0.0,
        lower_bound=-1.0,
        upper_bound=1.0,
        dim=2,
        max_iter=10,
        seed=0,
    )
    optimizer.search()
    assert optimizer.n_evaluations == 10


def test_does_not_touch_global_numpy_rng() -> None:
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    _make().search()
    assert np.random.random() == expected


@pytest.mark.parametrize(
    "overrides", [{"max_iter": 0}, {"population_size": 0}], ids=["iter", "pop"]
)
def test_invalid_budget_raises(overrides: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _make(**overrides).search()
