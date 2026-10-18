"""Tests for the budgeted evaluation protocol."""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.protocol import BASELINE
from benchmarks.protocol import FUNCTIONS
from benchmarks.protocol import TARGETS
from benchmarks.protocol import BudgetExhausted
from benchmarks.protocol import BudgetedObjective
from benchmarks.protocol import budget_for
from benchmarks.protocol import checkpoint_grid
from benchmarks.protocol import discover_optimizers
from benchmarks.protocol import probe_max_iter
from benchmarks.protocol import run_cell
from benchmarks.protocol import run_once
from opt.abstract import AbstractOptimizer


SPHERE = FUNCTIONS["sphere"]
OPTIMIZERS = discover_optimizers()


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x) ** 2))


class _Swallower(AbstractOptimizer):
    """Evaluates forever and tries to hide every exception it can."""

    def search(self) -> tuple[np.ndarray, float]:
        rng = np.random.default_rng(self.seed)
        best = np.zeros(self.dim)
        for _ in range(10**9):
            try:
                self.func(rng.uniform(self.lower_bound, self.upper_bound, self.dim))
            except Exception:
                pass
        return best, 0.0


class TestShiftedFunctions:
    @pytest.mark.parametrize("name", list(FUNCTIONS))
    @pytest.mark.parametrize("dim", [2, 5, 20])
    def test_optimum_is_where_the_shift_puts_it(self, name: str, dim: int) -> None:
        config = FUNCTIONS[name]
        objective = config.objective(dim)
        assert objective(config.optimum(dim)) == pytest.approx(config.f_opt, abs=1e-9)

    @pytest.mark.parametrize("name", list(FUNCTIONS))
    def test_optimum_is_inside_the_domain_and_off_centre(self, name: str) -> None:
        config = FUNCTIONS[name]
        optimum = config.optimum(10)
        centre = (config.upper + config.lower) / 2
        half = (config.upper - config.lower) / 2
        assert np.all(np.abs(optimum - centre) <= 0.8 * half + 1e-12)
        # The optimum is not at the origin or the centre, where averaging
        # optimizers would find it by construction.
        assert np.linalg.norm(optimum - centre) > 0.1 * half
        assert config.objective(10)(np.full(10, centre)) > 1e-3

    def test_shift_is_deterministic_and_differs_between_functions(self) -> None:
        a = FUNCTIONS["sphere"]
        assert np.array_equal(a.optimum(5), a.optimum(5))
        assert not np.array_equal(a.optimum(5), FUNCTIONS["rastrigin"].optimum(5))
        assert not np.array_equal(a.optimum(5)[:2], a.optimum(2))

    def test_shift_translates_the_native_function(self) -> None:
        config = FUNCTIONS["rosenbrock"]
        x = np.array([0.3, -1.2, 2.0])
        assert config.objective(3)(x) == pytest.approx(config.func(x - config.shift(3)))


class TestBudgetedObjective:
    def test_cuts_off_exactly_at_the_budget(self) -> None:
        obj = BudgetedObjective(_sphere, -5, 5, 2, budget=5)
        for _ in range(5):
            obj(np.array([1.0, 1.0]))
        assert obj.n_evaluations == 5
        with pytest.raises(BudgetExhausted) as info:
            obj(np.array([1.0, 1.0]))
        assert info.value.reason == "budget"
        assert obj.n_evaluations == 5

    def test_budget_signal_is_not_an_exception(self) -> None:
        assert not issubclass(BudgetExhausted, Exception)

    def test_optimizer_cannot_swallow_the_budget(self) -> None:
        record = run_once(
            _Swallower, SPHERE, 2, seed=1, max_iter=10, budget=100, time_cap=None
        )
        assert record["n_evaluations"] == 100
        assert record["exhausted"] is True
        assert record["status"] == "ok"

    def test_time_cap_truncates(self) -> None:
        def slow(x: np.ndarray) -> float:
            total = 0.0
            for _ in range(20000):
                total += float(np.sum(x))
            return float(np.sum(x**2))

        obj = BudgetedObjective(slow, -5, 5, 2, budget=10**9, time_cap=0.2)
        with pytest.raises(BudgetExhausted) as info:
            while True:
                obj(np.array([1.0, 2.0]))
        assert info.value.reason == "time"
        assert obj.n_evaluations > 0

    def test_returns_the_original_value_type(self) -> None:
        obj = BudgetedObjective(lambda x: np.float64(1.5), -5, 5, 2, budget=3)
        assert isinstance(obj(np.zeros(2)), np.float64)

    def test_hits_are_the_first_evaluation_reaching_each_target(self) -> None:
        points = [
            np.array([10.0, 0.0]),  # f = 100
            np.array([2.0, 0.0]),  # f = 4   -> reaches 1e1
            np.array([0.5, 0.0]),  # f = 0.25 -> reaches 1e0
            np.array([2.0, 0.0]),  # worse again: no new hits
            np.array([1e-5, 0.0]),  # f = 1e-10 -> reaches everything left
        ]
        obj = BudgetedObjective(_sphere, -50, 50, 2, budget=10)
        for p in points:
            obj(p)
        hits = dict(zip(TARGETS, obj.hits, strict=True))
        assert hits[1e1] == 2
        assert hits[1e0] == 3
        assert hits[1e-1] == 5
        assert hits[1e-8] == 5

    def test_best_so_far_at_checkpoints_is_monotone_and_ends_at_the_best(self) -> None:
        rng = np.random.default_rng(0)
        cps = checkpoint_grid(200)
        obj = BudgetedObjective(_sphere, -5, 5, 3, budget=200, checkpoints=cps)
        for _ in range(130):  # stops early: the tail must hold the final best
            obj(rng.uniform(-5, 5, 3))
        result = obj.result()
        best_at = np.array(result["best_at"], dtype=float)
        assert len(best_at) == len(cps)
        assert np.all(np.diff(best_at) <= 0)
        assert best_at[-1] == pytest.approx(result["best_fitness"])

    def test_dispersion_is_zero_for_identical_points_and_normalised(self) -> None:
        cps = [2, 40]
        same = BudgetedObjective(_sphere, 0, 10, 2, budget=40, checkpoints=cps)
        for _ in range(40):
            same(np.array([3.0, 3.0]))
        assert same.result()["dispersion_at"] == [0.0, 0.0]

        spread = BudgetedObjective(_sphere, 0, 10, 2, budget=40, checkpoints=cps)
        rng = np.random.default_rng(1)
        for _ in range(40):
            spread(rng.uniform(0, 10, 2))
        late = spread.result()["dispersion_at"][-1]
        assert 0.2 < late < 0.35  # std of U(0,1) is 0.289

    def test_trajectory_is_capped_and_ends_at_the_best(self) -> None:
        obj = BudgetedObjective(_sphere, -5, 5, 2, budget=10**5, track_trajectory=True)
        for k in range(400, 0, -1):  # every evaluation improves
            obj(np.array([k / 100, 0.0]))
        trajectory = obj.result(max_trajectory=100)["trajectory"]
        assert len(trajectory) <= 100
        assert trajectory[-1][0] == 400
        assert [row[0] for row in trajectory] == sorted(row[0] for row in trajectory)

    def test_non_finite_values_do_not_poison_the_record(self) -> None:
        obj = BudgetedObjective(lambda x: math.nan, -5, 5, 2, budget=3)
        for _ in range(3):
            obj(np.zeros(2))
        record = obj.result()
        assert record["best_fitness"] is None
        assert record["n_evaluations"] == 3


class TestGrid:
    def test_budget_scales_with_dimension(self) -> None:
        assert budget_for(5) == 10_000

    def test_checkpoints_are_increasing_and_end_at_the_budget(self) -> None:
        grid = checkpoint_grid(40_000)
        assert grid[0] == 1
        assert grid[-1] == 40_000
        assert grid == sorted(set(grid))


class TestDiscovery:
    def test_finds_the_single_objective_library_and_the_baseline(self) -> None:
        assert len(OPTIMIZERS) > 100
        assert BASELINE in OPTIMIZERS
        for name in ("MOEAD", "NSGAII", "SPEA2"):
            assert name not in OPTIMIZERS


class TestRunCell:
    def test_cell_record_structure_and_seeds(self) -> None:
        cell = run_cell(
            OPTIMIZERS["ParticleSwarm"],
            SPHERE,
            2,
            n_runs=3,
            function_index=0,
            budget_per_dim=100,
        )
        assert cell["protocol"]["budget"] == 200
        assert cell["protocol"]["runs_completed"] == 3
        assert [r["seed"] for r in cell["runs"]] == [0, 1, 2]
        assert all(r["n_evaluations"] == 200 for r in cell["runs"])
        assert len(cell["runs"][0]["best_at"]) == len(cell["protocol"]["checkpoints"])
        assert cell["runs"][0]["trajectory"]  # 2D keeps trajectories of the first runs

    def test_seeds_differ_between_functions(self) -> None:
        a = run_cell(
            OPTIMIZERS["RandomSearch"],
            SPHERE,
            2,
            n_runs=1,
            function_index=0,
            budget_per_dim=50,
        )
        b = run_cell(
            OPTIMIZERS["RandomSearch"],
            SPHERE,
            2,
            n_runs=1,
            function_index=3,
            budget_per_dim=50,
        )
        assert a["runs"][0]["seed"] != b["runs"][0]["seed"]

    def test_probe_sets_max_iter_to_use_the_budget(self) -> None:
        iters = probe_max_iter(OPTIMIZERS["ParticleSwarm"], SPHERE, 2, budget=600)
        record = run_once(
            OPTIMIZERS["ParticleSwarm"], SPHERE, 2, seed=0, max_iter=iters, budget=600
        )
        assert record["exhausted"]
        assert record["n_evaluations"] == 600


def _small_run(name: str, seed: int) -> dict:
    cls = OPTIMIZERS[name]
    iters = probe_max_iter(cls, SPHERE, 3, seed=seed, budget=240)
    return run_once(cls, SPHERE, 3, seed=seed, max_iter=iters, budget=240, time_cap=5)


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_every_optimizer_respects_the_budget_and_the_seed(name: str) -> None:
    first = _small_run(name, 1)
    again = _small_run(name, 1)
    other = _small_run(name, 2)

    assert first["status"] in {"ok", "truncated_time"}, first["error"]
    assert 0 < first["n_evaluations"] <= 240
    assert first["best_fitness"] is not None

    if first["status"] == "truncated_time":
        return  # cut by the wall clock: counts depend on the machine, not the seed
    # reproducible for one seed ...
    assert first["best_fitness"] == again["best_fitness"]
    assert first["n_evaluations"] == again["n_evaluations"]
    # ... and different for another, unless both hit the exact optimum (an optimizer
    # that ignored the seed would give the same curve and the same point)
    if first["best_fitness"] == 0.0 and other["best_fitness"] == 0.0:
        return
    assert (first["best_fitness"], first["best_at"]) != (
        other["best_fitness"],
        other["best_at"],
    ), f"{name} gives identical runs for different seeds"
