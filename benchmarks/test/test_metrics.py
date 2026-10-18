"""Tests for the benchmark metrics against hand-computed examples."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from scipy import stats

from benchmarks import metrics


def run(hits: list[int | None], n_evaluations: int, best: float = 1.0, **extra) -> dict:
    return {
        "status": "ok",
        "best_fitness": best,
        "hits": hits,
        "n_evaluations": n_evaluations,
        "elapsed": 0.5,
        **extra,
    }


class TestErt:
    def test_coco_definition_with_unsuccessful_runs(self) -> None:
        runs = [run([100], 500), run([300], 500), run([None], 1000)]
        ert, sp1 = metrics.ert_and_sp1(runs, 0)
        assert ert == pytest.approx((100 + 300 + 1000) / 2)
        assert sp1 == pytest.approx(((100 + 300) / 2) / (2 / 3))

    def test_all_successful_is_the_mean_hitting_time(self) -> None:
        runs = [run([100], 500), run([300], 500)]
        ert, sp1 = metrics.ert_and_sp1(runs, 0)
        assert ert == pytest.approx(200)
        assert sp1 == pytest.approx(200)

    def test_none_without_success(self) -> None:
        assert metrics.ert_and_sp1([run([None], 100)], 0) == (None, None)

    def test_success_rates(self) -> None:
        runs = [
            run([1, None], 10),
            run([2, 3], 10),
            run([None, None], 10),
            run([5, None], 10),
        ]
        assert metrics.success_rates(runs, 2) == [0.75, 0.25]
        assert metrics.success_rates([], 2) == [None, None]


class TestEcdf:
    def test_counts_run_target_pairs_and_is_monotone(self) -> None:
        # two runs x two targets = 4 pairs; hits at 20, 40, 80 evaluations, one never
        runs = [run([20, 80], 100), run([40, None], 100)]
        budgets = np.array([5.0, 10.0, 20.0, 40.0])  # evals / dim with dim=2 -> 10..80
        values = metrics.ecdf(runs, 2, budgets)
        assert values == [0.0, 0.25, 0.5, 0.75]
        assert values == sorted(values)

    def test_budget_grid_spans_one_to_the_budget_per_dim(self) -> None:
        grid = metrics.ecdf_budgets(2000, 2)
        assert grid[0] == pytest.approx(1)
        assert grid[-1] == pytest.approx(1000)
        assert len(grid) == metrics.ECDF_POINTS


class TestStatistics:
    def test_summary(self) -> None:
        result = metrics.summary_stats(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert result["median"] == 3.0
        assert result["q1"] == 2.0
        assert result["q3"] == 4.0
        assert result["mean"] == 3.0
        assert metrics.summary_stats(np.array([])) is None

    def test_failed_runs_are_excluded(self) -> None:
        cell = {"runs": [run([1], 1, best=2.0), {**run([1], 1), "status": "failed"}]}
        assert list(metrics.final_errors(cell)) == [2.0]

    def test_errors_are_never_negative(self) -> None:
        cell = {"runs": [run([1], 1, best=-1e-12)]}
        assert list(metrics.final_errors(cell)) == [0.0]

    def test_quantile_curves(self) -> None:
        runs = [
            {"best_at": [4.0, 2.0, None]},
            {"best_at": [2.0, 1.0, 0.5]},
            {"best_at": [3.0, 3.0, 1.5]},
        ]
        curves = metrics.quantile_curves(runs, "best_at", subtract_optimum=True)
        assert curves["median"] == [3.0, 2.0, 1.0]
        assert curves["q1"][0] == 2.5
        assert curves["q3"][2] == pytest.approx(1.25)
        assert metrics.quantile_curves([], "best_at", subtract_optimum=True) is None

    def test_time_stats(self) -> None:
        runs = [
            {"elapsed": 1.0, "n_evaluations": 1000},
            {"elapsed": 3.0, "n_evaluations": 1000},
        ]
        result = metrics.time_stats(runs)
        assert result["median_run_seconds"] == 2.0
        assert result["median_us_per_evaluation"] == pytest.approx(2000.0)


class TestHolm:
    def test_known_values(self) -> None:
        assert metrics.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])

    def test_caps_at_one_and_keeps_order(self) -> None:
        assert metrics.holm([0.6, 0.9]) == [1.0, 1.0]
        assert metrics.holm([]) == []

    def test_never_below_the_raw_p_value(self) -> None:
        rng = np.random.default_rng(3)
        p = list(rng.uniform(0, 1, 20))
        assert all(a >= b for a, b in zip(metrics.holm(p), p, strict=True))


class TestCliffsDelta:
    def test_matches_brute_force_and_mann_whitney(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(0, 1, 15)
        b = rng.normal(0.8, 1, 12)
        expected = sum(int(x < y) - int(x > y) for x, y in itertools.product(a, b)) / (
            len(a) * len(b)
        )
        assert metrics.cliffs_delta(a, b) == pytest.approx(expected)
        u_a = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
        # U counts wins of `a` being larger; delta here counts lower-is-better
        assert metrics.cliffs_delta(a, b) == pytest.approx(
            1 - 2 * u_a / (len(a) * len(b))
        )

    def test_extremes(self) -> None:
        assert metrics.cliffs_delta(np.array([1, 2]), np.array([3, 4])) == 1.0
        assert metrics.cliffs_delta(np.array([3, 4]), np.array([1, 2])) == -1.0
        assert metrics.cliffs_delta(np.array([1, 1]), np.array([1, 1])) == 0.0


class TestBaselineComparison:
    def test_clearly_better_candidate(self) -> None:
        candidate = np.array([1e-3, 2e-3, 1e-3, 3e-3, 2e-3, 1e-3])
        baseline = np.array([1.0, 2.0, 1.5, 3.0, 2.5, 1.2])
        result = metrics.compare_to_baseline(candidate, baseline)
        assert result["delta"] == 1.0
        assert result["p"] < 0.01

    def test_worse_candidate_has_a_large_p(self) -> None:
        result = metrics.compare_to_baseline(
            np.array([5.0, 6.0, 7.0]), np.array([1.0, 2.0, 3.0])
        )
        assert result["p"] > 0.9
        assert result["delta"] == -1.0

    def test_solved_runs_tie_at_the_floor(self) -> None:
        # both reach the optimum up to numerical noise: no evidence either way
        result = metrics.compare_to_baseline(
            np.array([1e-15, 1e-12, 0.0]), np.array([1e-14, 0.0, 1e-13])
        )
        assert result["p"] == 1.0
        assert result["delta"] == 0.0

    def test_empty_samples(self) -> None:
        assert metrics.compare_to_baseline(np.array([]), np.array([1.0])) is None


class TestRanks:
    def test_ranks_with_ties(self) -> None:
        medians = {
            "f-2": {"A": 1e-9, "B": 1e-9, "C": 0.5, "D": 2.0},
            "f-5": {"A": 3.0, "B": 1.0, "C": 2.0, "D": 4.0},
        }
        per_optimizer, mean_rank = metrics.average_ranks(medians)
        assert per_optimizer["A"]["f-2"] == 1.5
        assert per_optimizer["B"]["f-2"] == 1.5
        assert per_optimizer["D"]["f-5"] == 4.0
        assert mean_rank["A"] == pytest.approx((1.5 + 3) / 2)
        assert mean_rank["B"] == pytest.approx((1.5 + 1) / 2)

    def test_optimizer_missing_in_a_block_is_ranked_where_present(self) -> None:
        medians = {"f-2": {"A": 1.0, "B": 2.0}, "f-5": {"A": 1.0}}
        per_optimizer, _ = metrics.average_ranks(medians)
        assert set(per_optimizer["B"]) == {"f-2"}


class TestFriedman:
    def test_detects_a_consistent_ordering(self) -> None:
        medians = {
            f"b{i}": {"A": 1.0 + 0.01 * i, "B": 2.0 + 0.01 * i, "C": 3.0 + 0.01 * i}
            for i in range(6)
        }
        result = metrics.friedman(medians)
        assert result["blocks"] == 6
        assert result["optimizers"] == 3
        assert result["p"] < 0.01
        assert result["chi2"] == pytest.approx(12.0)

    def test_uses_complete_blocks_only(self) -> None:
        medians = {
            "b1": {"A": 1.0, "B": 2.0, "C": 3.0},
            "b2": {"A": 1.0, "B": 2.0, "C": 3.0},
            "b3": {"A": 1.0, "B": 2.0},
        }
        assert metrics.friedman(medians)["blocks"] == 2

    def test_falls_back_to_common_optimizers(self) -> None:
        medians = {
            "b1": {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0},
            "b2": {"A": 1.0, "B": 2.0, "C": 3.0},
            "b3": {"A": 1.0, "B": 2.0, "C": 3.0},
        }
        result = metrics.friedman(medians)
        assert result is not None
        assert result["optimizers"] >= 3

    def test_none_when_too_small(self) -> None:
        assert metrics.friedman({}) is None
        assert metrics.friedman({"b": {"A": 1.0, "B": 2.0, "C": 3.0}}) is None


def test_floor_is_the_last_target() -> None:
    assert metrics.PRECISION_FLOOR == 1e-8
    assert math.isclose(float(metrics.floored(np.array([0.0]))[0]), 1e-8)
