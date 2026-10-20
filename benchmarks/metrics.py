"""Scientific metrics computed from the raw per-cell benchmark results.

All functions work on the unrounded raw data written by
``run_benchmark_suite.py`` (one *cell* = one optimizer on one function and
dimension) and are pure, so they are tested against hand-computed examples.

Definitions follow the COCO/BBOB platform where one exists:

- **ERT** (expected running time) of a target: the evaluations spent by all runs up
  to the target (successful runs: until the target is reached; unsuccessful runs:
  everything they used), divided by the number of successful runs.
- **SP1**: mean evaluations of the successful runs divided by the success rate.
- **ECDF**: the fraction of (run, target) pairs reached within a given budget.

The statistical comparison with the baseline is a one-sided Mann-Whitney U test
(the runs of two optimizers are independent samples, so a rank-sum rather than a
signed-rank test), Holm-adjusted within each function and dimension, with Cliff's
delta as effect size. Final precisions are floored at the last target, so runs that
solved the problem tie instead of being ranked by rounding noise.
"""

from __future__ import annotations

import math
import warnings

from typing import Any

import numpy as np

from scipy import stats


ECDF_POINTS = 40
PRECISION_FLOOR = 1e-8
"""Errors below this count as solved; they tie in tests and ranks."""


def to_array(values: list[float | None]) -> np.ndarray:
    """Convert a JSON list with ``None`` for missing values to a float array."""
    return np.array([math.nan if v is None else v for v in values], dtype=float)


def usable_runs(cell: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the runs that carry a result (failed runs are excluded)."""
    return [
        run
        for run in cell["runs"]
        if run["status"] != "failed" and run["best_fitness"] is not None
    ]


def final_errors(cell: dict[str, Any], f_opt: float = 0.0) -> np.ndarray:
    """Final precision ``f - f_opt`` of every usable run (never negative)."""
    return np.array(
        [max(run["best_fitness"] - f_opt, 0.0) for run in usable_runs(cell)],
        dtype=float,
    )


def summary_stats(errors: np.ndarray) -> dict[str, float] | None:
    """Mean, std, median, quartiles, best and worst of ``errors``."""
    if errors.size == 0:
        return None
    return {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "median": float(np.median(errors)),
        "q1": float(np.quantile(errors, 0.25)),
        "q3": float(np.quantile(errors, 0.75)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
    }


def success_rates(runs: list[dict[str, Any]], n_targets: int) -> list[float | None]:
    """Fraction of runs that reached each target (``None`` without runs)."""
    if not runs:
        return [None] * n_targets
    return [
        sum(run["hits"][i] is not None for run in runs) / len(runs)
        for i in range(n_targets)
    ]


def ert_and_sp1(
    runs: list[dict[str, Any]], target_index: int
) -> tuple[float | None, float | None]:
    """COCO expected running time and SP1 for one target.

    Both are ``None`` when no run reached the target.

    Example:
        >>> runs = [
        ...     {"hits": [100], "n_evaluations": 500},
        ...     {"hits": [300], "n_evaluations": 500},
        ...     {"hits": [None], "n_evaluations": 1000},
        ... ]
        >>> ert_and_sp1(runs, 0)
        (700.0, 300.0)
    """
    successful = [
        r["hits"][target_index] for r in runs if r["hits"][target_index] is not None
    ]
    if not successful:
        return None, None
    unsuccessful = [r["n_evaluations"] for r in runs if r["hits"][target_index] is None]
    n_success = len(successful)
    ert = (sum(successful) + sum(unsuccessful)) / n_success
    sp1 = (sum(successful) / n_success) * len(runs) / n_success
    return float(ert), float(sp1)


def ecdf_budgets(budget: int, dim: int, n_points: int = ECDF_POINTS) -> np.ndarray:
    """Log-spaced budgets in evaluations per dimension, from 1 to ``budget / dim``."""
    return np.geomspace(1, budget / dim, n_points)


def ecdf(
    runs: list[dict[str, Any]], dim: int, budgets: np.ndarray
) -> list[float] | None:
    """Fraction of (run, target) pairs reached within each budget (evals / dim)."""
    pairs = [hit for run in runs for hit in run["hits"]]
    if not pairs:
        return None
    reached = np.array(sorted(h for h in pairs if h is not None), dtype=float)
    return [
        float(np.searchsorted(reached, b * dim, side="right") / len(pairs))
        for b in budgets
    ]


def quantile_curves(
    runs: list[dict[str, Any]], key: str, f_opt: float = 0.0, *, subtract_optimum: bool
) -> dict[str, list[float | None]] | None:
    """Median and quartiles over runs of a per-checkpoint series.

    Args:
        runs: Usable runs.
        key: ``"best_at"`` (precision curves) or ``"dispersion_at"``.
        f_opt: Optimum subtracted from ``best_at`` values.
        subtract_optimum: Whether to subtract ``f_opt`` (and clip at zero).
    """
    if not runs:
        return None
    matrix = np.array([to_array(run[key]) for run in runs])
    if subtract_optimum:
        matrix = np.maximum(matrix - f_opt, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns -> None
        q1, median, q3 = (np.nanquantile(matrix, q, axis=0) for q in (0.25, 0.5, 0.75))
    return {
        "q1": [None if math.isnan(v) else float(v) for v in q1],
        "median": [None if math.isnan(v) else float(v) for v in median],
        "q3": [None if math.isnan(v) else float(v) for v in q3],
    }


def time_stats(runs: list[dict[str, Any]]) -> dict[str, float] | None:
    """Median seconds per run and median microseconds per evaluation."""
    if not runs:
        return None
    seconds = np.array([r["elapsed"] for r in runs], dtype=float)
    per_eval = np.array(
        [r["elapsed"] / r["n_evaluations"] for r in runs if r["n_evaluations"]],
        dtype=float,
    )
    return {
        "median_run_seconds": float(np.median(seconds)),
        "median_us_per_evaluation": float(np.median(per_eval) * 1e6)
        if per_eval.size
        else 0.0,
    }


def floored(errors: np.ndarray) -> np.ndarray:
    """Floor errors at the last target so solved runs tie."""
    return np.maximum(errors, PRECISION_FLOOR)


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1).

    Example:
        >>> [round(p, 2) for p in holm([0.01, 0.04, 0.03])]
        [0.03, 0.06, 0.06]
    """
    count = len(p_values)
    order = sorted(range(count), key=lambda i: p_values[i])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def cliffs_delta(better_candidate: np.ndarray, reference: np.ndarray) -> float:
    """Cliff's delta of lower-is-better samples: P(candidate < reference) - P(candidate > reference).

    +1 means every candidate run beat every reference run, -1 the opposite, 0 no
    difference.
    """
    candidate = np.asarray(better_candidate, dtype=float)[:, None]
    other = np.asarray(reference, dtype=float)[None, :]
    wins = np.sum(candidate < other)
    losses = np.sum(candidate > other)
    return float((wins - losses) / (candidate.size * other.size))


def compare_to_baseline(
    candidate: np.ndarray, baseline: np.ndarray
) -> dict[str, float] | None:
    """One-sided Mann-Whitney U test "candidate has lower precision error than baseline".

    Returns the raw p-value, ``u`` (the candidate's U statistic) and Cliff's delta,
    or ``None`` when a sample is empty.
    """
    if candidate.size == 0 or baseline.size == 0:
        return None
    a, b = floored(candidate), floored(baseline)
    delta = cliffs_delta(a, b)
    if np.ptp(np.concatenate([a, b])) == 0:  # identical constants: no evidence at all
        return {"p": 1.0, "u": float(a.size * b.size / 2), "delta": 0.0}
    result = stats.mannwhitneyu(a, b, alternative="less")
    return {"p": float(result.pvalue), "u": float(result.statistic), "delta": delta}


def average_ranks(
    medians: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Rank the optimizers of every function-dimension block by their median precision.

    Args:
        medians: ``{block: {optimizer: median floored error}}``.

    Returns:
        ``({optimizer: {block: rank}}, {optimizer: mean rank})``. Rank 1 is best;
        ties share the average rank. An optimizer is ranked only among those with a
        result in the block.
    """
    per_optimizer: dict[str, dict[str, float]] = {}
    for block, values in medians.items():
        names = sorted(values)
        ranks = stats.rankdata([values[n] for n in names], method="average")
        for name, rank in zip(names, ranks, strict=True):
            per_optimizer.setdefault(name, {})[block] = float(rank)
    mean_rank = {n: float(np.mean(list(r.values()))) for n, r in per_optimizer.items()}
    return per_optimizer, mean_rank


def friedman(medians: dict[str, dict[str, float]]) -> dict[str, float] | None:
    """Friedman test over the blocks in which every optimizer has a result.

    If fewer than two blocks are complete (some optimizer is missing nearly
    everywhere), the test runs over the optimizers present in all blocks instead.

    Returns:
        ``chi2``, ``p``, and the number of ``blocks`` and ``optimizers`` used, or
        ``None`` when fewer than two blocks or three optimizers remain.
    """
    if not medians:
        return None
    everyone = set().union(*(set(v) for v in medians.values()))
    blocks = [b for b, values in medians.items() if set(values) == everyone]
    names = sorted(everyone)
    if len(blocks) < 2:
        blocks = list(medians)
        names = sorted(set.intersection(*(set(v) for v in medians.values())))
    if len(names) < 3 or len(blocks) < 2:
        return None
    samples = [[medians[block][name] for block in blocks] for name in names]
    result = stats.friedmanchisquare(*samples)
    return {
        "chi2": float(result.statistic),
        "p": float(result.pvalue),
        "blocks": len(blocks),
        "optimizers": len(names),
    }
