# ruff: noqa: S101
"""Tests for benchmark Pydantic models."""

from __future__ import annotations

from benchmarks.models import Run


def test_run_model_accepts_convergence_history() -> None:
    r = Run(
        best_fitness=0.123,
        best_solution=[0.0, 0.0],
        n_evaluations=100,
        history=None,
        convergence_history=[0.5, 0.3, 0.2, 0.123],
    )

    # Basic validation: fields are set and typed
    assert r.convergence_history is not None
    assert len(r.convergence_history) == 4
    assert isinstance(r.convergence_history[0], (float, int))
