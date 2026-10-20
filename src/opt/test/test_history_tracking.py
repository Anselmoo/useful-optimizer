"""Every single-objective optimizer must be able to record a convergence history.

The benchmark runner and the docs charts rely on ``history["best_fitness"]``.
Optimizers are discovered from ``opt.__all__`` so a new one cannot ship without
history support.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import opt

from opt.abstract import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley


def _single_objective_optimizers() -> list[type[AbstractOptimizer]]:
    classes = []
    for name in opt.__all__:
        cls = getattr(opt, name)
        if (
            inspect.isclass(cls)
            and issubclass(cls, AbstractOptimizer)
            and not inspect.isabstract(cls)
        ):
            classes.append(cls)
    return classes


OPTIMIZERS = _single_objective_optimizers()


def test_discovery_finds_the_library() -> None:
    assert len(OPTIMIZERS) > 100


@pytest.mark.parametrize("cls", OPTIMIZERS, ids=lambda c: c.__name__)
def test_records_finite_history(cls: type[AbstractOptimizer]) -> None:
    params = inspect.signature(cls.__init__).parameters
    has_varkw = any(p.kind is p.VAR_KEYWORD for p in params.values())
    candidate = {
        "func": shifted_ackley,
        "lower_bound": -5.0,
        "upper_bound": 5.0,
        "dim": 2,
        "max_iter": 4,
        "population_size": 30,
        "n_bats": 30,
        "seed": 1,
    }
    kwargs = {k: v for k, v in candidate.items() if has_varkw or k in params}

    optimizer = cls(**kwargs)
    if not optimizer.track_history:
        optimizer.enable_history_tracking()
    optimizer.search()

    history = optimizer.history["best_fitness"]
    assert len(history) > 0
    assert np.all(np.isfinite(history))
