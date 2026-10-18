"""Budgeted evaluation protocol for the benchmark suite.

Optimizers differ enormously in how many function evaluations one iteration costs
(from 0.1 to about 3000), so a fixed ``max_iter`` is not a common budget. This
module follows the COCO/BBOB convention instead: **every run gets the same
evaluation budget** (``BUDGET_PER_DIM * dim``) and everything is measured against
evaluations, not iterations.

One wrapper, :class:`BudgetedObjective`, is placed around the objective function.
It works for every optimizer without touching it:

- it counts every call (the optimizer's own ``n_evaluations`` can be wrong),
- it cuts the run off at the budget by raising :exc:`BudgetExhausted`, a
  ``BaseException`` so an optimizer's ``except Exception`` cannot swallow it,
- it records the true best-so-far, the first evaluation that reaches each target
  precision (exact hitting times, also for the optimizers without
  ``evaluations_to_target``), best-so-far at log-spaced checkpoints, a dispersion
  proxy and, in 2D, the best-so-far trajectory.
"""

from __future__ import annotations

import contextlib
import inspect
import math
import time

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

import opt

from opt.abstract import AbstractOptimizer
from opt.benchmark.functions import ackley
from opt.benchmark.functions import griewank
from opt.benchmark.functions import rastrigin
from opt.benchmark.functions import rosenbrock
from opt.benchmark.functions import shifted_ackley
from opt.benchmark.functions import sphere


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from numpy import ndarray


BUDGET_PER_DIM = 2000
"""Function evaluations per dimension available to every run."""

TARGETS: tuple[float, ...] = (1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
"""Target precisions ``f - f_opt``, one per decade (descending)."""

N_CHECKPOINTS = 60
DISPERSION_WINDOW = 30
DEFAULT_TIME_CAP = 20.0
"""Wall-clock seconds after which a run is cut off and flagged ``truncated_time``."""

DEFAULT_CELL_TIME = 150.0
"""Wall-clock seconds after which a cell stops starting new runs (see ``run_cell``)."""
MIN_RUNS = 5
"""A cell always completes at least this many runs, whatever the time limit."""

MAX_ITER_CEILING = 100_000
BASELINE = "RandomSearch"
"""Class name of the baseline every other optimizer is tested against."""


@dataclass(frozen=True)
class FunctionConfig:
    """A benchmark function with its search domain and known optimum."""

    name: str
    func: Callable[[ndarray], float]
    lower: float
    upper: float
    f_opt: float = 0.0


FUNCTIONS: dict[str, FunctionConfig] = {
    cfg.name: cfg
    for cfg in (
        FunctionConfig("sphere", sphere, -5.12, 5.12),
        FunctionConfig("rosenbrock", rosenbrock, -5.0, 10.0),
        FunctionConfig("rastrigin", rastrigin, -5.12, 5.12),
        FunctionConfig("ackley", ackley, -32.768, 32.768),
        FunctionConfig("shifted_ackley", shifted_ackley, -32.768, 32.768),
        FunctionConfig("griewank", griewank, -600.0, 600.0),
    )
}


class BudgetExhausted(BaseException):
    """Raised inside the objective when the budget or the time cap is used up.

    Deriving from ``BaseException`` keeps optimizers that wrap their loop in
    ``except Exception`` from swallowing it.
    """

    def __init__(self, reason: str) -> None:
        """Store why the run was cut off (``"budget"`` or ``"time"``)."""
        super().__init__(reason)
        self.reason = reason


_BUDGET = "budget"
_TIME = "time"


def budget_for(dim: int, per_dim: int = BUDGET_PER_DIM) -> int:
    """Return the evaluation budget of a run in ``dim`` dimensions."""
    return per_dim * dim


def checkpoint_grid(budget: int, n_points: int = N_CHECKPOINTS) -> list[int]:
    """Return strictly increasing, log-spaced evaluation counts ending at ``budget``."""
    grid = np.unique(np.round(np.geomspace(1, budget, n_points)).astype(int))
    return [int(v) for v in grid]


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


class BudgetedObjective:
    """Callable wrapper that enforces the budget and records the run.

    Example:
        >>> obj = BudgetedObjective(lambda x: float(np.sum(x**2)), -5, 5, 2, budget=3)
        >>> _ = [obj(np.array([1.0, 1.0])) for _ in range(3)]
        >>> obj.n_evaluations
        3
        >>> try:
        ...     obj(np.array([0.0, 0.0]))
        ... except BudgetExhausted as stop:
        ...     stop.reason
        'budget'
    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower: float,
        upper: float,
        dim: int,
        budget: float,
        *,
        targets: Sequence[float] = TARGETS,
        checkpoints: Sequence[int] | None = None,
        f_opt: float = 0.0,
        time_cap: float | None = None,
        track_trajectory: bool = False,
    ) -> None:
        """Create the wrapper.

        Args:
            func: The objective function.
            lower: Lower bound of the search space (used to normalise dispersion).
            upper: Upper bound of the search space.
            dim: Problem dimension.
            budget: Maximum number of evaluations (``math.inf`` for none).
            targets: Target precisions ``f - f_opt`` in descending order.
            checkpoints: Evaluation counts at which best-so-far and dispersion are
                stored; ``None`` disables checkpointing.
            f_opt: Known optimum, subtracted before comparing with the targets.
            time_cap: Wall-clock limit in seconds, ``None`` for no limit.
            track_trajectory: Also record the best-so-far point (2D runs).
        """
        self._func = func
        self._range = float(upper - lower) or 1.0
        self.dim = dim
        self.budget = budget
        self.targets = tuple(targets)
        self.checkpoints = list(checkpoints) if checkpoints is not None else []
        self.f_opt = f_opt
        self._time_cap = time_cap
        self._track_trajectory = track_trajectory

        self.n_evaluations = 0
        self.best_f = math.inf
        self.best_x: ndarray | None = None
        self.stop_reason: str | None = None
        self.hits: list[int | None] = [None] * len(self.targets)
        self._first_unhit = 0
        self._window = np.zeros((DISPERSION_WINDOW, dim))
        self._best_at: list[float] = []
        self._dispersion_at: list[float] = []
        self._trajectory: list[tuple[int, float, float, float]] = []
        self._started = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Seconds since the wrapper was created."""
        return time.perf_counter() - self._started

    def __call__(self, x: ndarray) -> float:
        """Evaluate ``x``, or stop the run when the budget or time cap is reached."""
        if self.n_evaluations >= self.budget:
            self.stop_reason = _BUDGET
            raise BudgetExhausted(_BUDGET)
        if self._time_cap is not None and self.elapsed > self._time_cap:
            self.stop_reason = _TIME
            raise BudgetExhausted(_TIME)

        point = np.asarray(x, dtype=float).ravel()
        raw = self._func(x)  # returned unchanged: some optimizers need the numpy type
        value = float(raw)
        self.n_evaluations += 1
        self._window[(self.n_evaluations - 1) % DISPERSION_WINDOW, : point.size] = (
            point[: self.dim]
        )

        if value < self.best_f:
            self.best_f = value
            self.best_x = point.copy()
            error = value - self.f_opt
            while self._first_unhit < len(self.targets) and (
                error <= self.targets[self._first_unhit]
            ):
                self.hits[self._first_unhit] = self.n_evaluations
                self._first_unhit += 1
            if self._track_trajectory and point.size >= 2:
                self._trajectory.append(
                    (self.n_evaluations, float(point[0]), float(point[1]), value)
                )

        while len(self._best_at) < len(self.checkpoints) and (
            self.n_evaluations >= self.checkpoints[len(self._best_at)]
        ):
            self._best_at.append(self.best_f)
            self._dispersion_at.append(self._dispersion())
        return raw

    def _dispersion(self) -> float:
        """Std of the most recently evaluated points, normalised by the range."""
        filled = min(self.n_evaluations, DISPERSION_WINDOW)
        if filled < 2:
            return math.nan
        return float(np.mean(np.std(self._window[:filled], axis=0)) / self._range)

    def result(self, max_trajectory: int = 100) -> dict[str, Any]:
        """Return the JSON-serialisable record of the run (no timing, no status)."""
        n_cp = len(self.checkpoints)
        best_at = self._best_at + [self.best_f] * (n_cp - len(self._best_at))
        dispersion_at = self._dispersion_at + [math.nan] * (
            n_cp - len(self._dispersion_at)
        )

        trajectory = self._trajectory
        if len(trajectory) > max_trajectory:
            keep = np.unique(
                np.linspace(0, len(trajectory) - 1, max_trajectory).astype(int)
            )
            trajectory = [trajectory[i] for i in keep]

        return {
            "best_fitness": _finite_or_none(self.best_f),
            "best_solution": (
                [float(v) for v in self.best_x] if self.best_x is not None else None
            ),
            "n_evaluations": self.n_evaluations,
            "hits": list(self.hits),
            "best_at": [_finite_or_none(v) for v in best_at],
            "dispersion_at": [_finite_or_none(v) for v in dispersion_at],
            "trajectory": [[n, x0, x1, f] for n, x0, x1, f in trajectory],
        }


def discover_optimizers() -> dict[str, type[AbstractOptimizer]]:
    """Return every single-objective optimizer exported by ``opt``, by class name.

    The multi-objective optimizers (MOEAD, NSGAII, SPEA2) derive from another base
    class and are not part of this suite: they do not minimise one scalar.
    """
    found: dict[str, type[AbstractOptimizer]] = {}
    for name in opt.__all__:
        cls = getattr(opt, name)
        if (
            inspect.isclass(cls)
            and issubclass(cls, AbstractOptimizer)
            and not inspect.isabstract(cls)
        ):
            found[name] = cls
    return dict(sorted(found.items()))


def category_of(cls: type) -> str:
    """Return the category package of an optimizer class, e.g. ``"classical"``."""
    return cls.__module__.split(".")[1]


def construct_optimizer(
    cls: type[AbstractOptimizer],
    func: Callable[[ndarray], float],
    lower: float,
    upper: float,
    dim: int,
    *,
    max_iter: int,
    seed: int,
    population_size: int = 30,
) -> AbstractOptimizer:
    """Instantiate ``cls`` passing only the keyword arguments its ``__init__`` takes.

    Optimizer subclasses declare very different constructors (some take ``seed`` or
    ``population_size``, one calls its population ``n_bats``), so unsupported keys
    are dropped instead of failing the run. ``seed`` is also set on the instance
    afterwards for the optimizers that read ``self.seed`` lazily but do not take it.
    """
    signature = inspect.signature(cls.__init__)
    accepted = set(signature.parameters)
    has_varkw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    candidates: dict[str, Any] = {
        "func": func,
        "lower_bound": lower,
        "upper_bound": upper,
        "dim": dim,
        "max_iter": max_iter,
        "seed": seed,
        "population_size": population_size,
        "n_bats": population_size,
    }
    kwargs = {k: v for k, v in candidates.items() if has_varkw or k in accepted}
    instance = cls(**kwargs)
    if "seed" not in accepted and hasattr(instance, "seed"):
        instance.seed = seed
    return instance


def run_once(
    cls: type[AbstractOptimizer],
    config: FunctionConfig,
    dim: int,
    *,
    seed: int,
    max_iter: int,
    budget: int | None = None,
    population_size: int = 30,
    time_cap: float | None = DEFAULT_TIME_CAP,
    track_trajectory: bool = False,
) -> dict[str, Any]:
    """Run one optimizer once under the budget and return its record.

    The record has ``status`` ``"ok"`` (finished or budget used), ``"truncated_time"``
    (cut off by the wall-clock cap) or ``"failed"`` (an exception; the partial result
    up to that point is kept), ``exhausted`` (budget fully used) and the fields of
    :meth:`BudgetedObjective.result`.
    """
    budget = budget if budget is not None else budget_for(dim)
    objective = BudgetedObjective(
        config.func,
        config.lower,
        config.upper,
        dim,
        budget,
        checkpoints=checkpoint_grid(budget),
        f_opt=config.f_opt,
        time_cap=time_cap,
        track_trajectory=track_trajectory,
    )
    np.random.seed(seed)
    status = "ok"
    error: str | None = None
    try:
        optimizer = construct_optimizer(
            cls,
            objective,
            config.lower,
            config.upper,
            dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        optimizer.search()
    except BudgetExhausted as stop:
        if stop.reason == "time":
            status = "truncated_time"
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"

    record = objective.result()
    record.update(
        {
            "seed": seed,
            "status": status,
            "error": error,
            "exhausted": objective.n_evaluations >= budget,
            "elapsed": objective.elapsed,
        }
    )
    return record


def _evaluations_after(
    cls: type[AbstractOptimizer],
    config: FunctionConfig,
    dim: int,
    iterations: int,
    *,
    seed: int,
    cap: int,
    population_size: int,
) -> int:
    """Evaluations a run of ``iterations`` iterations needs (capped at ``cap``)."""
    objective = BudgetedObjective(
        config.func,
        config.lower,
        config.upper,
        dim,
        cap,
        f_opt=config.f_opt,
        time_cap=DEFAULT_TIME_CAP / 2,
    )
    np.random.seed(seed)
    with contextlib.suppress(BudgetExhausted):
        construct_optimizer(
            cls,
            objective,
            config.lower,
            config.upper,
            dim,
            max_iter=iterations,
            seed=seed,
            population_size=population_size,
        ).search()
    return objective.n_evaluations


def probe_max_iter(
    cls: type[AbstractOptimizer],
    config: FunctionConfig,
    dim: int,
    *,
    seed: int = 0,
    budget: int | None = None,
    population_size: int = 30,
) -> int:
    """Choose ``max_iter`` so the run just uses up the evaluation budget.

    Two short runs (3 and 6 iterations) give the evaluations per iteration and the
    fixed initialisation cost; ``max_iter`` is then ``(budget - init) / per_iter``.
    Setting it from the budget keeps schedules tied to ``max_iter`` (cooling,
    inertia, step size) coherent instead of cutting them off half way.
    """
    budget = budget if budget is not None else budget_for(dim)
    cap = 5 * budget
    try:
        few = _evaluations_after(
            cls, config, dim, 3, seed=seed, cap=cap, population_size=population_size
        )
        more = _evaluations_after(
            cls, config, dim, 6, seed=seed, cap=cap, population_size=population_size
        )
    except Exception:
        return 1000
    per_iter = (more - few) / 3
    if per_iter <= 0:
        per_iter = max(more / 6, 1.0)
    init = max(few - 3 * per_iter, 0.0)
    iterations = math.ceil((budget - init) / per_iter)
    return int(min(max(iterations, 1), MAX_ITER_CEILING))


def run_cell(
    cls: type[AbstractOptimizer],
    config: FunctionConfig,
    dim: int,
    *,
    n_runs: int,
    function_index: int,
    budget_per_dim: int = BUDGET_PER_DIM,
    population_size: int = 30,
    time_cap: float | None = DEFAULT_TIME_CAP,
    cell_time: float | None = DEFAULT_CELL_TIME,
    trajectory_runs: int = 3,
) -> dict[str, Any]:
    """Run all ``n_runs`` of one optimizer on one function and dimension.

    Seeds are ``run + 100 * function_index``: identical for every optimizer of the
    cell (same starting conditions for the random streams) and different between
    functions. ``max_iter`` is chosen once per cell by :func:`probe_max_iter` and
    corrected once if the first run clearly stopped far below the budget. A cell
    stops starting new runs once ``cell_time`` seconds are used (after at least
    ``MIN_RUNS``); ``protocol.runs_completed`` then records how many were done.
    """
    budget = budget_for(dim, budget_per_dim)
    base_seed = 100 * function_index
    max_iter = probe_max_iter(
        cls, config, dim, seed=base_seed, budget=budget, population_size=population_size
    )

    runs: list[dict[str, Any]] = []
    cell_started = time.perf_counter()
    for run in range(n_runs):
        if (
            cell_time is not None
            and run >= min(MIN_RUNS, n_runs)
            and time.perf_counter() - cell_started > cell_time
        ):
            break  # a very slow optimizer: report the runs done, flagged below
        record = run_once(
            cls,
            config,
            dim,
            seed=base_seed + run,
            max_iter=max_iter,
            budget=budget,
            population_size=population_size,
            time_cap=time_cap,
            track_trajectory=dim == 2 and run < trajectory_runs,
        )
        underused = (
            run == 0
            and record["status"] == "ok"
            and not record["exhausted"]
            and record["n_evaluations"] < budget / 2
            and (record["hits"][-1] is None)
        )
        if underused:
            scale = budget / max(record["n_evaluations"], 1) * 1.1
            max_iter = int(min(max(max_iter * scale, 1), MAX_ITER_CEILING))
            record = run_once(
                cls,
                config,
                dim,
                seed=base_seed + run,
                max_iter=max_iter,
                budget=budget,
                population_size=population_size,
                time_cap=time_cap,
                track_trajectory=dim == 2 and run < trajectory_runs,
            )
        runs.append(record)

    return {
        "schema": "raw-cell-1",
        "function": config.name,
        "dim": dim,
        "optimizer": cls.__name__,
        "category": category_of(cls),
        "protocol": {
            "budget_per_dim": budget_per_dim,
            "budget": budget,
            "targets": list(TARGETS),
            "checkpoints": checkpoint_grid(budget),
            "n_runs": n_runs,
            "max_iter": max_iter,
            "population_size": population_size,
            "time_cap": time_cap,
            "cell_time": cell_time,
            "base_seed": base_seed,
            "runs_completed": len(runs),
        },
        "runs": runs,
    }
