"""Data model of the published benchmark data (schema version 2).

These Pydantic models are the **single source of truth**. The aggregator validates
every file it writes against them, ``export_schema.py`` derives
``docs/schemas/benchmark-data-schema.json`` and the TypeScript types of the docs
site (``docs/.vitepress/theme/types/benchmark.ts``) from them, and a test fails if
the committed files drift from the models.

Published layout (``docs/public/benchmarks/``):

- ``index.json``                  :class:`Index`      protocol, optimizers, available cells
- ``cells/<function>-<dim>.json`` :class:`Cell`       every optimizer on one function and dimension
- ``ranks.json``                  :class:`Ranks`      average ranks over all function-dimension blocks
- ``trajectory-<function>-2.json`` :class:`Trajectory` best-so-far paths of 2D runs

Missing values are ``null``; nothing is ``NaN`` or ``Infinity`` so the files are
plain JSON that every browser can parse.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


SCHEMA_VERSION = 2


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FunctionInfo(_Model):
    """A benchmark function with its search domain and known optimum."""

    name: str
    lower: float
    upper: float
    f_opt: float


class Versions(_Model):
    """Software versions the data was produced with."""

    python: str
    numpy: str
    scipy: str


class Protocol(_Model):
    """How the data was produced; everything a reader needs to interpret it."""

    budget_per_dim: int = Field(
        ge=1, description="Evaluations per dimension for every run"
    )
    targets: list[float] = Field(description="Target precisions f - f_opt, descending")
    n_runs: int = Field(ge=1)
    dimensions: list[int]
    functions: list[FunctionInfo]
    baseline: str = Field(description="Optimizer every other one is tested against")
    ecdf_points: int
    precision_floor: float = Field(
        description="Errors below this tie in tests and ranks"
    )
    time_cap_seconds: float | None
    cell_time_seconds: float | None
    test: str = Field(description="Statistical procedure, in words")
    excluded: list[str] = Field(description="Library optimizers not part of the suite")


class OptimizerInfo(_Model):
    """One benchmarked optimizer."""

    name: str
    category: str
    baseline: bool = False
    page: str | None = Field(
        default=None, description="Docs page path, if the optimizer has one"
    )


class CellRef(_Model):
    """Pointer to a cell file."""

    function: str
    dim: int
    file: str
    optimizers: int


class Index(_Model):
    """Entry point of the benchmark data: ``index.json``."""

    schema_version: Literal[2] = SCHEMA_VERSION
    generated: str
    git_sha: str | None
    versions: Versions
    protocol: Protocol
    optimizers: list[OptimizerInfo]
    cells: list[CellRef]


class Stats(_Model):
    """Distribution of the final precision over runs."""

    mean: float
    std: float
    median: float
    q1: float
    q3: float
    min: float
    max: float


class Curve(_Model):
    """Median and quartiles over runs at every checkpoint."""

    q1: list[float | None]
    median: list[float | None]
    q3: list[float | None]


class TimeStats(_Model):
    """Wall-clock cost."""

    median_run_seconds: float
    median_us_per_evaluation: float


class BaselineTest(_Model):
    """One-sided Mann-Whitney U test "better than the baseline" and its effect size."""

    p: float = Field(description="Raw one-sided p-value")
    p_holm: float = Field(description="Holm-adjusted within the function and dimension")
    u: float
    delta: float = Field(
        description="Cliff's delta, +1 = every run beat every baseline run"
    )


class OptimizerCell(_Model):
    """All metrics of one optimizer on one function and dimension."""

    runs: int = Field(description="Runs with a result")
    runs_planned: int
    failed: int
    truncated: int = Field(description="Runs cut off by the wall-clock cap")
    exhausted: int = Field(description="Runs that used the whole evaluation budget")
    max_iter: int
    finals: list[float] = Field(description="Final precision of every usable run")
    stats: Stats | None
    success: list[float | None] = Field(description="Success rate per target")
    ert: list[float | None] = Field(description="COCO expected running time per target")
    sp1: list[float | None] = Field(description="COCO SP1 per target")
    ecdf: list[float] | None = Field(
        description="Fraction of (run, target) pairs reached"
    )
    convergence: Curve | None = Field(description="Precision at the checkpoints")
    dispersion: list[float | None] | None = Field(
        description="Median normalised spread of recently evaluated points at the checkpoints"
    )
    time: TimeStats | None
    test: BaselineTest | None = Field(description="None for the baseline itself")


class Cell(_Model):
    """One function and dimension: ``cells/<function>-<dim>.json``."""

    function: str
    dim: int
    budget: int
    checkpoints: list[int] = Field(description="Evaluation counts of the curve points")
    targets: list[float]
    ecdf_budgets: list[float] = Field(
        description="Evaluations per dimension of the ECDF points"
    )
    optimizers: dict[str, OptimizerCell]


class RankEntry(_Model):
    """Rank of one optimizer."""

    name: str
    category: str
    mean_rank: float
    ranks: dict[str, float] = Field(
        description="Rank per block, key '<function>-<dim>'"
    )


class Friedman(_Model):
    """Friedman test over the blocks."""

    chi2: float
    p: float
    blocks: int
    optimizers: int


class Ranks(_Model):
    """Average ranks: ``ranks.json``."""

    schema_version: Literal[2] = SCHEMA_VERSION
    blocks: list[str]
    entries: list[RankEntry]
    friedman: Friedman | None


class Trajectory(_Model):
    """Best-so-far path of the first run of every optimizer on a 2D function."""

    function: str
    seed: int
    optimizers: dict[str, list[list[float]]] = Field(
        description="Per optimizer rows [evaluations, x, y, f]"
    )


class BenchmarkBundle(_Model):
    """Wrapper that exists only to export every model in one JSON schema."""

    index: Index
    cell: Cell
    ranks: Ranks
    trajectory: Trajectory
