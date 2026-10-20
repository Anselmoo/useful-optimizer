r"""Benchmark suite runner for the optimization algorithms.

Every optimizer runs on standard benchmark functions under the same evaluation
budget (see :mod:`benchmarks.protocol`). The unit of work is a **cell**: one
optimizer on one function in one dimension, with ``--n-runs`` independent runs.
Each finished cell is written to its own file,

    <output-dir>/raw/<function>/<dim>/<optimizer>.json

so a timeout or crash loses nothing (``--resume`` skips the finished cells) and the
work can be split across CI jobs (``--category``, ``--functions``, ``--shard i/n``)
and merged afterwards by ``aggregate_results.py``.

Tiers:

- ``showcase``: 4 representative optimizers, for quick checks.
- ``standard``: 13 optimizers across all major families.
- ``comprehensive``: every single-objective optimizer of the library.

The baseline (``RandomSearch``) is part of every tier, because the statistical tests
compare against it.

Examples:
    # Quick local check
    python benchmarks/run_benchmark_suite.py --tier showcase --functions sphere \
        --dimensions 2 --n-runs 5

    # One CI shard of the full suite
    python benchmarks/run_benchmark_suite.py --tier comprehensive \
        --category swarm_intelligence --functions rastrigin --workers 4 --resume

    # The matrix for CI
    python benchmarks/run_benchmark_suite.py --tier comprehensive --list-shards
"""

# ruff: noqa: E402
# (the environment and sys.path setup below must run before the imports that need it)
from __future__ import annotations

import os


# One BLAS thread per process: the workers already use every core, and oversubscribed
# BLAS threads make small matrix operations slower. Must happen before numpy loads.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import platform
import subprocess
import sys
import time

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy


# Run as a script (`python benchmarks/run_benchmark_suite.py`) the repo root is not
# on sys.path, but the shared protocol lives in the `benchmarks` package.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.protocol import BASELINE
from benchmarks.protocol import BUDGET_PER_DIM
from benchmarks.protocol import DEFAULT_CELL_TIME
from benchmarks.protocol import DEFAULT_TIME_CAP
from benchmarks.protocol import FUNCTIONS
from benchmarks.protocol import TARGETS
from benchmarks.protocol import category_of
from benchmarks.protocol import discover_optimizers
from benchmarks.protocol import run_cell


DIMENSIONS = [2, 5, 10, 20]  # BBOB standard dimensions
N_RUNS = 15  # BBOB standard: 15 independent runs
RAW_SCHEMA = "raw-cell-1"

TIER_OPTIMIZERS: dict[str, list[str] | None] = {
    "showcase": ["ParticleSwarm", "DifferentialEvolution", "AdamW", "HarmonySearch"],
    "standard": [
        "ParticleSwarm",
        "AntColony",
        "FireflyAlgorithm",
        "BatAlgorithm",
        "GreyWolfOptimizer",
        "GeneticAlgorithm",
        "DifferentialEvolution",
        "HarmonySearch",
        "SimulatedAnnealing",
        "HillClimbing",
        "NelderMead",
        "AdamW",
        "SGDMomentum",
    ],
    "comprehensive": None,  # everything discover_optimizers() finds
}


@dataclass(frozen=True)
class Cell:
    """One optimizer on one function and dimension."""

    function: str
    dim: int
    optimizer: str
    category: str

    def path(self, output_dir: Path) -> Path:
        """Location of the raw result file of this cell."""
        return (
            output_dir
            / "raw"
            / self.function
            / str(self.dim)
            / f"{self.optimizer}.json"
        )


@dataclass(frozen=True)
class Settings:
    """Everything that defines how a cell is run (and when a stored one is stale)."""

    n_runs: int = N_RUNS
    budget_per_dim: int = BUDGET_PER_DIM
    time_cap: float = DEFAULT_TIME_CAP
    cell_time: float = DEFAULT_CELL_TIME
    population_size: int = 30


def plan_cells(
    tier: str,
    *,
    categories: set[str] | None = None,
    optimizer_names: set[str] | None = None,
    functions: list[str] | None = None,
    dimensions: list[int] | None = None,
) -> list[Cell]:
    """List the cells of a tier, ordered by function, dimension and optimizer."""
    if tier not in TIER_OPTIMIZERS:
        msg = f"Unknown tier '{tier}'. Choose one of: {', '.join(TIER_OPTIMIZERS)}"
        raise ValueError(msg)

    available = discover_optimizers()
    wanted = TIER_OPTIMIZERS[tier]
    names = sorted(available) if wanted is None else sorted({*wanted, BASELINE})
    missing = [n for n in names if n not in available]
    if missing:
        msg = f"Tier '{tier}' lists unknown optimizer(s): {', '.join(missing)}"
        raise ValueError(msg)
    if optimizer_names is not None:
        unknown = optimizer_names - set(available)
        if unknown:
            msg = f"Unknown optimizer(s): {', '.join(sorted(unknown))}"
            raise ValueError(msg)
        names = [n for n in names if n in optimizer_names]
    if categories is not None:
        names = [n for n in names if category_of(available[n]) in categories]

    function_names = functions if functions is not None else list(FUNCTIONS)
    unknown_functions = set(function_names) - set(FUNCTIONS)
    if unknown_functions:
        msg = (
            f"Unknown function(s): {', '.join(sorted(unknown_functions))}. "
            f"Available: {', '.join(FUNCTIONS)}"
        )
        raise ValueError(msg)

    return [
        Cell(function, dim, name, category_of(available[name]))
        for function in function_names
        for dim in (dimensions if dimensions is not None else DIMENSIONS)
        for name in names
    ]


def list_shards(cells: list[Cell]) -> list[dict[str, str]]:
    """Return the ``category x function`` combinations present in ``cells``.

    This is the job matrix for CI: one job per combination keeps every job well
    under the six hour limit of a GitHub-hosted runner.
    """
    combos = sorted({(c.category, c.function) for c in cells})
    return [{"category": cat, "function": fn} for cat, fn in combos]


def shard_cells(cells: list[Cell], index: int, count: int) -> list[Cell]:
    """Return slice ``index`` of ``count``: every ``count``-th cell, round robin.

    Round robin spreads the expensive optimizers (which sit next to each other in
    the ordered list) over the slices; the slices are disjoint and cover all cells.
    """
    return [cell for position, cell in enumerate(cells) if position % count == index]


def is_done(cell: Cell, output_dir: Path, settings: Settings) -> bool:
    """Whether a valid result with the same settings is already stored."""
    path = cell.path(output_dir)
    if not path.exists():
        return False
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    protocol = stored.get("protocol", {})
    return (
        stored.get("schema") == RAW_SCHEMA
        and protocol.get("budget_per_dim") == settings.budget_per_dim
        and protocol.get("n_runs") == settings.n_runs
        and protocol.get("targets") == list(TARGETS)
    )


def run_and_store(cell: Cell, output_dir: str, settings: Settings) -> dict[str, Any]:
    """Run one cell and write its raw file atomically. Executed in worker processes."""
    started = time.perf_counter()
    cls = discover_optimizers()[cell.optimizer]
    result = run_cell(
        cls,
        FUNCTIONS[cell.function],
        cell.dim,
        n_runs=settings.n_runs,
        function_index=list(FUNCTIONS).index(cell.function),
        budget_per_dim=settings.budget_per_dim,
        population_size=settings.population_size,
        time_cap=settings.time_cap,
        cell_time=settings.cell_time,
    )
    path = cell.path(Path(output_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)

    runs = result["runs"]
    finals = [r["best_fitness"] for r in runs if r["best_fitness"] is not None]
    return {
        "cell": asdict(cell),
        "runs": len(runs),
        "failed": sum(r["status"] == "failed" for r in runs),
        "truncated": sum(r["status"] == "truncated_time" for r in runs),
        "median_best": float(np.median(finals)) if finals else None,
        "seconds": time.perf_counter() - started,
    }


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def write_run_meta(
    output_dir: Path, tier: str, settings: Settings, started: str, cells: int
) -> Path:
    """Write ``run-meta.json`` describing this invocation (merged later by CI)."""
    meta = {
        "tier": tier,
        "settings": asdict(settings),
        "targets": list(TARGETS),
        "functions": list(FUNCTIONS),
        "cells": cells,
        "started": started,
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }
    path = output_dir / "run-meta.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def _progress(done: int, total: int, summary: dict[str, Any]) -> None:
    cell = summary["cell"]
    flags = ""
    if summary["failed"]:
        flags += f" failed={summary['failed']}"
    if summary["truncated"]:
        flags += f" truncated={summary['truncated']}"
    best = summary["median_best"]
    best_text = "n/a" if best is None else f"{best:.3e}"
    print(
        f"[{done}/{total}] {cell['function']} {cell['dim']}D {cell['optimizer']}: "
        f"{summary['runs']} runs, median best {best_text}, "
        f"{summary['seconds']:.1f}s{flags}",
        flush=True,
    )


def run_cells(
    cells: list[Cell], output_dir: Path, settings: Settings, workers: int
) -> list[dict[str, Any]]:
    """Run ``cells``, in this process or in a pool, and return their summaries."""
    summaries: list[dict[str, Any]] = []
    total = len(cells)
    if workers <= 1:
        for index, cell in enumerate(cells, start=1):
            summary = run_and_store(cell, str(output_dir), settings)
            summaries.append(summary)
            _progress(index, total, summary)
        return summaries

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_and_store, c, str(output_dir), settings) for c in cells
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            summary = future.result()
            summaries.append(summary)
            _progress(index, total, summary)
    return summaries


def parse_shard(text: str) -> tuple[int, int]:
    """Parse ``"i/n"`` (0-based ``i``) into ``(i, n)``."""
    try:
        index_text, count_text = text.split("/")
        index, count = int(index_text), int(count_text)
    except ValueError as exc:
        msg = f"--shard must look like 0/4, got '{text}'"
        raise argparse.ArgumentTypeError(msg) from exc
    if count < 1 or not 0 <= index < count:
        msg = f"--shard {text}: need 0 <= i < n"
        raise argparse.ArgumentTypeError(msg)
    return index, count


def _csv(text: str | None) -> list[str] | None:
    return [part.strip() for part in text.split(",") if part.strip()] if text else None


def build_parser() -> argparse.ArgumentParser:
    """Create the command line parser."""
    parser = argparse.ArgumentParser(
        description="Run the optimizer benchmark suite under an evaluation budget.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output-dir", default="benchmarks/output")
    parser.add_argument("--tier", choices=list(TIER_OPTIMIZERS), default="standard")
    parser.add_argument("--category", help="comma-separated optimizer categories")
    parser.add_argument("--optimizers", help="comma-separated optimizer class names")
    parser.add_argument(
        "--functions", help=f"comma-separated, from: {', '.join(FUNCTIONS)}"
    )
    parser.add_argument("--dimensions", help="comma-separated, default 2,5,10,20")
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--budget-per-dim", type=int, default=BUDGET_PER_DIM)
    parser.add_argument(
        "--time-cap",
        type=float,
        default=DEFAULT_TIME_CAP,
        help="seconds after which one run is cut off",
    )
    parser.add_argument(
        "--cell-time",
        type=float,
        default=DEFAULT_CELL_TIME,
        help="seconds after which a cell stops starting new runs",
    )
    parser.add_argument(
        "--shard", type=parse_shard, help="run only slice i of n (0-based)"
    )
    parser.add_argument("--workers", type=int, default=1, help="parallel processes")
    parser.add_argument("--resume", action="store_true", help="skip finished cells")
    parser.add_argument(
        "--list-shards",
        action="store_true",
        help="print the CI job matrix (category x function) as JSON and exit",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="count the cells and exit"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    args = build_parser().parse_args(argv)
    try:
        cells = plan_cells(
            args.tier,
            categories=set(_csv(args.category) or []) or None,
            optimizer_names=set(_csv(args.optimizers) or []) or None,
            functions=_csv(args.functions),
            dimensions=[int(d) for d in _csv(args.dimensions) or []] or None,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.list_shards:
        print(json.dumps(list_shards(cells)))
        return 0

    if args.shard is not None:
        index, count = args.shard
        cells = [
            cell for position, cell in enumerate(cells) if position % count == index
        ]

    settings = Settings(
        n_runs=args.n_runs,
        budget_per_dim=args.budget_per_dim,
        time_cap=args.time_cap,
        cell_time=args.cell_time,
    )
    output_dir = Path(args.output_dir)
    pending = [
        c for c in cells if not (args.resume and is_done(c, output_dir, settings))
    ]
    print(
        f"{args.tier}: {len(cells)} cells planned, {len(cells) - len(pending)} already "
        f"done, {len(pending)} to run, budget {settings.budget_per_dim} x dim, "
        f"{settings.n_runs} runs each, {args.workers} worker(s)",
        flush=True,
    )
    if args.dry_run:
        return 0

    started = datetime.now(UTC).isoformat(timespec="seconds")
    run_cells(pending, output_dir, settings, args.workers)
    meta = write_run_meta(output_dir, args.tier, settings, started, len(cells))
    print(f"Done. Raw results in {output_dir / 'raw'}, run metadata in {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
