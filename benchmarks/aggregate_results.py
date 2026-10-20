"""Aggregate raw benchmark cells into the published data (schema version 2).

Reads ``<input>/raw/<function>/<dim>/<optimizer>.json`` (written by
``run_benchmark_suite.py``, possibly merged from several CI shards) and writes

    index.json, cells/<function>-<dim>.json, ranks.json, trajectory-<function>-2.json

to the output directory, every file validated against :mod:`benchmarks.models`.
All statistics are computed from the unrounded raw data; only the written numbers
are rounded (4 significant digits) to keep the files small.

Usage:
    python benchmarks/aggregate_results.py --input benchmarks/output \
        --output-dir benchmarks/output/processed
"""

from __future__ import annotations

import argparse
import json
import operator
import sys

from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks import metrics  # noqa: E402
from benchmarks.models import Cell  # noqa: E402
from benchmarks.models import Index  # noqa: E402
from benchmarks.models import Ranks  # noqa: E402
from benchmarks.models import Trajectory  # noqa: E402
from benchmarks.protocol import BASELINE  # noqa: E402
from benchmarks.protocol import DEFAULT_CELL_TIME  # noqa: E402
from benchmarks.protocol import DEFAULT_TIME_CAP  # noqa: E402
from benchmarks.protocol import FUNCTIONS  # noqa: E402
from benchmarks.protocol import TARGETS  # noqa: E402


SIGNIFICANT_DIGITS = 4
TRAJECTORY_POINTS = 60
EXCLUDED = ["MOEAD", "NSGAII", "SPEA2"]
TEST_DESCRIPTION = (
    "One-sided Mann-Whitney U test that an optimizer's final precision is lower than "
    "the baseline's, on 15 independent runs, with precisions floored at the last "
    "target; p-values Holm-adjusted over the optimizers of each function and "
    "dimension; effect size is Cliff's delta."
)


def round_numbers(value: Any, digits: int = SIGNIFICANT_DIGITS) -> Any:  # noqa: ANN401
    """Round every float in a JSON-like structure to ``digits`` significant digits."""
    if isinstance(value, bool) or value is None or isinstance(value, int | str):
        return value
    if isinstance(value, float):
        return float(f"{value:.{digits}g}")
    if isinstance(value, list):
        return [round_numbers(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: round_numbers(v, digits) for k, v in value.items()}
    return value


def load_raw(raw_dir: Path) -> dict[tuple[str, int], dict[str, dict[str, Any]]]:
    """Load every raw cell, grouped as ``{(function, dim): {optimizer: cell}}``."""
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for path in sorted(raw_dir.glob("*/*/*.json")):
        cell = json.loads(path.read_text(encoding="utf-8"))
        grouped.setdefault((cell["function"], cell["dim"]), {})[cell["optimizer"]] = (
            cell
        )
    return grouped


def optimizer_cell(raw: dict[str, Any], f_opt: float) -> dict[str, Any]:
    """Compute the published metrics of one raw cell (without the baseline test)."""
    protocol = raw["protocol"]
    runs = metrics.usable_runs(raw)
    n_targets = len(protocol["targets"])
    errors = metrics.final_errors(raw, f_opt)
    budgets = metrics.ecdf_budgets(protocol["budget"], raw["dim"])

    dispersion = metrics.quantile_curves(runs, "dispersion_at", subtract_optimum=False)
    return {
        "runs": len(runs),
        "runs_planned": protocol["n_runs"],
        "failed": sum(r["status"] == "failed" for r in raw["runs"]),
        "truncated": sum(r["status"] == "truncated_time" for r in raw["runs"]),
        "exhausted": sum(bool(r["exhausted"]) for r in runs),
        "max_iter": protocol["max_iter"],
        "finals": [float(e) for e in errors],
        "stats": metrics.summary_stats(errors),
        "success": metrics.success_rates(runs, n_targets),
        "ert": [metrics.ert_and_sp1(runs, i)[0] for i in range(n_targets)],
        "sp1": [metrics.ert_and_sp1(runs, i)[1] for i in range(n_targets)],
        "ecdf": metrics.ecdf(runs, raw["dim"], budgets),
        "convergence": metrics.quantile_curves(
            runs, "best_at", f_opt, subtract_optimum=True
        ),
        "dispersion": dispersion["median"] if dispersion else None,
        "time": metrics.time_stats(runs),
        "test": None,
    }


def add_baseline_tests(
    optimizers: dict[str, dict[str, Any]], raw: dict[str, dict[str, Any]], f_opt: float
) -> None:
    """Fill ``test`` of every non-baseline optimizer of one function and dimension."""
    if BASELINE not in raw:
        return
    baseline_errors = metrics.final_errors(raw[BASELINE], f_opt)
    names, results = [], []
    for name in sorted(optimizers):
        if name == BASELINE:
            continue
        result = metrics.compare_to_baseline(
            metrics.final_errors(raw[name], f_opt), baseline_errors
        )
        if result is not None:
            names.append(name)
            results.append(result)
    adjusted = metrics.holm([r["p"] for r in results])
    for name, result, p_holm in zip(names, results, adjusted, strict=True):
        optimizers[name]["test"] = {**result, "p_holm": p_holm}


def thin(rows: list[list[float]], limit: int) -> list[list[float]]:
    """Keep at most ``limit`` evenly spread rows, always including the last."""
    if len(rows) <= limit:
        return rows
    keep = np.unique(np.linspace(0, len(rows) - 1, limit).astype(int))
    return [rows[i] for i in keep]


def write_json(path: Path, model: Any, *, validate: type | None = None) -> None:  # noqa: ANN401
    """Round, validate against ``validate`` and write ``model`` as compact JSON.

    The file ends with a newline, as the ``end-of-file-fixer`` hook requires.
    """
    payload = round_numbers(model)
    if validate is not None:
        validate.model_validate(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def page_links(optimizers_json: Path | None) -> dict[str, str]:
    """Map optimizer class names to their docs page paths."""
    if optimizers_json is None or not optimizers_json.exists():
        return {}
    data = json.loads(optimizers_json.read_text(encoding="utf-8"))
    return {o["class_name"]: o["link"] for o in data["optimizers"]}


def aggregate(
    input_dir: Path, output_dir: Path, optimizers_json: Path | None = None
) -> dict[str, int]:
    """Aggregate ``input_dir/raw`` into ``output_dir`` and return counts."""
    grouped = load_raw(input_dir / "raw")
    if not grouped:
        msg = f"No raw cells found in {input_dir / 'raw'}"
        raise FileNotFoundError(msg)

    meta_path = input_dir / "run-meta.json"
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    sample = next(iter(next(iter(grouped.values())).values()))["protocol"]

    links = page_links(optimizers_json)
    categories: dict[str, str] = {}
    cell_refs: list[dict[str, Any]] = []
    medians: dict[str, dict[str, float]] = {}
    written_files = 0

    for (function, dim), raw in sorted(grouped.items()):
        f_opt = FUNCTIONS[function].f_opt
        first = next(iter(raw.values()))["protocol"]
        optimizers = {name: optimizer_cell(cell, f_opt) for name, cell in raw.items()}
        add_baseline_tests(optimizers, raw, f_opt)

        block = f"{function}-{dim}"
        for name, cell in raw.items():
            categories[name] = cell["category"]
            errors = metrics.final_errors(cell, f_opt)
            if errors.size:
                medians.setdefault(block, {})[name] = float(
                    np.median(metrics.floored(errors))
                )

        cell_model = {
            "function": function,
            "dim": dim,
            "budget": first["budget"],
            "checkpoints": first["checkpoints"],
            "targets": first["targets"],
            "ecdf_budgets": [
                float(b) for b in metrics.ecdf_budgets(first["budget"], dim)
            ],
            "optimizers": optimizers,
        }
        filename = f"cells/{block}.json"
        write_json(output_dir / filename, cell_model, validate=Cell)
        cell_refs.append(
            {"function": function, "dim": dim, "file": filename, "optimizers": len(raw)}
        )
        written_files += 1

        if dim == 2:
            paths = {}
            seed = None
            for name, cell in raw.items():
                runs = [r for r in cell["runs"] if r["trajectory"]]
                if runs:
                    seed = runs[0]["seed"]
                    paths[name] = thin(runs[0]["trajectory"], TRAJECTORY_POINTS)
            if paths:
                write_json(
                    output_dir / f"trajectory-{function}-2.json",
                    {"function": function, "seed": seed, "optimizers": paths},
                    validate=Trajectory,
                )
                written_files += 1

    per_optimizer, mean_rank = metrics.average_ranks(medians)
    ranks_model = {
        "schema_version": 2,
        "blocks": sorted(medians),
        "entries": sorted(
            (
                {
                    "name": name,
                    "category": categories[name],
                    "mean_rank": mean_rank[name],
                    "ranks": per_optimizer[name],
                }
                for name in per_optimizer
            ),
            key=operator.itemgetter("mean_rank"),
        ),
        "friedman": metrics.friedman(medians),
    }
    write_json(output_dir / "ranks.json", ranks_model, validate=Ranks)

    dims = sorted({d for _, d in grouped})
    index_model = {
        "schema_version": 2,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": meta.get("git_sha"),
        "versions": {
            "python": meta.get("python", "unknown"),
            "numpy": meta.get("numpy", "unknown"),
            "scipy": meta.get("scipy", "unknown"),
        },
        "protocol": {
            "budget_per_dim": sample["budget_per_dim"],
            "targets": list(TARGETS),
            "n_runs": sample["n_runs"],
            "dimensions": dims,
            "functions": [
                {
                    "name": f.name,
                    "lower": f.lower,
                    "upper": f.upper,
                    "f_opt": f.f_opt,
                    "shifts": {str(d): [float(v) for v in f.shift(d)] for d in dims},
                }
                for f in FUNCTIONS.values()
                if any(fn == f.name for fn, _ in grouped)
            ],
            "baseline": BASELINE,
            "ecdf_points": metrics.ECDF_POINTS,
            "precision_floor": metrics.PRECISION_FLOOR,
            "time_cap_seconds": sample.get("time_cap", DEFAULT_TIME_CAP),
            "cell_time_seconds": sample.get("cell_time", DEFAULT_CELL_TIME),
            "test": TEST_DESCRIPTION,
            "excluded": EXCLUDED,
        },
        "optimizers": [
            {
                "name": name,
                "category": categories[name],
                "baseline": name == BASELINE,
                "page": links.get(name),
            }
            for name in sorted(categories)
        ],
        "cells": cell_refs,
    }
    write_json(output_dir / "index.json", index_model, validate=Index)

    return {
        "cells": len(cell_refs),
        "optimizers": len(categories),
        "files": written_files + 2,
    }


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", default="benchmarks/output", type=Path)
    parser.add_argument(
        "--output-dir", default="benchmarks/output/processed", type=Path
    )
    parser.add_argument(
        "--optimizers-json",
        default=Path("docs/public/optimizers/optimizers.json"),
        type=Path,
        help="used to link every optimizer to its docs page",
    )
    args = parser.parse_args(argv)
    try:
        counts = aggregate(args.input, args.output_dir, args.optimizers_json)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Aggregated {counts['cells']} function-dimension cells of "
        f"{counts['optimizers']} optimizers into {counts['files']} files in {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
