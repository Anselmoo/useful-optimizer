"""Tests for the aggregator and the published schema."""

from __future__ import annotations

import json

from typing import TYPE_CHECKING

import pytest

from benchmarks.aggregate_results import aggregate
from benchmarks.aggregate_results import round_numbers
from benchmarks.aggregate_results import thin
from benchmarks.export_schema import SCHEMA_PATH
from benchmarks.export_schema import schema_text
from benchmarks.models import Cell
from benchmarks.models import Index
from benchmarks.models import Ranks
from benchmarks.models import Trajectory
from benchmarks.run_benchmark_suite import Cell as PlannedCell
from benchmarks.run_benchmark_suite import Settings
from benchmarks.run_benchmark_suite import run_and_store


if TYPE_CHECKING:
    from pathlib import Path


OPTIMIZERS = [
    ("RandomSearch", "classical"),
    ("HillClimbing", "classical"),
    ("ParticleSwarm", "swarm_intelligence"),
    ("DifferentialEvolution", "evolutionary"),
]


@pytest.fixture(scope="module")
def processed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("bench")
    settings = Settings(n_runs=6, budget_per_dim=100)
    for function in ("sphere", "rastrigin"):
        for dim in (2, 3):
            for name, category in OPTIMIZERS:
                run_and_store(
                    PlannedCell(function, dim, name, category), str(root), settings
                )
    out = root / "processed"
    aggregate(root, out)
    return out


def test_layout(processed: Path) -> None:
    names = {p.name for p in processed.iterdir()}
    assert {"index.json", "ranks.json", "cells"} <= names
    assert {p.name for p in (processed / "cells").iterdir()} == {
        "sphere-2.json",
        "sphere-3.json",
        "rastrigin-2.json",
        "rastrigin-3.json",
    }
    assert {"trajectory-sphere-2.json", "trajectory-rastrigin-2.json"} <= names


def test_files_validate_against_the_models(processed: Path) -> None:
    Index.model_validate_json((processed / "index.json").read_text())
    Ranks.model_validate_json((processed / "ranks.json").read_text())
    for path in (processed / "cells").iterdir():
        Cell.model_validate_json(path.read_text())
    Trajectory.model_validate_json((processed / "trajectory-sphere-2.json").read_text())


def test_index_describes_the_protocol(processed: Path) -> None:
    index = json.loads((processed / "index.json").read_text())
    assert index["protocol"]["baseline"] == "RandomSearch"
    assert index["protocol"]["budget_per_dim"] == 100
    assert index["protocol"]["dimensions"] == [2, 3]
    assert {o["name"] for o in index["optimizers"]} == {n for n, _ in OPTIMIZERS}
    assert [o["name"] for o in index["optimizers"] if o["baseline"]] == ["RandomSearch"]
    assert len(index["cells"]) == 4
    assert all((processed / c["file"]).exists() for c in index["cells"])


def test_cell_metrics_are_consistent(processed: Path) -> None:
    cell = json.loads((processed / "cells" / "rastrigin-3.json").read_text())
    assert cell["budget"] == 300
    n_checkpoints = len(cell["checkpoints"])
    for name, data in cell["optimizers"].items():
        assert data["runs"] == 6, name
        assert len(data["finals"]) == 6
        assert len(data["convergence"]["median"]) == n_checkpoints
        assert len(data["ecdf"]) == len(cell["ecdf_budgets"])
        assert data["ecdf"] == sorted(data["ecdf"])
        assert len(data["success"]) == len(cell["targets"])
        curve = [v for v in data["convergence"]["median"] if v is not None]
        assert curve == sorted(curve, reverse=True)  # best-so-far never gets worse


def test_baseline_tests_are_holm_adjusted(processed: Path) -> None:
    cell = json.loads((processed / "cells" / "rastrigin-3.json").read_text())[
        "optimizers"
    ]
    assert cell["RandomSearch"]["test"] is None
    others = [d["test"] for n, d in cell.items() if n != "RandomSearch"]
    assert all(t is not None for t in others)
    assert all(t["p_holm"] >= t["p"] for t in others)
    assert all(-1 <= t["delta"] <= 1 for t in others)


def test_ranks_cover_every_optimizer_and_block(processed: Path) -> None:
    ranks = json.loads((processed / "ranks.json").read_text())
    assert len(ranks["blocks"]) == 4
    assert {e["name"] for e in ranks["entries"]} == {n for n, _ in OPTIMIZERS}
    means = [e["mean_rank"] for e in ranks["entries"]]
    assert means == sorted(means)
    assert all(set(e["ranks"]) == set(ranks["blocks"]) for e in ranks["entries"])
    assert ranks["friedman"]["blocks"] == 4


def test_trajectories_are_thinned_and_end_at_the_best(processed: Path) -> None:
    data = json.loads((processed / "trajectory-sphere-2.json").read_text())
    assert data["function"] == "sphere"
    for rows in data["optimizers"].values():
        assert 1 <= len(rows) <= 60
        assert all(len(r) == 4 for r in rows)
        assert [r[0] for r in rows] == sorted(r[0] for r in rows)


def test_aggregation_is_deterministic(processed: Path, tmp_path: Path) -> None:
    raw_root = processed.parent
    again = tmp_path / "again"
    aggregate(raw_root, again)
    for name in ("cells/sphere-2.json", "ranks.json", "trajectory-sphere-2.json"):
        assert (processed / name).read_text() == (again / name).read_text()


def test_missing_input_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        aggregate(tmp_path, tmp_path / "out")


def test_rounding_and_thinning() -> None:
    assert round_numbers({"a": [1.23456789, None, 3], "b": True}) == {
        "a": [1.235, None, 3],
        "b": True,
    }
    rows = [[float(i)] for i in range(500)]
    thinned = thin(rows, 60)
    assert len(thinned) == 60
    assert thinned[0] == rows[0]
    assert thinned[-1] == rows[-1]


def test_committed_schema_matches_the_models() -> None:
    assert SCHEMA_PATH.read_text() == schema_text(), (
        "docs/schemas/benchmark-data-schema.json is stale: "
        "run `uv run python benchmarks/export_schema.py --types`"
    )
