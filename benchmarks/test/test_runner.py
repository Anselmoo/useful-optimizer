"""Tests for the suite runner: planning, sharding, storage and resume."""

from __future__ import annotations

import json
import operator

from typing import TYPE_CHECKING

import pytest

from benchmarks.protocol import BASELINE
from benchmarks.protocol import FUNCTIONS
from benchmarks.protocol import discover_optimizers
from benchmarks.run_benchmark_suite import DIMENSIONS
from benchmarks.run_benchmark_suite import RAW_SCHEMA
from benchmarks.run_benchmark_suite import Cell
from benchmarks.run_benchmark_suite import Settings
from benchmarks.run_benchmark_suite import is_done
from benchmarks.run_benchmark_suite import list_shards
from benchmarks.run_benchmark_suite import main
from benchmarks.run_benchmark_suite import plan_cells
from benchmarks.run_benchmark_suite import run_and_store
from benchmarks.run_benchmark_suite import shard_cells


if TYPE_CHECKING:
    from pathlib import Path


def test_tiers_always_contain_the_baseline() -> None:
    for tier, expected in (("showcase", 5), ("standard", 14)):
        names = {c.optimizer for c in plan_cells(tier)}
        assert BASELINE in names
        assert len(names) == expected


def test_cell_count_is_the_full_product() -> None:
    cells = plan_cells("showcase")
    assert len(cells) == 5 * len(FUNCTIONS) * len(DIMENSIONS)
    assert len(set(cells)) == len(cells)


def test_comprehensive_covers_the_whole_library() -> None:
    names = {
        c.optimizer
        for c in plan_cells("comprehensive", dimensions=[2], functions=["sphere"])
    }
    assert names == set(discover_optimizers())


def test_filters() -> None:
    cells = plan_cells(
        "comprehensive",
        categories={"classical"},
        functions=["sphere"],
        dimensions=[2, 5],
    )
    assert {c.category for c in cells} == {"classical"}
    assert {c.dim for c in cells} == {2, 5}
    assert BASELINE in {c.optimizer for c in cells}  # RandomSearch is classical

    only = plan_cells(
        "comprehensive", optimizer_names={"ParticleSwarm"}, dimensions=[2]
    )
    assert {c.optimizer for c in only} == {"ParticleSwarm"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tier": "nope"},
        {"tier": "showcase", "functions": ["nope"]},
        {"tier": "showcase", "optimizer_names": {"Nope"}},
    ],
)
def test_planning_rejects_unknown_names(kwargs: dict) -> None:
    with pytest.raises(ValueError, match=r"nope|Nope"):
        plan_cells(**kwargs)


def test_shards_are_disjoint_and_complete() -> None:
    cells = plan_cells("standard")
    slices = [shard_cells(cells, i, 7) for i in range(7)]
    flat = [c for part in slices for c in part]
    assert sorted(flat, key=lambda c: (c.function, c.dim, c.optimizer)) == sorted(
        cells, key=lambda c: (c.function, c.dim, c.optimizer)
    )
    assert len(flat) == len(set(flat))


def test_ci_matrix_is_category_times_function() -> None:
    cells = plan_cells("comprehensive")
    matrix = list_shards(cells)
    assert len(matrix) == len({c.category for c in cells}) * len(FUNCTIONS)
    assert matrix == sorted(matrix, key=operator.itemgetter("category", "function"))
    covered = {(m["category"], m["function"]) for m in matrix}
    assert all((c.category, c.function) in covered for c in cells)


def test_run_store_and_resume(tmp_path: Path) -> None:
    cell = Cell("sphere", 2, "RandomSearch", "classical")
    settings = Settings(n_runs=2, budget_per_dim=50)
    assert not is_done(cell, tmp_path, settings)

    summary = run_and_store(cell, str(tmp_path), settings)
    path = cell.path(tmp_path)
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()
    assert summary["runs"] == 2

    stored = json.loads(path.read_text())
    assert stored["schema"] == RAW_SCHEMA
    assert stored["optimizer"] == "RandomSearch"
    assert all(r["n_evaluations"] == 100 for r in stored["runs"])

    assert is_done(cell, tmp_path, settings)
    assert not is_done(cell, tmp_path, Settings(n_runs=3, budget_per_dim=50))
    assert not is_done(cell, tmp_path, Settings(n_runs=2, budget_per_dim=60))

    path.write_text("{not json")
    assert not is_done(cell, tmp_path, settings)


def test_cli_dry_run_and_shard_listing(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--tier", "showcase", "--dry-run", "--functions", "sphere"]) == 0
    assert "20 cells planned" in capsys.readouterr().out

    assert main(["--tier", "showcase", "--list-shards"]) == 0
    matrix = json.loads(capsys.readouterr().out)
    assert {"category": "classical", "function": "sphere"} in matrix

    assert main(["--tier", "showcase", "--functions", "nope"]) == 2
