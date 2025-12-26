from __future__ import annotations

import json

from pathlib import Path

import jsonschema

from opt.multi_objective.nsga_ii import NSGAII


def f1(x):
    return float((x**2).sum())


def f2(x):
    return float(((x - 2) ** 2).sum())


def test_nsga2_benchmark_writes_valid_json(tmp_path: Path):
    out_file = tmp_path / "nsga-benchmark.json"
    opt = NSGAII(objectives=[f1, f2], lower_bound=-5, upper_bound=5, dim=3, max_iter=10)
    res = opt.benchmark(store=True, out_path=str(out_file))
    assert "path" in res
    assert Path(res["path"]).exists()

    schema_path = Path("docs/schemas/benchmark-data-schema.json")
    with Path(res["path"]).open(encoding="utf-8") as fh:
        data = json.load(fh)
    with schema_path.open(encoding="utf-8") as sh:
        schema = json.load(sh)
    jsonschema.Draft7Validator(schema).validate(data)
