from __future__ import annotations

import json

from pathlib import Path

import jsonschema

from opt.benchmark.functions import sphere
from opt.classical.simulated_annealing import SimulatedAnnealing


def test_simulated_annealing_benchmark_writes_valid_json(tmp_path: Path):
    out_file = tmp_path / "sa-benchmark.json"
    opt = SimulatedAnnealing(
        func=sphere, lower_bound=-5, upper_bound=5, dim=2, max_iter=10, seed=42
    )
    res = opt.benchmark(store=True, out_path=str(out_file))
    assert "path" in res
    assert Path(res["path"]).exists()

    # Validate against schema
    schema_path = Path("docs/schemas/benchmark-data-schema.json")
    with Path(res["path"]).open(encoding="utf-8") as fh:
        data = json.load(fh)
    with schema_path.open(encoding="utf-8") as sh:
        schema = json.load(sh)
    jsonschema.Draft7Validator(schema).validate(data)
