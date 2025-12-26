from __future__ import annotations

import json

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from opt.benchmark.utils import export_benchmark_json
from opt.benchmark.utils import validate_benchmark_json


def _sample_valid_data() -> dict:
    return {
        "metadata": {
            "max_iterations": 10,
            "n_runs": 1,
            "dimensions": [2],
            "timestamp": "2025-01-01T00:00:00Z",
        },
        "benchmarks": {
            "sphere": {
                "2": {
                    "SimulatedAnnealing": {
                        "runs": [
                            {
                                "best_fitness": 0.1,
                                "best_solution": [0.0, 0.0],
                                "n_evaluations": 10,
                            }
                        ],
                        "statistics": {
                            "mean_fitness": 0.1,
                            "std_fitness": 0.0,
                            "min_fitness": 0.1,
                            "max_fitness": 0.1,
                            "median_fitness": 0.1,
                        },
                        "success_rate": 1.0,
                    }
                }
            }
        },
    }


def test_export_and_validate_writes_file_and_validates():
    data = _sample_valid_data()
    with TemporaryDirectory() as td:
        out_path = Path(td) / "out.json"
        returned = export_benchmark_json(
            data, out_path, schema_path=Path("docs/schemas/benchmark-data-schema.json")
        )
        assert Path(returned).exists()
        # validate explicitly
        assert validate_benchmark_json(
            returned, schema_path=Path("docs/schemas/benchmark-data-schema.json")
        )


def test_validate_raises_on_invalid_file():
    data = _sample_valid_data()
    # make it invalid by removing required metadata
    data.pop("metadata")
    with TemporaryDirectory() as td:
        out_path = Path(td) / "bad.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh)
        with pytest.raises(ValueError, match="Benchmark JSON validation failed"):
            validate_benchmark_json(
                out_path, schema_path=Path("docs/schemas/benchmark-data-schema.json")
            )
