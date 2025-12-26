from __future__ import annotations

import gzip
import json
import sys

from pathlib import Path

import jsonschema
import pytest

from opt.benchmark.functions import sphere
from opt.probabilistic.sequential_monte_carlo import SequentialMonteCarloOptimizer


SCHEMA_PATH = Path("docs/schemas/benchmark-data-schema.json")


def generate_sample_artifact(tmp_path: Path) -> Path:
    # Run a short seeded benchmark and write a compressed artifact following schema
    opt = SequentialMonteCarloOptimizer(
        func=sphere,
        lower_bound=-5,
        upper_bound=5,
        dim=10,
        max_iter=50,
        seed=42,
        track_history=True,
    )
    solution, fitness = opt.search()

    artifact = {
        "metadata": {
            "max_iterations": 50,
            "n_runs": 1,
            "dimensions": [10],
            "timestamp": "2025-12-25T00:00:00Z",
            "python_version": sys.version.split()[0],
            "numpy_version": __import__("numpy").__version__,
        },
        "benchmarks": {
            "sphere": {
                "10": {
                    "SequentialMonteCarloOptimizer": {
                        "runs": [
                            {
                                "best_fitness": float(fitness),
                                "best_solution": list(map(float, solution.tolist())),
                                "n_evaluations": 50,
                                "history": {
                                    "best_fitness": opt.history.get("best_fitness", []),
                                    "mean_fitness": opt.history.get("mean_fitness", []),
                                },
                            }
                        ],
                        "statistics": {
                            "mean_fitness": float(fitness),
                            "std_fitness": 0.0,
                            "min_fitness": float(fitness),
                            "max_fitness": float(fitness),
                            "median_fitness": float(fitness),
                        },
                        "success_rate": 1.0,
                    }
                }
            }
        },
    }

    out = tmp_path / "history.json.gz"
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(artifact, f)
    return out


def test_history_artifact_schema_and_size(tmp_path):
    schema = json.load(open(SCHEMA_PATH))
    artifact_path = generate_sample_artifact(tmp_path)

    # Size check
    size_bytes = artifact_path.stat().st_size
    assert size_bytes <= 200 * 1024 * 1024, f"Artifact too large: {size_bytes} bytes"

    # Schema validation
    with gzip.open(artifact_path, "rt", encoding="utf-8") as f:
        content = json.load(f)
    jsonschema.validate(content, schema)


if __name__ == "__main__":
    pytest.main([str(Path(__file__))])
