"""Utilities to export and validate benchmark JSON artifacts.

Functions:
- export_benchmark_json(data, path=None, schema_path='docs/schemas/benchmark-data-schema.json') -> Path
- validate_benchmark_json(path, schema_path='docs/schemas/benchmark-data-schema.json') -> bool

These helpers are deterministic, local, and have no external network calls.
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any


def _load_schema(schema_path: Path) -> dict:
    with schema_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_benchmark_json(
    path: str | Path,
    schema_path: str | Path = "docs/schemas/benchmark-data-schema.json",
) -> bool:
    """Validate a benchmark JSON file against the repository schema.

    Args:
        path: Path to JSON file to validate.
        schema_path: Path to the JSON schema file.

    Returns:
        True if the file validates; raises ValueError on validation failure.
    """
    p = Path(path)
    s = Path(schema_path)
    if not p.exists():
        msg = f"Benchmark JSON not found: {p}"
        raise FileNotFoundError(msg)
    if not s.exists():
        msg = f"Schema not found: {s}"
        raise FileNotFoundError(msg)

    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    try:
        from jsonschema import Draft7Validator

        Draft7Validator(_load_schema(s)).validate(data)
    except (
        ModuleNotFoundError
    ) as exc:  # pragma: no cover - environment should have jsonschema
        msg = "jsonschema is required to validate benchmark JSON files"
        raise RuntimeError(msg) from exc
    except Exception as exc:  # pragma: no cover - validation errors
        msg = f"Benchmark JSON validation failed for {p}: {exc}"
        raise ValueError(msg) from exc
    else:
        return True


def export_benchmark_json(
    data: dict[str, Any],
    path: str | Path | None = None,
    *,
    schema_path: str | Path = "docs/schemas/benchmark-data-schema.json",
) -> str:
    """Write a benchmark JSON artifact and validate it against the schema.

    If `path` is not provided, a file will be created under `benchmarks/output/` with a safe name.

    Returns:
        The path to the written JSON file (as string).
    """
    out_dir = Path("benchmarks/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    if path is None:
        # Use a simple deterministic naming convention when not provided
        import time

        ts = int(time.time())
        path = out_dir / f"benchmark-{ts}.json"
    else:
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

    def _to_serializable(o: object) -> object:
        import numpy as np

        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, dict):
            return {k: _to_serializable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_serializable(v) for v in o]
        return o

    with path.open("w", encoding="utf-8") as fh:
        json.dump(_to_serializable(data), fh, indent=2, ensure_ascii=False)

    # Validate after writing
    validate_benchmark_json(path, schema_path=schema_path)

    return str(path)
