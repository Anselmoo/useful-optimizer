"""IO helpers for benchmark data export (COCO/IOHprofiler friendly).

This module provides a small helper to export optimizer history and metadata to
a JSON file usable by docs and visualization tooling.
"""

from __future__ import annotations

import json
import numbers

from typing import Any


def _to_serializable(obj: Any):
    """Convert common numpy types and other objects to JSON-serializable values."""
    try:
        # numpy arrays / scalars expose .tolist()
        tolist = getattr(obj, "tolist", None)
        if callable(tolist):
            return tolist()
    except Exception:
        pass
    if isinstance(obj, numbers.Number):
        return float(obj)
    return obj


def save_run_history(optimizer: object, path: str) -> None:
    """Save optimizer history and metadata to JSON.

    The exported structure contains at least the keys:
      - best_fitness: list[float]
      - best_solution: list[list[float]]
      - final_result: dict with keys `best_fitness`, `best_solution`, `seed`
      - metadata: dict with `population_size`, `max_iter`, `seed`

    Args:
        optimizer: An optimizer instance with a `history` attribute and common
            attributes like `max_iter`, `population_size`, and `seed`.
        path: Path to write JSON file.
    """
    history = getattr(optimizer, "history", {}) or {}

    out = {
        "best_fitness": [_to_serializable(v) for v in history.get("best_fitness", [])],
        "best_solution": [
            _to_serializable(v) for v in history.get("best_solution", [])
        ],
        "final_result": {
            "best_fitness": (
                float(history.get("best_fitness", [])[-1])
                if history.get("best_fitness")
                else None
            ),
            "best_solution": (
                _to_serializable(history.get("best_solution", [])[-1])
                if history.get("best_solution")
                else None
            ),
            "seed": getattr(optimizer, "seed", None),
        },
        "metadata": {
            "population_size": getattr(optimizer, "population_size", None),
            "max_iter": getattr(optimizer, "max_iter", None),
            "seed": getattr(optimizer, "seed", None),
        },
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
