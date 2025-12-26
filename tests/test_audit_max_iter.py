from __future__ import annotations

import json

from pathlib import Path


def test_audit_max_iter_finds_many():
    p = (
        Path(__file__).resolve().parents[1]
        / "specs"
        / "002-replace-doctests-with-spec-driven-benchmarks"
        / "discovery"
        / "files.json"
    )
    assert p.exists(), "Missing audit file"
    data = json.loads(p.read_text())
    # After excluding virtualenv and build dirs, we still expect many occurrences (>=100)
    assert len(data) >= 100, (
        f"Expected >=100 occurrences after excluding .venv, found {len(data)}"
    )
    # Ensure no results come from .venv or other excluded dirs
    assert not any(
        d["path"].startswith(".venv/") or d["path"].startswith("venv/") for d in data
    ), "Audit results include virtualenv paths"
