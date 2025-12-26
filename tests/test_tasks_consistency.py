from __future__ import annotations

import json
import re

from pathlib import Path


def test_single_canonical_tasks_md():
    # Ensure root tasks.md points to canonical spec tasks.md
    root = Path(__file__).resolve().parents[1] / "tasks.md"
    text = root.read_text(encoding="utf-8")
    assert (
        "Canonical tasks path" in text
        or "specs/002-replace-doctests-with-spec-driven-benchmarks/tasks.md" in text
    ), "Root tasks.md must point to canonical spec tasks.md"


def test_tasks_json_matches_spec_tasks():
    spec_tasks = (
        Path(__file__).resolve().parents[1]
        / "specs"
        / "002-replace-doctests-with-spec-driven-benchmarks"
        / "tasks.md"
    )
    text = spec_tasks.read_text(encoding="utf-8")
    tids_in_spec = set(re.findall(r"\bT\d{3}\b", text))
    with (Path(__file__).resolve().parents[1] / "tasks.json").open(
        encoding="utf-8"
    ) as fh:
        tasks_json = json.load(fh)
    tids_in_json = {item["id"] for item in tasks_json["tasks"]}
    assert tids_in_spec.issuperset(tids_in_json) or tids_in_json.issuperset(
        tids_in_spec
    ), (
        f"Task ID sets differ between spec tasks ({sorted(tids_in_spec)}) and tasks.json ({sorted(tids_in_json)})"
    )
