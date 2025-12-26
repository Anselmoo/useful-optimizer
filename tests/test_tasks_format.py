from __future__ import annotations

import re

from pathlib import Path


def test_tasks_file_and_format():
    p = Path(__file__).resolve().parents[1] / "tasks.md"
    assert p.exists(), "Missing `tasks.md`"
    text = p.read_text(encoding="utf-8")
    # Check TIDs present
    for i in range(1, 12):
        tid = f"T{i:03d}"
        assert tid in text, f"Task id {tid} not found in tasks.md"
    # Check owners and acceptance commands presence
    assert re.search(r"Owner: @\w+", text), "No owner mentions found"
    assert re.search(r"Acceptance:", text), "No Acceptance lines found in tasks.md"
