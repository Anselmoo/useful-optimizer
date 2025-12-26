from __future__ import annotations

import json

from pathlib import Path


def _load_tasks(path: Path) -> list:
    return json.load(path.open())["tasks"]


def test_tasks_with_mcp_have_logging_instruction():
    root = Path(__file__).resolve().parents[1] / "tasks.json"
    spec = (
        Path(__file__).resolve().parents[1]
        / "specs"
        / "002-replace-doctests-with-spec-driven-benchmarks"
        / "tasks.json"
    )
    for p in [root, spec]:
        tasks = _load_tasks(p)
        for t in tasks:
            if "mcp" in t:
                assert t.get("mcp_log_instruction"), (
                    f"Task {t['id']} in {p} declares MCP usage but has no mcp_log_instruction"
                )
