"""Tests ensuring COCO benchmark examples record history or save run history."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_coco_examples_record_history() -> None:
    py_files = list((ROOT / "opt").rglob("*.py"))
    offenders = []
    for p in py_files:
        text = p.read_text(encoding="utf-8")
        if "COCO benchmark example" in text:
            # Heuristic: example must include either `track_history=True` or `save_run_history(`
            # Search in the snippet starting at 'COCO benchmark example' up to 60 lines after
            idx = text.find("COCO benchmark example")
            snippet = text[idx : idx + 4000]
            if ("track_history=True" not in snippet) and (
                "save_run_history(" not in snippet
            ):
                offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, (
        f"COCO benchmark examples must record history or save run: {offenders}"
    )
