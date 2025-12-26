from __future__ import annotations

from pathlib import Path


def test_benchmarks_checklist_exists_and_references_benchmark_examples():
    p = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "checklists"
        / "benchmarks-spec.md"
    )
    assert p.exists(), "Missing `.github/checklists/benchmarks-spec.md`"
    text = p.read_text(encoding="utf-8")
    assert "benchmark(store=True)" in text or "benchmark_quick" in text, (
        "Checklist does not reference benchmark examples or quick tests"
    )
