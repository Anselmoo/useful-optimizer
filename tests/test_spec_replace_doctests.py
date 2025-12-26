from __future__ import annotations

from pathlib import Path


def test_spec_contains_implementation_targets():
    p = (
        Path(__file__).resolve().parents[1]
        / "specs"
        / "002-replace-doctests-with-spec-driven-benchmarks"
        / "spec.md"
    )
    assert p.exists(), "Missing spec file for replace-doctests feature"
    text = p.read_text(encoding="utf-8")
    assert (
        "### Implementation Targets" in text
        or "Implementation Targets (mandatory)" in text
    ), "Implementation Targets section not found in spec"
    assert "In-spec minimal example" in text, (
        "Minimal in-spec example not found in spec"
    )
