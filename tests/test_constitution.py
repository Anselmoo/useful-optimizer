from __future__ import annotations

from pathlib import Path


def test_constitution_has_principles():
    p = Path(__file__).resolve().parents[1] / ".github" / "CONSTITUTION.md"
    assert p.exists(), "Missing `.github/CONSTITUTION.md`"
    text = p.read_text(encoding="utf-8")
    assert "## Principles" in text, "`Principles` heading not found in constitution"
