from __future__ import annotations

from scripts.replace_trivial_doctests import apply_replacements
from scripts.replace_trivial_doctests import find_matches
from scripts.replace_trivial_doctests import propose_replacements


def test_dry_run_reports_matches(tmp_path):
    # Create a sample file with a trivial doctest
    sample_dir = tmp_path / "opt" / "sample"
    sample_dir.mkdir(parents=True)
    sample_file = sample_dir / "opt_example.py"
    sample_file.write_text(
        '''"""Example optimizer

>>> len(solution) == 10
"""
''',
        encoding="utf-8",
    )

    matches = find_matches(tmp_path)
    assert matches, "Dry-run should find at least one trivial doctest"
    proposals = propose_replacements(matches)
    assert isinstance(proposals, list)
    assert "path" in proposals[0]
    assert "original" in proposals[0]
    assert "suggested" in proposals[0]


def test_apply_replaces_line(tmp_path):
    sample_dir = tmp_path / "opt" / "sample"
    sample_dir.mkdir(parents=True)
    sample_file = sample_dir / "opt_example.py"
    sample_file.write_text(
        '''"""Example optimizer

>>> len(solution) == 10
"""
''',
        encoding="utf-8",
    )

    matches = find_matches(tmp_path)
    proposals = propose_replacements(matches)

    changed = apply_replacements(tmp_path, proposals)
    assert changed == 1

    content = sample_file.read_text(encoding="utf-8")
    assert "TODO: Replaced trivial doctest" in content
    assert "Suggested mini-benchmark" in content
