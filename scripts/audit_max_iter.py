#!/usr/bin/env python3
"""Audit repository for `max_iter` numeric literals.

Usage:
    python scripts/audit_max_iter.py --min 1000 --out reports/max_iter_audit.json

Outputs a JSON array of objects: {path, line, value, snippet}
"""

from __future__ import annotations

import argparse
import json
import re

from pathlib import Path


PATTERN = re.compile(r"max_iter\s*=\s*(\d+)")


def scan(root: Path, min_value: int) -> list[dict]:
    """Scan repository files for `max_iter` literals >= min_value, excluding common virtualenv and build dirs.

    Returns:
        list[dict]: each dict has path, line, value, snippet
    """
    EXCLUDE_DIRS = {".venv", "venv", "env", "build", "dist", "__pycache__"}

    def is_excluded(p: Path) -> bool:
        return any(part in EXCLUDE_DIRS for part in p.parts)

    results: list[dict] = []

    for p in root.rglob("*.py"):
        if is_excluded(p):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            m = PATTERN.search(line)
            if m:
                val = int(m.group(1))
                if val >= min_value:
                    results.append(
                        {
                            "path": str(p.relative_to(root)),
                            "line": i,
                            "value": val,
                            "snippet": line.strip(),
                        }
                    )
    # Also scan markdown and md files (docs) for literal examples
    for p in root.rglob("*.md"):
        if is_excluded(p):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            m = PATTERN.search(line)
            if m:
                val = int(m.group(1))
                if val >= min_value:
                    results.append(
                        {
                            "path": str(p.relative_to(root)),
                            "line": i,
                            "value": val,
                            "snippet": line.strip(),
                        }
                    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min",
        type=int,
        default=5000,
        help="Minimum max_iter value to report (default: 5000)",
    )
    parser.add_argument("--out", type=str, default="reports/max_iter_audit.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results = scan(repo_root, args.min)
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} results to {out_path}")
