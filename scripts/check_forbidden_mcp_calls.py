#!/usr/bin/env python3
"""Pre-commit script: detect forbidden MCP/agent calls in library code.

Exits non-zero if any forbidden pattern is found in files under `opt/`.

Usage:
    python scripts/check_forbidden_mcp_calls.py [paths...]

If no paths provided, scans `opt/` recursively.
"""

from __future__ import annotations

import argparse
import re

from pathlib import Path


FORBIDDEN_PATTERNS = [
    r"\bmcp_\w+",  # mcp_ prefixed calls
    r"\bmcpContext7\b",
    r"\bagent\.run\b",
    r"\bopenai\.\w+\b",
]


def scan(path: Path) -> list[tuple[Path, int, str]]:
    matches: list[tuple[Path, int, str]] = []
    for p in path.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        text = p.read_text(encoding="utf-8")
        for pat in FORBIDDEN_PATTERNS:
            for m in re.finditer(pat, text):
                lineno = text.count("\n", 0, m.start()) + 1
                excerpt = text.splitlines()[lineno - 1].strip()
                matches.append((p, lineno, excerpt))
    return matches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="Paths to scan (default: opt/)")
    args = parser.parse_args(argv)

    paths = [Path(p) for p in args.paths] if args.paths else [Path("opt")]
    all_matches: list[tuple[Path, int, str]] = []
    for p in paths:
        if not p.exists():
            print(f"[WARN] Path not found: {p}")
            continue
        all_matches.extend(scan(p))

    if not all_matches:
        print("[OK] No forbidden MCP/agent patterns found in scanned paths.")
        return 0

    print("[ERROR] Forbidden MCP/agent patterns found:")
    for p, lineno, excerpt in all_matches:
        print(f" - {p}:{lineno}: {excerpt}")

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
