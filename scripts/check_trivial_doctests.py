#!/usr/bin/env python3
"""Check for trivial doctest examples such as 'len(solution) == <dim>' in docstrings.

Exits with non-zero status if any matches are found. Intended for pre-commit/local runs.
"""

from __future__ import annotations

import re
import sys

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"len\(solution\)\s*==\s*\d+")

matches = []
for p in ROOT.rglob("opt/**/*.py"):
    try:
        txt = p.read_text()
    except Exception:
        continue
    for i, line in enumerate(txt.splitlines(), start=1):
        if PATTERN.search(line):
            matches.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")

if matches:
    print("Trivial doctest patterns found (please replace with spec-driven examples):")
    for m in matches:
        print(m)
    sys.exit(1)

print("No trivial doctest patterns found.")
sys.exit(0)
