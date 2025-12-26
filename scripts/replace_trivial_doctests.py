#!/usr/bin/env python3
"""Replace trivial doctest examples such as 'len(solution) == <dim>' with suggested mini-benchmark examples.

This script supports two modes:
- --dry-run (default): print a JSON report of proposed replacements and exit without modifying files.
- --apply: apply the suggested replacement in-place (still conservative; adds a TODO and suggested snippet).

The suggested replacement is intentionally conservative and must be reviewed manually before committing.
"""

from __future__ import annotations

import argparse
import json
import re

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"(\s*)>>>\s*len\(solution\)\s*==\s*(\d+)")


def find_matches(root: Path) -> list[dict]:
    matches: list[dict] = []
    for p in root.rglob("opt/**/*.py"):
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(txt.splitlines(), start=1):
            m = PATTERN.search(line)
            if m:
                indent = m.group(1)
                dim = int(m.group(2))
                matches.append(
                    {
                        "path": str(p.relative_to(root)),
                        "line": i,
                        "text": line.rstrip(),
                        "indent": indent,
                        "dim": dim,
                    }
                )
    return matches


SUGGESTED_SNIPPET = (
    "{indent}>>> # TODO: Replaced trivial doctest with a suggested mini-benchmark — please review.\n"
    "{indent}>>> # Suggested mini-benchmark (seeded, quick):\n"
    "{indent}>>> # >>> res = optimizer.benchmark(store=True, quick=True, quick_max_iter=10, seed=0)\n"
    "{indent}>>> # >>> assert isinstance(res, dict) and res.get('metadata') is not None\n"
)


def propose_replacements(matches: list[dict]) -> list[dict]:
    proposals: list[dict] = []
    for m in matches:
        proposals.append(
            {
                "path": m["path"],
                "line": m["line"],
                "original": m["text"],
                "suggested": SUGGESTED_SNIPPET.format(indent=m["indent"]),
            }
        )
    return proposals


def apply_replacements(root: Path, proposals: list[dict]) -> int:
    grouped = {}
    for p in proposals:
        grouped.setdefault(p["path"], []).append(p)

    changed = 0
    for relpath, items in grouped.items():
        fp = root / relpath
        txt = fp.read_text(encoding="utf-8")
        lines = txt.splitlines()
        # Apply from last to first so indices don't shift
        for item in sorted(items, key=lambda x: x["line"], reverse=True):
            idx = item["line"] - 1
            if 0 <= idx < len(lines) and lines[idx].strip().startswith(">>>"):
                indent = item["original"][: item["original"].index(">>>")]
                snippet = item["suggested"].rstrip("\n")
                snippet_lines = snippet.splitlines()
                # Remove the original line, insert snippet lines at that position
                lines[idx : idx + 1] = snippet_lines
                changed += 1
        fp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply", action="store_true", help="Apply the suggested replacements in-place"
    )
    parser.add_argument(
        "--report", type=Path, help="Optional path to write JSON report"
    )
    args = parser.parse_args(argv)

    matches = find_matches(ROOT)
    proposals = propose_replacements(matches)

    if args.report:
        args.report.write_text(json.dumps(proposals, indent=2), encoding="utf-8")

    if not proposals:
        print("No trivial doctest patterns found.")
        return 0

    print(json.dumps(proposals, indent=2))

    if args.apply:
        changed = apply_replacements(ROOT, proposals)
        print(
            f"Applied {changed} replacements; please review changes before committing."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
