"""Script to detect and optionally fix COCO benchmark examples in optimizer docstrings.

Usage:
    python scripts/fix_coco_examples.py --dry-run    # list files that need change
    python scripts/fix_coco_examples.py --apply      # apply canonical snippet replacements

Behavior:
- Finds files containing the string 'COCO benchmark example:' and a trivial check
  like 'len(solution) == 10' and replaces the example with a canonical snippet that
  includes `track_history=True` and a call to `benchmarks.save_run_history()`.
- Replacements are idempotent and preserve the optimizer class name in-place.

This script is conservative: it only replaces examples that exactly match the
pattern with 'len(solution) == 10'. Use --dry-run to inspect before applying.
"""

from __future__ import annotations

import argparse
import re

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(
    r"(\n\s*COCO benchmark example:\n\n)([\s\S]*?)>>> solution, fitness = optimizer.search\(\)[\s\S]*?len\(solution\) == 10\n\s*True",
    re.MULTILINE,
)

CANON_TEMPLATE = (
    "\n        COCO benchmark example:\n\n"
    "        >>> from opt.benchmark.functions import sphere\n"
    "        >>> import tempfile, os\n"
    "        >>> from benchmarks import save_run_history\n"
    "        >>> optimizer = {class_name}(\n"
    "        ...     func=sphere,\n"
    "        ...     lower_bound=-5,\n"
    "        ...     upper_bound=5,\n"
    "        ...     dim=10,\n"
    "        ...     max_iter=10000,\n"
    "        ...     seed=42,\n"
    "        ...     track_history=True\n"
    "        ... )\n"
    "        >>> solution, fitness = optimizer.search()\n"
    "        >>> isinstance(fitness, float) and fitness >= 0\n"
    "        True\n"
    '        >>> len(optimizer.history.get("best_fitness", [])) > 0\n'
    "        True\n"
    "        >>> out = tempfile.NamedTemporaryFile(delete=False).name\n"
    "        >>> save_run_history(optimizer, out)\n"
    "        >>> os.path.exists(out)\n"
    "        True"
)


def find_files() -> list[Path]:
    """Return list of optimizer files that contain trivial COCO examples."""
    py_files = list((ROOT / "opt").rglob("*.py"))
    matches = []
    for p in py_files:
        text = p.read_text(encoding="utf-8")
        if "COCO benchmark example" in text and "len(solution) == 10" in text:
            matches.append(p)
    return matches


def replace_in_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    m = PATTERN.search(text)
    if not m:
        return False
    # Attempt to find the class name used in the example: look for "optimizer = ClassName(" on the example lines
    # Fallback to generic 'optimizer' if not found
    snippet = m.group(2)
    class_name_match = re.search(r"optimizer = ([A-Za-z0-9_]+)\(", snippet)
    class_name = class_name_match.group(1) if class_name_match else "OptimizerClass"
    new_snippet = CANON_TEMPLATE.format(class_name=class_name)
    new_text = text[: m.start(1)] + new_snippet + text[m.end(0) :]
    path.write_text(new_text, encoding="utf-8")
    return True


def main(*, dry_run: bool) -> None:
    """Detects and optionally applies fixes to COCO examples.

    Parameters
    ----------
    dry_run:
        If True: do not modify files, only report candidates. If False: apply changes.
    """
    files = find_files()
    if not files:
        print("No matching COCO examples found.")
        return
    print(f"Found {len(files)} files with trivial COCO examples:")
    for f in files:
        print(" -", f.relative_to(ROOT))

    if dry_run:
        print("\nDry-run complete. Use --apply to patch these examples.")
        return

    # Apply changes
    for f in files:
        changed = replace_in_file(f)
        print(("Updated" if changed else "Skipped"), f.relative_to(ROOT))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    main(dry_run=not args.apply)
