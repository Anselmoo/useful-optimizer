"""Check Google-style docstring entries have inline summaries.

This enforces a small style rule used in the abstract base classes:
Within "Args:" and "Attributes:" sections, the item line should contain the
short description on the same line as the type.

Example (required):
    history (dict[str, list]): Dictionary containing optimization history.

Example (rejected):
    history (dict[str, list]):
        Dictionary containing optimization history.

This script is intentionally scoped via pre-commit to only a small set of files
to avoid mass changes across the repository.
"""

from __future__ import annotations

import argparse
import re

from pathlib import Path


_SECTION_HEADERS = {"args:", "arguments:", "attributes:"}

_NEXT_SECTION_HEADER_RE = re.compile(
    r"^\s{4,}[A-Z][A-Za-z0-9_\- ]*:\s*$"  # e.g. "Returns:", "Notes:", "Methods:"
)

# Match lines like:
#     foo (int): Description...
# where we require at least one non-space char after the colon.
_ITEM_WITH_INLINE_SUMMARY_RE = re.compile(r"^\s{8,}[A-Za-z_]\w*\s*\([^)]*\):\s*\S")

# Match lines like:
#     foo (int):
# (no inline summary) -> error
_ITEM_MISSING_INLINE_SUMMARY_RE = re.compile(r"^\s{8,}[A-Za-z_]\w*\s*\([^)]*\):\s*$")


def _check_file(path: Path) -> list[str]:
    errors: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()

    in_target_section = False

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip().lower()

        if stripped in _SECTION_HEADERS:
            in_target_section = True
            continue

        if in_target_section:
            # End the section when we hit a blank line or a new section header.
            if stripped == "" or _NEXT_SECTION_HEADER_RE.match(line):
                in_target_section = False
                continue

            # Ignore bullet lists inside sections.
            if line.lstrip().startswith("-"):
                continue

            if _ITEM_MISSING_INLINE_SUMMARY_RE.match(line):
                errors.append(
                    f"{path}:{idx}: Docstring entry missing inline summary after type: {line.strip()}"
                )
                continue

            # If it looks like an item, ensure it has an inline summary.
            if (
                "(" in line
                and ")" in line
                and ":" in line
                and line.startswith(" " * 8)
                and not _ITEM_WITH_INLINE_SUMMARY_RE.match(line)
            ):
                name_type_prefix = re.match(r"^\s{8,}[A-Za-z_]\w*\s*\([^)]*\):", line)
                if name_type_prefix and line.strip().endswith(":"):
                    errors.append(
                        f"{path}:{idx}: Docstring entry missing inline summary after type: {line.strip()}"
                    )

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="Files to check")
    args = parser.parse_args(argv)

    paths = [Path(p) for p in args.files]
    all_errors: list[str] = []

    for path in paths:
        if path.suffix != ".py" or not path.exists():
            continue
        all_errors.extend(_check_file(path))

    if all_errors:
        for err in all_errors:
            print(err)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
