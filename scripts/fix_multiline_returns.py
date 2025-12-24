#!/usr/bin/env python3
"""Fix multi-line Returns/Raises/Yields descriptions.

Merges type declarations with their descriptions onto a single line as required
by Google Python Style Guide.
"""

from __future__ import annotations

import re

from pathlib import Path


def fix_multiline_descriptions(file_path: Path) -> bool:
    """Fix multi-line descriptions in Returns/Raises/Yields sections.

    Args:
        file_path: Path to the Python file to fix

    Returns:
        True if file was modified, False otherwise
    """
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)

    in_section = False
    section_name = None
    modified_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip().lower()

        # Check if this is a section header
        if stripped in ("returns:", "raises:", "yields:"):
            in_section = True
            section_name = stripped.rstrip(":")
            modified_lines.append(line)
            i += 1
            continue

        # Exit section on blank line, closing quotes, or new section header
        if in_section and (
            stripped == ""
            or stripped in ('"""', "'''")
            or re.match(r"^[A-Z][A-Za-z0-9_\- ]*:\s*$", stripped)
        ):
            in_section = False
            section_name = None
            modified_lines.append(line)
            i += 1
            continue

        # Fix multi-line descriptions
        if in_section:
            # Check if this line is a type declaration ending with ":"
            # Pattern: 8+ spaces, type info, colon, nothing after
            if re.match(r"^\s{8,}[\w\[\],\s|]+:\s*$", line) and i + 1 < len(lines):
                # Check if next line has the description
                next_line = lines[i + 1]
                next_stripped = next_line.strip()

                # If next line has content and isn't a new section/marker
                if (
                    next_stripped
                    and not next_stripped.startswith("-")
                    and next_stripped.lower()
                    not in (
                        "returns:",
                        "raises:",
                        "yields:",
                        "notes:",
                        "examples:",
                        '"""',
                        "'''",
                    )
                    and not re.match(r"^[A-Z][A-Za-z0-9_\- ]*:\s*$", next_stripped)
                ):
                    # Merge: keep the type line's indentation and colon, add description
                    merged_line = line.rstrip() + " " + next_stripped + "\n"
                    modified_lines.append(merged_line)
                    i += 2  # Skip both lines
                    continue

            modified_lines.append(line)
        else:
            modified_lines.append(line)

        i += 1

    new_content = "".join(modified_lines)

    if new_content != content:
        file_path.write_text(new_content, encoding="utf-8")
        return True

    return False


def main() -> None:
    """Fix multi-line descriptions in all Python files in opt/ directory."""
    opt_dir = Path("opt")
    fixed_count = 0

    for py_file in opt_dir.rglob("*.py"):
        if fix_multiline_descriptions(py_file):
            print(f"Fixed: {py_file}")
            fixed_count += 1

    print(f"\nTotal files fixed: {fixed_count}")


if __name__ == "__main__":
    main()
