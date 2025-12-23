#!/usr/bin/env python3
"""Fix excessive indentation in Returns/Raises/Notes sections of docstrings.

Converts 16-space indentation to proper 8-space indentation for section content.
"""

from __future__ import annotations

import re

from pathlib import Path


def fix_docstring_indentation(file_path: Path) -> bool:
    """Fix indentation issues in a single file.

    Args:
        file_path: Path to the Python file to fix

    Returns:
        True if file was modified, False otherwise
    """
    content = file_path.read_text(encoding="utf-8")
    original_content = content
    lines = content.splitlines(keepends=True)

    # Track if we're in a docstring section that needs fixing
    in_section = False
    section_name = None
    modified_lines = []

    for line in lines:
        stripped = line.strip().lower()

        # Check if this is a section header (Returns:, Raises:, Notes:, etc.)
        if stripped in (
            "returns:",
            "raises:",
            "yields:",
            "notes:",
            "examples:",
            "references:",
        ):
            in_section = True
            section_name = stripped.rstrip(":")
            modified_lines.append(line)
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
            continue

        # Fix indentation if in a section
        if in_section:
            # Count leading spaces
            leading_spaces = len(line) - len(line.lstrip())

            # If line has 12+ spaces (excessive indentation), reduce to 8
            if leading_spaces >= 12 and line.strip():
                # Standard Google style uses 8 spaces for section content
                fixed_line = "        " + line.lstrip()
                modified_lines.append(fixed_line)
            else:
                modified_lines.append(line)
        else:
            modified_lines.append(line)

    new_content = "".join(modified_lines)

    if new_content != original_content:
        file_path.write_text(new_content, encoding="utf-8")
        return True

    return False


def main() -> None:
    """Fix indentation in all Python files in opt/ directory."""
    opt_dir = Path("opt")
    fixed_count = 0

    for py_file in opt_dir.rglob("*.py"):
        if fix_docstring_indentation(py_file):
            print(f"Fixed: {py_file}")
            fixed_count += 1

    print(f"\nTotal files fixed: {fixed_count}")


if __name__ == "__main__":
    main()
