#!/usr/bin/env python3
"""Fix NumPy 2.x doctest boolean assertion failures.

This script replaces doctest assertions that fail with NumPy 2.x because they
return np.True_ instead of Python True.

Pattern to fix:
    >>> isinstance(fitness, float) and fitness >= 0
    True

Replace with:
    >>> bool(isinstance(fitness, float) and fitness >= 0)
    True
"""

from __future__ import annotations

import sys
from pathlib import Path


def fix_doctest_bool_in_file(file_path: Path) -> bool:
    """Fix NumPy boolean doctest assertions in a single file.
    
    Args:
        file_path: Path to the Python file to fix
        
    Returns:
        True if file was modified, False otherwise
    """
    content = file_path.read_text()
    original_content = content
    
    # Pattern 1: isinstance(fitness, float) and fitness >= 0
    # Find the doctest line and wrap it in bool()
    lines = content.split('\n')
    modified = False
    
    for i, line in enumerate(lines):
        # Look for the specific pattern
        if '>>> isinstance(fitness, float) and fitness >= 0' in line:
            # Check if next line expects True (and hasn't been fixed yet)
            if i + 1 < len(lines) and lines[i + 1].strip() == 'True':
                # Check if not already wrapped in bool()
                if 'bool(' not in line:
                    # Wrap the expression in bool()
                    lines[i] = line.replace(
                        'isinstance(fitness, float) and fitness >= 0',
                        'bool(isinstance(fitness, float) and fitness >= 0)'
                    )
                    modified = True
                    print(f"  Fixed line {i + 1}: {file_path.name}")
    
    if modified:
        file_path.write_text('\n'.join(lines))
        return True
    
    return False


def main() -> int:
    """Main entry point for the script."""
    # Find all Python files in opt/ directory
    opt_dir = Path(__file__).parent.parent / 'opt'
    
    if not opt_dir.exists():
        print(f"Error: {opt_dir} does not exist", file=sys.stderr)
        return 1
    
    # Find all .py files recursively
    py_files = list(opt_dir.rglob('*.py'))
    
    print(f"Found {len(py_files)} Python files in {opt_dir}")
    print("Fixing NumPy 2.x doctest boolean assertions...\n")
    
    fixed_count = 0
    for py_file in py_files:
        if fix_doctest_bool_in_file(py_file):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")
    return 0


if __name__ == '__main__':
    sys.exit(main())
