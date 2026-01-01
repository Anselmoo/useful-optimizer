#!/usr/bin/env python3
"""Replace trivial doctests with COCO-compliant benchmark tests.

This script replaces the basic doctest examples in optimizer files with
proper COCO/BBOB compliant benchmark tests using run_single_benchmark.
"""

from __future__ import annotations

import re
import sys

from pathlib import Path


def get_optimizer_class_name(file_path: Path) -> str | None:
    """Extract the main optimizer class name from a Python file.

    Args:
        file_path: Path to the optimizer Python file

    Returns:
        Class name if found, None otherwise
    """
    content = file_path.read_text()

    # Look for class definition that inherits from AbstractOptimizer
    match = re.search(
        r"class\s+(\w+)\s*\([^)]*AbstractOptimizer[^)]*\):",
        content
    )

    if match:
        return match.group(1)

    # Fallback: look for any class definition
    match = re.search(r"class\s+(\w+)\s*\(", content)
    if match:
        return match.group(1)

    return None


def create_benchmark_doctest(class_name: str, module_path: str) -> str:
    """Create the new COCO-compliant doctest examples.

    Args:
        class_name: Name of the optimizer class
        module_path: Python module path (e.g., "opt.swarm_intelligence.particle_swarm")

    Returns:
        String containing the new doctest examples
    """
    return f"""    Example:
        COCO/BBOB compliant benchmark test:

        >>> from benchmarks.run_benchmark_suite import run_single_benchmark
        >>> from {module_path} import {class_name}
        >>> from opt.benchmark.functions import shifted_ackley
        >>> result = run_single_benchmark(
        ...     {class_name}, shifted_ackley, -32.768, 32.768,
        ...     dim=2, max_iter=50, seed=42
        ... )
        >>> result["status"] == "success"
        True
        >>> "convergence_history" in result
        True

        Metadata validation:

        >>> required_keys = {{"optimizer", "best_fitness", "best_solution", "status"}}
        >>> required_keys.issubset(result.keys())
        True
"""


def replace_doctests_in_file(file_path: Path) -> bool:
    """Replace doctests in a single optimizer file.

    Args:
        file_path: Path to the optimizer Python file

    Returns:
        True if file was modified, False otherwise
    """
    content = file_path.read_text()
    original_content = content

    # Get class name
    class_name = get_optimizer_class_name(file_path)
    if not class_name:
        print(f"  WARNING: Could not find class name in {file_path.name}")
        return False

    # Build module path from file path
    # e.g., opt/swarm_intelligence/particle_swarm.py -> opt.swarm_intelligence.particle_swarm
    rel_path = file_path.relative_to(file_path.parent.parent.parent)
    module_path = str(rel_path.with_suffix("")).replace("/", ".")

    # Pattern to match the Example section with both doctest blocks
    # This matches from "Example:" through both doctest blocks
    pattern = r"(    Example:\n        Basic usage with BBOB benchmark function:\n\n)(        >>> from .*?\n.*?>>> bool\(isinstance\(fitness, float\) and fitness >= 0\)\n        True\n\n        COCO benchmark example:\n\n        >>> from opt\.benchmark\.functions import sphere\n.*?>>> len\(solution\) == 10\n        True\n)"

    new_doctest = create_benchmark_doctest(class_name, module_path)

    # Try to replace
    new_content, count = re.subn(
        pattern,
        new_doctest,
        content,
        flags=re.DOTALL
    )

    if count > 0:
        file_path.write_text(new_content)
        print(f"  ✓ Replaced doctests in {file_path.name}")
        return True
    # Try alternative pattern matching - might have slightly different format
    print(f"  ⚠ Pattern not found in {file_path.name}, trying alternative...")
    return False


def main() -> int:
    """Main entry point."""
    opt_dir = Path(__file__).parent.parent / "opt"

    if not opt_dir.exists():
        print(f"Error: {opt_dir} does not exist", file=sys.stderr)
        return 1

    # Get all optimizer subdirectories
    categories = [
        "classical",
        "constrained",
        "evolutionary",
        "gradient_based",
        "metaheuristic",
        "physics_inspired",
        "probabilistic",
        "social_inspired",
        "swarm_intelligence",
    ]

    print("Replacing trivial doctests with COCO-compliant benchmarks...\n")

    total_files = 0
    modified_files = 0

    for category in categories:
        category_dir = opt_dir / category
        if not category_dir.exists():
            continue

        print(f"Processing {category}/:")

        py_files = sorted(category_dir.glob("*.py"))
        for py_file in py_files:
            # Skip __init__.py and abstract files
            if py_file.name.startswith("__") or "abstract" in py_file.name.lower():
                continue

            total_files += 1
            if replace_doctests_in_file(py_file):
                modified_files += 1

        print()

    print(f"Summary: Modified {modified_files} out of {total_files} files")

    if modified_files < total_files:
        print(f"WARNING: {total_files - modified_files} files not modified")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
