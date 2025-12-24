#!/usr/bin/env python3
"""Unified docstring validator using Pydantic schemas.

This script validates optimizer docstrings against the JSON schema using
Pydantic models. It provides comprehensive validation including:
- Schema compliance
- Inline description format
- COCO/BBOB requirements
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

from pathlib import Path

from pydantic import ValidationError


# Load DocstringParser from sibling docstring_parser.py without modifying sys.path
_DOCSTRING_PARSER_PATH = Path(__file__).parent / "docstring_parser.py"
_DOCSTRING_PARSER_SPEC = importlib.util.spec_from_file_location(
    "scripts.docstring_parser", _DOCSTRING_PARSER_PATH
)
if _DOCSTRING_PARSER_SPEC is None or _DOCSTRING_PARSER_SPEC.loader is None:
    msg = f"Could not load docstring_parser module from {_DOCSTRING_PARSER_PATH}"
    raise ImportError(msg)
_docstring_parser_module = importlib.util.module_from_spec(_DOCSTRING_PARSER_SPEC)
_DOCSTRING_PARSER_SPEC.loader.exec_module(_docstring_parser_module)
DocstringParser = _docstring_parser_module.DocstringParser


def validate_file(file_path: Path, verbose: bool = False) -> tuple[bool, list[str]]:
    """Validate a single Python file's docstring.

    Args:
        file_path: Path to the Python file.
        verbose: If True, print detailed validation info.

    Returns:
        Tuple of (success, error_messages).
    """
    parser = DocstringParser()
    errors = []

    try:
        schema = parser.parse_file(file_path)

        if verbose:
            print(f"✓ {file_path}: Schema validation passed")
            print(f"  Algorithm: {schema.algorithm_metadata.algorithm_name}")
            print(f"  Acronym: {schema.algorithm_metadata.acronym}")
            print(f"  Args: {len(schema.args.parameters)} parameters")
            print(f"  Attributes: {len(schema.attributes.attributes)} attributes")

        return True, []

    except ValidationError as e:
        # Format Pydantic validation errors
        for error in e.errors():
            loc = " → ".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"{file_path}: {loc}: {msg}")

        return False, errors

    except ValueError as e:
        # File has no docstring or parsing failed
        errors.append(f"{file_path}: {e}")
        return False, errors

    except Exception as e:
        # Unexpected error
        errors.append(f"{file_path}: Unexpected error: {e}")
        return False, errors


def main() -> int:
    """CLI entry point for unified validator."""
    parser = argparse.ArgumentParser(
        description="Validate optimizer docstrings using Pydantic schema"
    )
    parser.add_argument("files", nargs="+", type=Path, help="Python files to validate")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print detailed validation info"
    )
    parser.add_argument(
        "--all", action="store_true", help="Validate all optimizer files in opt/"
    )

    args = parser.parse_args()

    # Determine files to validate
    if args.all:
        # Find all optimizer files
        optimizer_dirs = [
            "classical",
            "constrained",
            "evolutionary",
            "gradient_based",
            "metaheuristic",
            "multi_objective",
            "physics_inspired",
            "probabilistic",
            "social_inspired",
            "swarm_intelligence",
        ]
        files = []
        opt_dir = Path("opt")
        for category in optimizer_dirs:
            category_path = opt_dir / category
            if category_path.exists():
                files.extend(category_path.glob("*.py"))
        # Filter out __init__.py
        files = [f for f in files if f.name != "__init__.py"]
    else:
        files = args.files

    if not files:
        print("No files to validate")
        return 1

    # Validate each file
    all_errors = []
    success_count = 0
    failure_count = 0

    for file_path in files:
        if not file_path.exists():
            print(f"✗ {file_path}: File not found")
            failure_count += 1
            continue

        success, errors = validate_file(file_path, args.verbose)

        if success:
            success_count += 1
            if not args.verbose:
                print(f"✓ {file_path}")
        else:
            failure_count += 1
            all_errors.extend(errors)
            if not args.verbose:
                print(f"✗ {file_path}")

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Validation Summary: {success_count} passed, {failure_count} failed")
    print(f"{'=' * 70}")

    # Print all errors
    if all_errors:
        print("\nValidation Errors:")
        for error in all_errors:
            print(f"  {error}")

    return 1 if failure_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
