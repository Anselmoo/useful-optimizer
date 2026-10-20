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


ROOT = Path(__file__).resolve().parents[1]
OPTIMIZER_DIR = ROOT / "src" / "opt"
OPTIMIZER_CATEGORIES = [
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
PLACEHOLDER_REFERENCE = "Available in scientific literature"
# Optimizers that still carry the placeholder citation. The list may only shrink:
# a new placeholder fails validation, and so does a fixed file left on the list.
PLACEHOLDER_BASELINE = Path(__file__).with_name("placeholder_references.txt")


def load_placeholder_baseline() -> set[str]:
    """Return the optimizer file stems allowed to keep a placeholder reference."""
    if not PLACEHOLDER_BASELINE.exists():
        return set()
    return {
        line.strip()
        for line in PLACEHOLDER_BASELINE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def check_placeholder_reference(file_path: Path, baseline: set[str]) -> list[str]:
    """Ratchet on placeholder citations; see PLACEHOLDER_BASELINE."""
    has_placeholder = PLACEHOLDER_REFERENCE in file_path.read_text(encoding="utf-8")
    listed = file_path.stem in baseline
    if has_placeholder and not listed:
        return [
            f"{file_path}: References has no real citation for the original "
            f"publication ('{PLACEHOLDER_REFERENCE}'); add it"
        ]
    if listed and not has_placeholder:
        return [
            f"{file_path}: reference is fixed; remove '{file_path.stem}' from "
            f"{PLACEHOLDER_BASELINE.name}"
        ]
    return []


def check_unique_acronyms(files: list[Path]) -> list[str]:
    """Every optimizer needs its own acronym: it names it in tables and search."""
    parser = DocstringParser()
    seen: dict[str, list[Path]] = {}
    for file_path in files:
        try:
            acronym = parser.parse_file(file_path).algorithm_metadata.acronym
        except Exception:
            continue
        seen.setdefault(acronym, []).append(file_path)
    return [
        f"acronym '{acronym}' is used by {', '.join(p.stem for p in paths)}; "
        "acronyms must be unique"
        for acronym, paths in seen.items()
        if len(paths) > 1
    ]


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

        errors.extend(
            check_placeholder_reference(file_path, load_placeholder_baseline())
        )
        if errors:
            return False, errors
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
    parser.add_argument("files", nargs="*", type=Path, help="Python files to validate")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print detailed validation info"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate every optimizer in src/opt/ and run the cross-file checks",
    )

    args = parser.parse_args()

    # Determine files to validate
    if args.all:
        files = [
            f
            for category in OPTIMIZER_CATEGORIES
            for f in sorted((OPTIMIZER_DIR / category).glob("*.py"))
            if f.name != "__init__.py"
        ]
    elif not args.files:
        print("Error: No files specified. Use --all or provide file paths.")
        return 1
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

    # Cross-file checks only make sense on the complete set
    if args.all:
        cross_errors = check_unique_acronyms(files)
        failure_count += len(cross_errors)
        all_errors.extend(cross_errors)

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
