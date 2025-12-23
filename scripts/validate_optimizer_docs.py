"""Validate optimizer docstrings for COCO/BBOB compliance.

This script validates that optimizer docstrings follow the COCO/BBOB template
defined in `.github/prompts/optimizer-docs-template.md`. It checks for:
- Algorithm Metadata block present
- BBOB Benchmark Settings section present
- Example includes seed=42
- Args documents seed parameter
- Attributes includes self.seed
- Notes includes BBOB Performance Characteristics
- References include DOI links
- Function evaluation complexity documented

Usage:
    python scripts/validate_optimizer_docs.py <file1.py> <file2.py> ...
    python scripts/validate_optimizer_docs.py opt/swarm_intelligence/particle_swarm.py
"""

from __future__ import annotations

import argparse
import ast
import re

from pathlib import Path


# Required sections in optimizer docstrings
REQUIRED_SECTIONS = [
    "Algorithm Metadata:",
    "COCO/BBOB Benchmark Settings:",
    "Example:",
    "Args:",
    "Attributes:",
    "Notes:",
    "References:",
]

# Optional but recommended sections
RECOMMENDED_SECTIONS = [
    "Mathematical Formulation:",
    "Hyperparameters:",
    "Methods:",
    "See Also:",
]


def _extract_class_docstring(file_path: Path) -> str | None:
    """Extract the first class docstring from a Python file.

    Args:
        file_path (Path): Path to the Python file.

    Returns:
        str | None: The docstring content, or None if no class found.
    """
    try:
        with file_path.open(encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if class inherits from AbstractOptimizer or similar
                docstring = ast.get_docstring(node)
                if docstring:
                    return docstring
    except SyntaxError:
        return None
    return None


def _check_required_sections(docstring: str, file_path: Path) -> list[str]:
    """Check if all required sections are present in docstring.

    Args:
        docstring (str): The docstring to check.
        file_path (Path): File path for error reporting.

    Returns:
        list[str]: List of error messages.
    """
    errors = []

    for section in REQUIRED_SECTIONS:
        if section not in docstring:
            errors.append(
                f"{file_path}: Missing required section '{section}' in class docstring"
            )

    return errors


def _check_algorithm_metadata(docstring: str, file_path: Path) -> list[str]:
    """Check if Algorithm Metadata table is properly formatted.

    Args:
        docstring (str): The docstring to check.
        file_path (Path): File path for error reporting.

    Returns:
        list[str]: List of error messages.
    """
    errors = []

    if "Algorithm Metadata:" not in docstring:
        return errors

    # Check for table headers in metadata
    required_metadata_fields = [
        "Algorithm Name",
        "Acronym",
        "Year Introduced",
        "Authors",
        "Algorithm Class",
        "Complexity",
        "Implementation",
        "COCO Compatible",
    ]

    for field in required_metadata_fields:
        if field not in docstring:
            errors.append(f"{file_path}: Algorithm Metadata missing field '{field}'")

    return errors


def _check_bbob_settings(docstring: str, file_path: Path) -> list[str]:
    """Check if BBOB Benchmark Settings section has required content.

    Args:
        docstring (str): The docstring to check.
        file_path (Path): File path for error reporting.

    Returns:
        list[str]: List of error messages.
    """
    errors = []

    if "COCO/BBOB Benchmark Settings:" not in docstring:
        return errors

    # Extract the BBOB section
    match = re.search(
        r"COCO/BBOB Benchmark Settings:(.*?)(?=\n\s{4}[A-Z][a-z]|\Z)",
        docstring,
        re.DOTALL,
    )

    if not match:
        return errors

    bbob_section = match.group(1)

    # Check for required subsections
    required_subsections = ["Search Space", "Evaluation Budget", "Performance Metrics"]

    for subsection in required_subsections:
        if subsection not in bbob_section:
            errors.append(
                f"{file_path}: BBOB Benchmark Settings missing '{subsection}' subsection"
            )

    # Check for standard dimensions
    if (
        "2, 3, 5, 10, 20, 40" not in bbob_section
        and "Dimensions tested:" in bbob_section
    ):
        errors.append(
            f"{file_path}: BBOB Benchmark Settings should specify standard dimensions (2, 3, 5, 10, 20, 40)"
        )

    return errors


def _check_example_seed(docstring: str, file_path: Path) -> list[str]:
    """Check if Example section includes seed=42.

    Args:
        docstring (str): The docstring to check.
        file_path (Path): File path for error reporting.

    Returns:
        list[str]: List of error messages.
    """
    errors = []

    if "Example:" not in docstring:
        return errors

    # Extract Example section
    match = re.search(r"Example:(.*?)(?=\n\s{4}[A-Z][a-z]|\Z)", docstring, re.DOTALL)

    if not match:
        return errors

    example_section = match.group(1)

    # Check for seed=42 in examples
    if "seed=42" not in example_section:
        errors.append(
            f"{file_path}: Example section should include 'seed=42' for reproducibility"
        )

    return errors


def _check_args_seed(docstring: str, file_path: Path) -> list[str]:
    """Check if Args section documents seed parameter.

    Args:
        docstring (str): The docstring to check.
        file_path (Path): File path for error reporting.

    Returns:
        list[str]: List of error messages.
    """
    errors = []

    if "Args:" not in docstring:
        return errors

    # Extract Args section
    match = re.search(r"Args:(.*?)(?=\n\s{4}[A-Z][a-z]|\Z)", docstring, re.DOTALL)

    if not match:
        return errors

    args_section = match.group(1)

    # Check for seed parameter documentation
    seed_pattern = r"seed\s*\([^)]*\):"
    if not re.search(seed_pattern, args_section, re.IGNORECASE):
        errors.append(
            f"{file_path}: Args section should document 'seed' parameter for BBOB compliance"
        )

    return errors


def _check_attributes_seed(docstring: str, file_path: Path) -> list[str]:
    """Check if Attributes section includes self.seed.

    Args:
        docstring (str): The docstring to check.
        file_path (Path): File path for error reporting.

    Returns:
        list[str]: List of error messages.
    """
    errors = []

    if "Attributes:" not in docstring:
        return errors

    # Extract Attributes section
    match = re.search(r"Attributes:(.*?)(?=\n\s{4}[A-Z][a-z]|\Z)", docstring, re.DOTALL)

    if not match:
        return errors

    attributes_section = match.group(1)

    # Check for seed attribute
    seed_pattern = r"seed\s*\([^)]*\):"
    if not re.search(seed_pattern, attributes_section, re.IGNORECASE):
        errors.append(
            f"{file_path}: Attributes section should document 'seed' attribute (REQUIRED for BBOB)"
        )

    return errors


def _check_notes_bbob_performance(docstring: str, file_path: Path) -> list[str]:
    """Check if Notes section includes BBOB Performance Characteristics.

    Args:
        docstring (str): The docstring to check.
        file_path (Path): File path for error reporting.

    Returns:
        list[str]: List of error messages.
    """
    errors = []

    if "Notes:" not in docstring:
        return errors

    # Extract Notes section
    match = re.search(r"Notes:(.*?)(?=\n\s{4}[A-Z][a-z]|\Z)", docstring, re.DOTALL)

    if not match:
        return errors

    notes_section = match.group(1)

    # Check for BBOB Performance Characteristics
    if "BBOB Performance Characteristics" not in notes_section:
        errors.append(
            f"{file_path}: Notes section should include 'BBOB Performance Characteristics' subsection"
        )

    # Check for Computational Complexity
    if "Computational Complexity" not in notes_section:
        errors.append(
            f"{file_path}: Notes section should include 'Computational Complexity' subsection"
        )

    # Check for Reproducibility
    if "Reproducibility" not in notes_section:
        errors.append(
            f"{file_path}: Notes section should include 'Reproducibility' subsection"
        )

    return errors


def _check_references_doi(docstring: str, file_path: Path) -> list[str]:
    """Check if References section includes DOI links.

    Args:
        docstring (str): The docstring to check.
        file_path (Path): File path for error reporting.

    Returns:
        list[str]: List of error messages.
    """
    errors = []

    if "References:" not in docstring:
        return errors

    # Extract References section
    match = re.search(r"References:(.*?)(?=\n\s{4}[A-Z][a-z]|\Z)", docstring, re.DOTALL)

    if not match:
        return errors

    references_section = match.group(1)

    # Check for DOI links
    doi_pattern = r"https?://doi\.org/[^\s]+"
    if not re.search(doi_pattern, references_section):
        errors.append(
            f"{file_path}: References section should include at least one DOI link (https://doi.org/...)"
        )

    # Check for COCO reference
    if (
        "COCO" not in references_section
        and "coco-platform.org" not in references_section
    ):
        errors.append(
            f"{file_path}: References section should include COCO/BBOB platform reference"
        )

    return errors


def _validate_file(file_path: Path) -> list[str]:
    """Validate a single optimizer file.

    Args:
        file_path (Path): Path to the file to validate.

    Returns:
        list[str]: List of error messages.
    """
    errors = []

    # Skip abstract base classes and test files
    if "abstract" in file_path.name.lower() or file_path.name.startswith("test_"):
        return errors

    # Skip __init__.py files
    if file_path.name == "__init__.py":
        return errors

    docstring = _extract_class_docstring(file_path)

    if not docstring:
        # Not all files need to be optimizers (e.g., utility modules)
        return errors

    # Only validate if it appears to be an optimizer class
    # (has certain keywords or inherits from AbstractOptimizer)
    if not any(
        keyword in docstring.lower()
        for keyword in ["optimizer", "optimization", "algorithm", "search"]
    ):
        return errors

    # Run all validation checks
    errors.extend(_check_required_sections(docstring, file_path))
    errors.extend(_check_algorithm_metadata(docstring, file_path))
    errors.extend(_check_bbob_settings(docstring, file_path))
    errors.extend(_check_example_seed(docstring, file_path))
    errors.extend(_check_args_seed(docstring, file_path))
    errors.extend(_check_attributes_seed(docstring, file_path))
    errors.extend(_check_notes_bbob_performance(docstring, file_path))
    errors.extend(_check_references_doi(docstring, file_path))

    return errors


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the validator.

    Args:
        argv (list[str] | None): Command line arguments.

    Returns:
        int: Exit code (0 for success, 1 for errors).
    """
    parser = argparse.ArgumentParser(
        description="Validate optimizer docstrings for COCO/BBOB compliance"
    )
    parser.add_argument("files", nargs="*", help="Files to validate")
    args = parser.parse_args(argv)

    if not args.files:
        print("No files provided")
        return 0

    all_errors: list[str] = []

    for file_str in args.files:
        file_path = Path(file_str)

        if not file_path.exists():
            continue

        if file_path.suffix != ".py":
            continue

        errors = _validate_file(file_path)
        all_errors.extend(errors)

    if all_errors:
        print("\n".join(all_errors))
        print(f"\nFound {len(all_errors)} COCO/BBOB compliance issue(s)")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
