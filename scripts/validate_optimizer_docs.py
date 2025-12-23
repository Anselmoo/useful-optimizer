"""Validate optimizer docstrings for COCO/BBOB compliance.

This script validates that optimizer docstrings follow the COCO/BBOB template
defined in `.github/prompts/optimizer-docs-template.prompt.md`. It checks for:
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

# Recognized section headers for parsing boundaries
SECTION_HEADERS = [
    "Algorithm Metadata",
    "Mathematical Formulation",
    "Hyperparameters",
    "COCO/BBOB Benchmark Settings",
    "Example",
    "Args",
    "Attributes",
    "Methods",
    "References",
    "See Also",
    "Notes",
]
SECTION_HEADERS_PATTERN = "|".join(re.escape(header) for header in SECTION_HEADERS)

# Regex pattern for section boundaries (next known section header at line start)
SECTION_BOUNDARY_PATTERN = rf"(?=\n(?:{SECTION_HEADERS_PATTERN}):|\Z)"


def _extract_section(docstring: str, header: str) -> str | None:
    """Extract the content of a specific section, using the last occurrence.

    Args:
        docstring: Docstring text to parse.
        header: Header name without trailing colon (e.g., 'Example').

    Returns:
        Section content or None if not found.
    """
    pattern = rf"{re.escape(header)}:(.*?){SECTION_BOUNDARY_PATTERN}"
    matches = list(re.finditer(pattern, docstring, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1)


# Keywords that identify optimizer classes
OPTIMIZER_KEYWORDS = ["optimizer", "optimization", "algorithm", "search"]

# Standard BBOB dimensions for testing
STANDARD_BBOB_DIMENSIONS = "2, 3, 5, 10, 20, 40"

# Regex patterns for validation
SEED_PARAMETER_PATTERN = r"seed\s*\([^)]*\):"
DOI_LINK_PATTERN = r"https?://doi\.org/[^\s]+"


def _extract_class_docstring(file_path: Path) -> tuple[str | None, str | None]:
    """Extract the first class docstring and its literal from a Python file.

    Args:
        file_path (Path): Path to the Python file.

    Returns:
        tuple[str | None, str | None]: The cleaned docstring content and the raw
            literal (including prefixes), or (None, None) if no class was found.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))

        optimizer_candidates: list[tuple[ast.ClassDef, str | None]] = []
        fallback_candidates: list[tuple[ast.ClassDef, str | None]] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            docstring = ast.get_docstring(node)
            if not docstring:
                continue

            base_names: list[str] = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_names.append(base.id)
                elif isinstance(base, ast.Attribute):
                    base_names.append(base.attr)

            # Attempt to capture the literal representation to check for raw strings.
            literal = None
            if node.body and isinstance(node.body[0], ast.Expr):
                literal = ast.get_source_segment(source, node.body[0].value)

            if (
                "AbstractOptimizer" in base_names
                or "AbstractMultiObjectiveOptimizer" in base_names
            ):
                optimizer_candidates.append((node, literal))
            else:
                fallback_candidates.append((node, literal))

        if optimizer_candidates:
            docstring = ast.get_docstring(optimizer_candidates[0][0])
            return docstring, optimizer_candidates[0][1]

        if fallback_candidates:
            docstring = ast.get_docstring(fallback_candidates[0][0])
            return docstring, fallback_candidates[0][1]
    except SyntaxError:
        return None, None
    return None, None


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
        rf"COCO/BBOB Benchmark Settings:(.*?){SECTION_BOUNDARY_PATTERN}",
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
        STANDARD_BBOB_DIMENSIONS not in bbob_section
        and "Dimensions tested:" in bbob_section
    ):
        errors.append(
            f"{file_path}: BBOB Benchmark Settings should specify standard dimensions ({STANDARD_BBOB_DIMENSIONS})"
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

    example_section = _extract_section(docstring, "Example")

    if not example_section:
        return errors

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

    args_section = _extract_section(docstring, "Args")

    if not args_section:
        return errors

    if not re.search(SEED_PARAMETER_PATTERN, args_section, re.IGNORECASE):
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

    attributes_section = _extract_section(docstring, "Attributes")

    if not attributes_section:
        return errors

    if not re.search(SEED_PARAMETER_PATTERN, attributes_section, re.IGNORECASE):
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

    notes_section = _extract_section(docstring, "Notes")

    if not notes_section:
        return errors

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

    references_section = _extract_section(docstring, "References")

    if not references_section:
        return errors

    if not re.search(DOI_LINK_PATTERN, references_section):
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

    docstring, raw_literal = _extract_class_docstring(file_path)

    if not docstring:
        # Not all files need to be optimizers (e.g., utility modules)
        return errors

    # Only validate if it appears to be an optimizer class
    # (has certain keywords or inherits from AbstractOptimizer)
    if all(keyword not in docstring.lower() for keyword in OPTIMIZER_KEYWORDS):
        return errors

    # Enforce migration to the COCO/BBOB template for all optimizers.
    if (
        "Algorithm Metadata:" not in docstring
        or "COCO/BBOB Benchmark Settings:" not in docstring
    ):
        errors.append(
            f"{file_path}: Missing COCO/BBOB template sections (Algorithm Metadata and COCO/BBOB Benchmark Settings)"
        )
        return errors

    # Require raw string docstrings when LaTeX, code fences, or escapes are present.
    if raw_literal and (
        ("\\" in docstring) or ("$" in docstring) or ("`" in docstring)
    ):
        stripped_literal = raw_literal.lstrip()
        if not stripped_literal.startswith(('r"""', "r'''")):
            errors.append(
                f'{file_path}: Docstring should use a raw string literal (r"""...""") when containing LaTeX or code fences'
            )

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
