"""Parse Python docstrings into Pydantic models for validation.

This module provides a DocstringParser class that extracts and parses
optimizer docstrings from Python files and validates them against the
Pydantic models defined in opt.docstring_models.
"""

from __future__ import annotations

import ast
import contextlib
import re

from pathlib import Path

from opt.docstring_models import CocoBbobOptimizerDocstringSchema


# Section headers that define boundaries in docstrings
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


class DocstringParser:
    """Parse Python docstrings into validated Pydantic models.

    This class extracts class docstrings from Python files and converts them
    into structured Pydantic models that can be validated against the schema.
    """

    def __init__(self) -> None:
        """Initialize the docstring parser."""
        self.section_headers_pattern = "|".join(
            re.escape(header) for header in SECTION_HEADERS
        )
        self.section_boundary_pattern = (
            rf"(?=\n(?:{self.section_headers_pattern}):|\Z)"
        )

    def extract_class_docstring(self, file_path: Path) -> str | None:
        """Extract the first optimizer class docstring from a Python file.

        Args:
            file_path: Path to the Python file.

        Returns:
            The docstring content, or None if no optimizer class was found.
        """
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))

            # Look for classes inheriting from AbstractOptimizer
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue

                docstring = ast.get_docstring(node)
                if not docstring:
                    continue

                # Check if this class inherits from AbstractOptimizer
                base_names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        base_names.append(base.attr)

                if "AbstractOptimizer" in base_names or "AbstractMultiObjectiveOptimizer" in base_names:
                    return docstring

            # Fallback: return first class with docstring
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        return docstring

        except SyntaxError:
            return None

        return None

    def extract_section(self, docstring: str, header: str) -> str | None:
        """Extract the content of a specific section from docstring.

        Args:
            docstring: The full docstring text.
            header: Section header name without trailing colon (e.g., 'Example').

        Returns:
            Section content or None if not found.
        """
        pattern = rf"{re.escape(header)}:(.*?){self.section_boundary_pattern}"
        matches = list(re.finditer(pattern, docstring, re.DOTALL))
        if not matches:
            return None
        return matches[-1].group(1).strip()

    def parse_algorithm_metadata(self, content: str) -> dict | None:
        """Parse Algorithm Metadata table section.

        Args:
            content: The metadata section content.

        Returns:
            Dictionary with metadata fields or None if parsing fails.
        """
        if not content:
            return None

        metadata = {}
        lines = content.split("\n")

        # Parse table rows (skip header and separator)
        for line in lines:
            if "|" not in line or line.strip().startswith("|--"):
                continue

            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 2:
                key = parts[0].strip()
                value = parts[1].strip()

                # Map keys to schema fields
                if key == "Algorithm Name":
                    metadata["algorithm_name"] = value
                elif key == "Acronym":
                    metadata["acronym"] = value
                elif key == "Year Introduced":
                    with contextlib.suppress(ValueError):
                        metadata["year_introduced"] = int(value)
                elif key == "Authors":
                    metadata["authors"] = value
                elif key == "Algorithm Class":
                    metadata["algorithm_class"] = value
                elif key == "Complexity":
                    metadata["complexity"] = value
                elif key == "Properties":
                    # Split comma-separated properties
                    props = [p.strip() for p in value.split(",")]
                    metadata["properties"] = props
                elif key == "Implementation":
                    metadata["implementation"] = value
                elif key == "COCO Compatible":
                    metadata["coco_compatible"] = value.lower() in ("yes", "true")

        return metadata if metadata else None

    def parse_args_section(self, content: str) -> list[dict] | None:
        """Parse Args section into list of parameter dictionaries.

        Args:
            content: The Args section content.

        Returns:
            List of parameter dictionaries or None if parsing fails.
        """
        if not content:
            return None

        parameters = []
        lines = content.split("\n")

        # Pattern for parameter definition: name (type): description
        param_pattern = re.compile(r"^\s*(\w+)\s*\(([^)]+)\):\s*(.+)")

        current_param = None

        for line in lines:
            match = param_pattern.match(line)
            if match:
                # Save previous parameter
                if current_param:
                    parameters.append(current_param)

                name, type_str, desc = match.groups()
                current_param = {
                    "name": name,
                    "type": type_str.strip(),
                    "description": desc.strip(),
                    "optional": "optional" in type_str.lower(),
                }
            elif current_param and line.strip():
                # Continuation of description
                current_param["description"] += " " + line.strip()

        # Add last parameter
        if current_param:
            parameters.append(current_param)

        return parameters if parameters else None

    def parse_file(self, file_path: Path) -> CocoBbobOptimizerDocstringSchema:
        """Extract and validate class docstring from file.

        Args:
            file_path: Path to the Python file to parse.

        Returns:
            Validated Pydantic model of the docstring.

        Raises:
            ValidationError: If the docstring doesn't match the schema.
            ValueError: If no docstring is found.
        """
        docstring = self.extract_class_docstring(file_path)
        if not docstring:
            msg = f"No optimizer class docstring found in {file_path}"
            raise ValueError(msg)

        # Extract sections
        metadata_content = self.extract_section(docstring, "Algorithm Metadata")
        example_content = self.extract_section(docstring, "Example")
        args_content = self.extract_section(docstring, "Args")
        attributes_content = self.extract_section(docstring, "Attributes")
        notes_content = self.extract_section(docstring, "Notes")

        # Build dictionary for Pydantic validation
        parsed_dict = {}

        # Extract summary (first line)
        summary_match = re.match(r"^(.+?)(?:\n|$)", docstring)
        if summary_match:
            parsed_dict["summary"] = summary_match.group(1).strip()

        # Parse Algorithm Metadata
        if metadata_content:
            metadata = self.parse_algorithm_metadata(metadata_content)
            if metadata:
                parsed_dict["algorithm_metadata"] = metadata

        # Basic COCO/BBOB settings (minimal for now)
        parsed_dict["coco_bbob_benchmark_settings"] = {
            "search_space": {"dimensions_tested": [2, 3, 5, 10, 20, 40]},
        }

        # Example section
        if example_content:
            parsed_dict["example"] = {
                "description": "Basic usage example",
                "code": example_content,
                "requires_seed": "seed=" in example_content,
            }
        else:
            parsed_dict["example"] = {"description": "No example provided"}

        # Args section
        if args_content:
            params = self.parse_args_section(args_content)
            if params:
                parsed_dict["args"] = {"parameters": params}
        else:
            parsed_dict["args"] = {"parameters": []}

        # Attributes section
        if attributes_content:
            attrs = self.parse_args_section(attributes_content)
            if attrs:
                parsed_dict["attributes"] = {"attributes": attrs}
        else:
            parsed_dict["attributes"] = {"attributes": []}

        # Notes section
        if notes_content:
            parsed_dict["notes"] = {
                "format": "simple",
                "simple_notes": [notes_content[:100]],  # Simplified
            }
        else:
            parsed_dict["notes"] = {
                "format": "simple",
                "simple_notes": ["No notes provided"],
            }

        # References section (minimal)
        parsed_dict["references"] = {"citations": []}

        # Validate with Pydantic
        return CocoBbobOptimizerDocstringSchema.model_validate(parsed_dict)


def main() -> None:
    """CLI entry point for testing the parser."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python docstring_parser.py <path_to_python_file>")
        sys.exit(1)

    parser = DocstringParser()
    file_path = Path(sys.argv[1])

    try:
        schema = parser.parse_file(file_path)
        print(f"✓ Successfully parsed and validated {file_path}")
        print(f"  Algorithm: {schema.algorithm_metadata.algorithm_name}")
        print(f"  Acronym: {schema.algorithm_metadata.acronym}")
        print(f"  Args count: {len(schema.args.parameters)}")
    except Exception as e:
        print(f"✗ Error parsing {file_path}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
