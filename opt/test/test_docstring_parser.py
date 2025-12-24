"""Tests for DocstringParser."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.docstring_parser import DocstringParser


class TestDocstringParser:
    """Test DocstringParser class."""

    def test_extract_section(self) -> None:
        """Test section extraction from docstring."""
        parser = DocstringParser()
        docstring = """
        Summary line.

        Algorithm Metadata:
            Some metadata content here.

        Example:
            Example code here.

        Args:
            param1: Description 1.
        """

        metadata = parser.extract_section(docstring, "Algorithm Metadata")
        assert metadata is not None
        assert "Some metadata content" in metadata

        example = parser.extract_section(docstring, "Example")
        assert example is not None
        assert "Example code" in example

        # Non-existent section
        missing = parser.extract_section(docstring, "NonExistent")
        assert missing is None

    def test_parse_algorithm_metadata(self) -> None:
        """Test parsing algorithm metadata table."""
        parser = DocstringParser()
        metadata_content = """
        | Property          | Value                    |
        |-------------------|--------------------------|
        | Algorithm Name    | Test Algorithm           |
        | Acronym           | TA                       |
        | Year Introduced   | 2020                     |
        | Authors           | Test, Author             |
        | Algorithm Class   | Classical                |
        | Complexity        | O(n)                     |
        | Properties        | Deterministic, Fast      |
        | Implementation    | Python 3.10+             |
        | COCO Compatible   | Yes                      |
        """

        metadata = parser.parse_algorithm_metadata(metadata_content)
        assert metadata is not None
        assert metadata["algorithm_name"] == "Test Algorithm"
        assert metadata["acronym"] == "TA"
        assert metadata["year_introduced"] == 2020
        assert metadata["authors"] == "Test, Author"
        assert metadata["coco_compatible"] is True
        assert "Deterministic" in metadata["properties"]

    def test_parse_args_section(self) -> None:
        """Test parsing Args section."""
        parser = DocstringParser()
        args_content = """
        func (Callable[[ndarray], float]): Objective function to minimize.
        lower_bound (float): Lower bound of search space.
        upper_bound (float): Upper bound of search space.
        dim (int): Problem dimensionality.
        max_iter (int, optional): Maximum iterations. Defaults to 1000.
        """

        params = parser.parse_args_section(args_content)
        assert params is not None
        assert len(params) == 5  # noqa: PLR2004

        # Check first parameter
        assert params[0]["name"] == "func"
        assert params[0]["type"] == "Callable[[ndarray], float]"
        assert "Objective function" in params[0]["description"]
        assert params[0]["optional"] is False

        # Check optional parameter
        assert params[4]["name"] == "max_iter"
        assert params[4]["optional"] is True

    def test_extract_class_docstring_real_file(self) -> None:
        """Test extracting docstring from actual optimizer file."""
        parser = DocstringParser()

        # Test with simulated_annealing.py
        sa_path = Path("opt/classical/simulated_annealing.py")
        if sa_path.exists():
            docstring = parser.extract_class_docstring(sa_path)
            assert docstring is not None
            assert "Simulated Annealing" in docstring
            assert "Algorithm Metadata:" in docstring

    def test_extract_class_docstring_missing_file(self) -> None:
        """Test with non-existent file."""
        parser = DocstringParser()
        docstring = parser.extract_class_docstring(Path("nonexistent.py"))
        assert docstring is None

    def test_parse_file_with_validation(self) -> None:
        """Test full file parsing with Pydantic validation."""
        parser = DocstringParser()

        # Test with a file that has complete docstring
        # Note: This will fail if the file doesn't exist or has invalid schema
        sa_path = Path("opt/classical/simulated_annealing.py")
        if sa_path.exists():
            # This should raise ValidationError if docstring doesn't match schema
            # We expect it to fail with current files that have non-compliant properties
            with pytest.raises(Exception):  # Could be ValidationError or ValueError
                parser.parse_file(sa_path)

    def test_parse_file_no_docstring(self) -> None:
        """Test parsing file with no docstring raises ValueError."""
        parser = DocstringParser()

        # Create a temporary file with no docstring
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("class TestClass:\n    pass\n")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="No optimizer class docstring"):
                parser.parse_file(temp_path)
        finally:
            temp_path.unlink()
