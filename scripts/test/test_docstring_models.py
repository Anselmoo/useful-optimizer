"""Tests for Pydantic docstring models."""

from __future__ import annotations

import pytest

from pydantic import ValidationError

from scripts.docstring_models import AlgorithmClass
from scripts.docstring_models import AlgorithmMetadata
from scripts.docstring_models import Args
from scripts.docstring_models import Attributes
from scripts.docstring_models import COCOBBOBSettings
from scripts.docstring_models import CocoBbobOptimizerDocstringSchema
from scripts.docstring_models import Example
from scripts.docstring_models import Notes
from scripts.docstring_models import Parameter
from scripts.docstring_models import Property
from scripts.docstring_models import References


class TestAlgorithmMetadata:
    """Test AlgorithmMetadata model."""

    def test_valid_metadata(self) -> None:
        """Test valid algorithm metadata."""
        metadata = AlgorithmMetadata(
            algorithm_name="Particle Swarm Optimization",
            acronym="PSO",
            year_introduced=1995,
            authors="Kennedy, James; Eberhart, Russell",
            algorithm_class=AlgorithmClass.swarm_intelligence,
            complexity="O(population_size $\\times$ max_iter $\\times$ dim)",
            properties=[Property.population_based, Property.stochastic],
            implementation="Python 3.10+",
            coco_compatible=True,
        )
        assert metadata.acronym == "PSO"
        assert metadata.year_introduced == 1995
        assert len(metadata.properties) == 2

    def test_invalid_acronym(self) -> None:
        """Test that invalid acronym raises error."""
        with pytest.raises(ValidationError):
            AlgorithmMetadata(
                algorithm_name="Test",
                acronym="invalid_lowercase",  # Should be uppercase
                year_introduced=2000,
                authors="Test Author",
                algorithm_class=AlgorithmClass.classical,
                complexity="O(n)",
                properties=[Property.deterministic],
            )

    def test_invalid_year(self) -> None:
        """Test that invalid year raises error."""
        with pytest.raises(ValidationError):
            AlgorithmMetadata(
                algorithm_name="Test",
                acronym="TEST",
                year_introduced=1800,  # Before 1900
                authors="Test Author",
                algorithm_class=AlgorithmClass.classical,
                complexity="O(n)",
                properties=[Property.deterministic],
            )


class TestParameter:
    """Test Parameter model."""

    def test_valid_parameter(self) -> None:
        """Test valid parameter definition."""
        param = Parameter(
            name="population_size",
            type="int",
            description="Number of particles in the swarm",
            optional=True,
            default=100,
        )
        assert param.name == "population_size"
        assert param.optional is True
        assert param.default == 100

    def test_required_parameter(self) -> None:
        """Test required parameter without default."""
        param = Parameter(
            name="func",
            type="Callable[[ndarray], float]",
            description="Objective function to minimize",
        )
        assert param.optional is False
        assert param.default is None


class TestCOCOBBOBSettings:
    """Test COCOBBOBSettings model."""

    def test_valid_settings(self) -> None:
        """Test valid COCO/BBOB settings."""
        settings = COCOBBOBSettings(
            search_space={
                "dimensions_tested": [2, 3, 5, 10, 20, 40],
            },
        )
        assert settings.search_space is not None


class TestCocoBbobOptimizerDocstringSchema:
    """Test root docstring schema model."""

    def test_minimal_valid_schema(self) -> None:
        """Test minimal valid docstring schema."""
        schema = CocoBbobOptimizerDocstringSchema(
            summary="Test algorithm for optimization",
            algorithm_metadata=AlgorithmMetadata(
                algorithm_name="Test Algorithm",
                acronym="TA",
                year_introduced=2020,
                authors="Test, Author",
                algorithm_class=AlgorithmClass.classical,
                complexity="O(n)",
                properties=[Property.deterministic],
            ),
            coco_bbob_benchmark_settings=COCOBBOBSettings(),
            example=Example(description="Test example"),
            args=Args(parameters=[]),
            attributes=Attributes(attributes=[]),
            notes=Notes(format="simple", simple_notes=["Test note"]),
            references=References(),
        )
        assert schema.summary == "Test algorithm for optimization"
        assert schema.algorithm_metadata.acronym == "TA"

    def test_invalid_schema_missing_required(self) -> None:
        """Test that missing required fields raises error."""
        with pytest.raises(ValidationError):
            CocoBbobOptimizerDocstringSchema(
                summary="Test",
                # Missing algorithm_metadata
                coco_bbob_benchmark_settings=COCOBBOBSettings(),
                example=Example(description="Test"),
                args=Args(parameters=[]),
                attributes=Attributes(attributes=[]),
                notes=Notes(format="simple", simple_notes=[]),
                references=References(),
            )

    def test_summary_length_validation(self) -> None:
        """Test that summary length is validated."""
        with pytest.raises(ValidationError):
            CocoBbobOptimizerDocstringSchema(
                summary="x" * 100,  # Exceeds 80 char limit
                algorithm_metadata=AlgorithmMetadata(
                    algorithm_name="Test",
                    acronym="T",
                    year_introduced=2020,
                    authors="Test",
                    algorithm_class=AlgorithmClass.classical,
                    complexity="O(n)",
                    properties=[Property.deterministic],
                ),
                coco_bbob_benchmark_settings=COCOBBOBSettings(),
                example=Example(description="Test"),
                args=Args(parameters=[]),
                attributes=Attributes(attributes=[]),
                notes=Notes(format="simple", simple_notes=[]),
                references=References(),
            )
