"""Tests for batch_update_docstrings.py script.

This module tests the AST parsing, template generation, and file processing
functionality of the batch docstring update script.
"""

from __future__ import annotations

# Import the script as a module
import sys
import tempfile

from pathlib import Path
from typing import TYPE_CHECKING

import pytest


sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from batch_update_docstrings import OPTIMIZER_CATEGORIES
from batch_update_docstrings import OptimizerInfo
from batch_update_docstrings import extract_optimizer_info
from batch_update_docstrings import find_optimizer_files
from batch_update_docstrings import generate_bbob_docstring_template


if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def temp_optimizer_file() -> Generator[Path, None, None]:
    """Create a temporary optimizer file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(
            '''"""Test optimizer module."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from opt.abstract import AbstractOptimizer

if TYPE_CHECKING:
    from collections.abc import Callable
    from numpy import ndarray


class TestOptimizer(AbstractOptimizer):
    """Test optimizer for unit tests."""

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        population_size: int = 100,
        max_iter: int = 1000,
        learning_rate: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """Initialize the TestOptimizer."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        self.learning_rate = learning_rate

    def search(self) -> tuple[np.ndarray, float]:
        """Execute optimization."""
        return np.zeros(self.dim), 0.0
'''
        )
        tmp_path = Path(tmp.name)

    yield tmp_path

    # Cleanup
    tmp_path.unlink()


@pytest.fixture
def temp_multi_objective_file() -> Generator[Path, None, None]:
    """Create a temporary multi-objective optimizer file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(
            '''"""Test multi-objective optimizer module."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from opt.multi_objective.abstract_multi_objective import AbstractMultiObjectiveOptimizer

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from numpy import ndarray


class TestMultiObjectiveOptimizer(AbstractMultiObjectiveOptimizer):
    """Test multi-objective optimizer for unit tests."""

    def __init__(
        self,
        objectives: Sequence[Callable[[ndarray], float]],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = 1000,
        seed: int | None = None,
        population_size: int = 100,
    ) -> None:
        """Initialize the TestMultiObjectiveOptimizer."""
        super().__init__(
            objectives=objectives,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )

    def search(self) -> tuple[np.ndarray, np.ndarray]:
        """Execute optimization."""
        return np.zeros((10, self.dim)), np.zeros((10, self.num_objectives))
'''
        )
        tmp_path = Path(tmp.name)

    yield tmp_path

    # Cleanup
    tmp_path.unlink()


def test_optimizer_categories() -> None:
    """Test that all expected optimizer categories are defined."""
    assert len(OPTIMIZER_CATEGORIES) == 10
    assert "swarm_intelligence" in OPTIMIZER_CATEGORIES
    assert "gradient_based" in OPTIMIZER_CATEGORIES
    assert "classical" in OPTIMIZER_CATEGORIES
    assert "metaheuristic" in OPTIMIZER_CATEGORIES
    assert "multi_objective" in OPTIMIZER_CATEGORIES


def test_extract_optimizer_info_basic(temp_optimizer_file: Path) -> None:
    """Test basic extraction of optimizer information."""
    # Create a fake category structure
    category_dir = temp_optimizer_file.parent / "test_category"
    category_dir.mkdir(exist_ok=True)
    test_file = category_dir / "test_optimizer.py"
    test_file.write_text(temp_optimizer_file.read_text())

    try:
        info = extract_optimizer_info(test_file)

        assert info is not None
        assert info.class_name == "TestOptimizer"
        assert info.category == "test_category"
        assert "func" in info.parameters
        assert "lower_bound" in info.parameters
        assert "upper_bound" in info.parameters
        assert "dim" in info.parameters
        assert "learning_rate" in info.parameters
        assert not info.is_multi_objective
    finally:
        # Cleanup
        test_file.unlink()
        category_dir.rmdir()


def test_extract_optimizer_info_multi_objective(
    temp_multi_objective_file: Path,
) -> None:
    """Test extraction of multi-objective optimizer information."""
    # Create a fake category structure
    category_dir = temp_multi_objective_file.parent / "multi_objective"
    category_dir.mkdir(exist_ok=True)
    test_file = category_dir / "test_mo_optimizer.py"
    test_file.write_text(temp_multi_objective_file.read_text())

    try:
        info = extract_optimizer_info(test_file)

        assert info is not None
        assert info.class_name == "TestMultiObjectiveOptimizer"
        assert info.category == "multi_objective"
        assert "objectives" in info.parameters
        assert info.is_multi_objective
    finally:
        # Cleanup
        test_file.unlink()
        category_dir.rmdir()


def test_extract_optimizer_info_invalid_file() -> None:
    """Test extraction from an invalid Python file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write("This is not valid Python code {{{")
        tmp_path = Path(tmp.name)

    try:
        info = extract_optimizer_info(tmp_path)
        assert info is None
    finally:
        tmp_path.unlink()


def test_generate_bbob_docstring_template_single_objective() -> None:
    """Test BBOB template generation for single-objective optimizer."""
    info = OptimizerInfo(
        class_name="TestOptimizer",
        filepath=Path("opt/test_category/test_optimizer.py"),
        category="test_category",
        parameters=["func", "lower_bound", "upper_bound", "dim", "max_iter"],
        existing_docstring="Old docstring",
        is_multi_objective=False,
    )

    template = generate_bbob_docstring_template(info)

    # Check required sections are present
    assert "Algorithm Metadata:" in template
    assert "Mathematical Formulation:" in template
    assert "Hyperparameters:" in template
    assert "COCO/BBOB Benchmark Settings:" in template
    assert "Example:" in template
    assert "Args:" in template
    assert "Attributes:" in template
    assert "Methods:" in template
    assert "References:" in template
    assert "See Also:" in template
    assert "Notes:" in template

    # Check FIXME markers are present
    assert "FIXME:" in template
    assert template.count("FIXME:") >= 10  # Multiple FIXME markers

    # Check category is formatted correctly
    assert "Test Category" in template

    # Check return type for single-objective
    assert "tuple[np.ndarray, float]" in template


def test_generate_bbob_docstring_template_multi_objective() -> None:
    """Test BBOB template generation for multi-objective optimizer."""
    info = OptimizerInfo(
        class_name="TestMOOptimizer",
        filepath=Path("opt/multi_objective/test_mo_optimizer.py"),
        category="multi_objective",
        parameters=["objectives", "lower_bound", "upper_bound", "dim"],
        existing_docstring=None,
        is_multi_objective=True,
    )

    template = generate_bbob_docstring_template(info)

    # Check return type for multi-objective
    assert "tuple[ndarray, ndarray]" in template
    assert "Pareto-optimal solutions" in template


def test_generate_bbob_template_includes_all_sections() -> None:
    """Test that generated template includes all 11 required BBOB sections."""
    info = OptimizerInfo(
        class_name="DemoOptimizer",
        filepath=Path("opt/swarm_intelligence/demo.py"),
        category="swarm_intelligence",
        parameters=["func", "lower_bound", "upper_bound", "dim"],
        existing_docstring=None,
        is_multi_objective=False,
    )

    template = generate_bbob_docstring_template(info)

    # All 11 sections from the BBOB template
    required_sections = [
        "Algorithm Metadata:",
        "Mathematical Formulation:",
        "Hyperparameters:",
        "COCO/BBOB Benchmark Settings:",
        "Example:",
        "Args:",
        "Attributes:",
        "Methods:",
        "References:",
        "See Also:",
        "Notes:",
    ]

    for section in required_sections:
        assert section in template, f"Missing required section: {section}"


def test_find_optimizer_files_real_repo() -> None:
    """Test finding optimizer files in the real repository."""
    # This test requires the actual repository structure
    script_dir = Path(__file__).parent.parent.parent / "scripts"
    opt_dir = script_dir.parent / "opt"

    if not opt_dir.exists():
        pytest.skip("Repository structure not available")

    # Test finding all files
    all_files = find_optimizer_files(opt_dir)
    assert len(all_files) > 0

    # Test that abstract files are excluded
    for filepath in all_files:
        assert not filepath.name.startswith("abstract_")
        assert filepath.name != "__init__.py"

    # Test finding files by category
    swarm_files = find_optimizer_files(opt_dir, category="swarm_intelligence")
    assert len(swarm_files) > 0
    for filepath in swarm_files:
        assert filepath.parent.name == "swarm_intelligence"


def test_template_has_proper_latex_escaping() -> None:
    """Test that LaTeX expressions are properly formatted in template."""
    info = OptimizerInfo(
        class_name="TestOptimizer",
        filepath=Path("opt/test/test.py"),
        category="test",
        parameters=["func"],
        existing_docstring=None,
        is_multi_objective=False,
    )

    template = generate_bbob_docstring_template(info)

    # Check for proper LaTeX delimiters
    assert "$$" in template  # Display math
    assert "$x_t$" in template or "$x_{t+1}$" in template  # Inline math


def test_template_includes_seed_parameter() -> None:
    """Test that template includes required seed parameter for BBOB compliance."""
    info = OptimizerInfo(
        class_name="TestOptimizer",
        filepath=Path("opt/test/test.py"),
        category="test",
        parameters=["func", "seed"],
        existing_docstring=None,
        is_multi_objective=False,
    )

    template = generate_bbob_docstring_template(info)

    # Check seed is documented properly
    assert "seed" in template.lower()
    assert "reproducibility" in template.lower()
    assert "0-14" in template  # BBOB seed range


def test_template_includes_bbob_dimensions() -> None:
    """Test that template includes standard BBOB dimensions."""
    info = OptimizerInfo(
        class_name="TestOptimizer",
        filepath=Path("opt/test/test.py"),
        category="test",
        parameters=["func"],
        existing_docstring=None,
        is_multi_objective=False,
    )

    template = generate_bbob_docstring_template(info)

    # Check BBOB standard dimensions are mentioned
    assert "2, 3, 5, 10, 20, 40" in template


def test_real_optimizer_extraction() -> None:
    """Test extraction on a real optimizer from the repository."""
    # Try to extract from ParticleSwarm
    script_dir = Path(__file__).parent.parent.parent / "scripts"
    opt_dir = script_dir.parent / "opt"
    pso_file = opt_dir / "swarm_intelligence" / "particle_swarm.py"

    if not pso_file.exists():
        pytest.skip("ParticleSwarm file not available")

    info = extract_optimizer_info(pso_file)

    assert info is not None
    assert info.class_name == "ParticleSwarm"
    assert info.category == "swarm_intelligence"
    assert "func" in info.parameters
    assert "c1" in info.parameters  # PSO-specific parameter
    assert "c2" in info.parameters  # PSO-specific parameter
    assert not info.is_multi_objective
