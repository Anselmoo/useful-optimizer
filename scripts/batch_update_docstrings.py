#!/usr/bin/env python3
"""Batch docstring update script for COCO/BBOB compliance.

This script uses AST parsing to process all optimizer files and generate
COCO/BBOB-compliant docstring templates with FIXME markers for human review.

Workflow:
    ```mermaid
    flowchart TD
        A[Start: Parse CLI Args] --> B{Dry Run Mode?}
        B -->|Yes| C[Preview Mode]
        B -->|No| D[Write Mode]

        C --> E[Find Optimizer Files]
        D --> E

        E --> F[Filter by Category?]
        F -->|Yes| G[Select Category Files]
        F -->|No| H[All 10 Categories]

        G --> I[Process Each File]
        H --> I

        I --> J[AST Parse File]
        J --> K{Valid Optimizer Class?}

        K -->|No| L[Skip File]
        K -->|Yes| M[Extract Class Info]

        M --> N[Detect Parameters]
        N --> O[Generate BBOB Template]

        O --> P{Dry Run?}
        P -->|Yes| Q[Print Preview]
        P -->|No| R[Write to File]

        Q --> S{More Files?}
        R --> S
        L --> S

        S -->|Yes| I
        S -->|No| T[Print Summary]
        T --> U[End]
    ```

Features:
- AST parsing to extract class names, parameters, existing docstrings
- Template generation with FIXME markers
- Category auto-detection from directory structure
- Skips abstract base classes (abstract_*.py files)
- Dry-run mode to preview changes before applying

Usage:
    python scripts/batch_update_docstrings.py [--dry-run] [--category CATEGORY]

Examples:
    # Dry run to preview all changes
    python scripts/batch_update_docstrings.py --dry-run

    # Process only swarm_intelligence category
    python scripts/batch_update_docstrings.py --category swarm_intelligence

    # Process all optimizers and apply changes
    python scripts/batch_update_docstrings.py
"""

from __future__ import annotations

import argparse
import ast
import sys

from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence


# Optimizer categories to process
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


class OptimizerInfo:
    """Container for extracted optimizer information."""

    def __init__(
        self,
        class_name: str,
        filepath: Path,
        category: str,
        parameters: list[str],
        existing_docstring: str | None,
        *,
        is_multi_objective: bool,
    ) -> None:
        """Initialize OptimizerInfo."""
        self.class_name = class_name
        self.filepath = filepath
        self.category = category
        self.parameters = parameters
        self.existing_docstring = existing_docstring
        self.is_multi_objective = is_multi_objective


def extract_optimizer_info(filepath: Path) -> OptimizerInfo | None:
    """Extract optimizer information from a Python file using AST parsing.

    Args:
        filepath: Path to the optimizer Python file.

    Returns:
        OptimizerInfo object if a valid optimizer class is found, None otherwise.
    """
    try:
        with filepath.open("r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)

        # Process only top-level classes in definition order for deterministic behavior
        optimizer_candidates: list[tuple[ast.ClassDef, bool]] = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue

            # Check if it inherits from AbstractOptimizer or AbstractMultiObjectiveOptimizer
            base_names: list[str] = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_names.append(base.id)
                elif isinstance(base, ast.Attribute):
                    base_names.append(base.attr)

            is_optimizer = "AbstractOptimizer" in base_names
            is_multi_objective = "AbstractMultiObjectiveOptimizer" in base_names

            if not (is_optimizer or is_multi_objective):
                continue

            optimizer_candidates.append((node, is_multi_objective))

        if not optimizer_candidates:
            # No optimizer-like classes found in this file
            return None

        if len(optimizer_candidates) > 1:
            class_names = ", ".join(cls.name for cls, _ in optimizer_candidates)
            print(
                f"⚠️  Multiple optimizer classes found in {filepath}: {class_names}. "
                "Using the first one in definition order.",
                file=sys.stderr,
            )

        node, is_multi_objective = optimizer_candidates[0]
        class_name = node.name
        existing_docstring = ast.get_docstring(node)

        # Extract __init__ parameters
        parameters: list[str] = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                for arg in item.args.args:
                    if arg.arg != "self":
                        parameters.append(arg.arg)
                break

        # Detect category from filepath
        category = filepath.parent.name

        return OptimizerInfo(
            class_name=class_name,
            filepath=filepath,
            category=category,
            parameters=parameters,
            existing_docstring=existing_docstring,
            is_multi_objective=is_multi_objective,
        )

    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"⚠️  Error parsing {filepath}: {e}", file=sys.stderr)

    return None


def generate_bbob_docstring_template(info: OptimizerInfo) -> str:
    """Generate COCO/BBOB-compliant docstring template with FIXME markers.

    Args:
        info: OptimizerInfo object containing extracted class information.

    Returns:
        Generated docstring template as a string.
    """
    # Determine return type based on optimizer type
    if info.is_multi_objective:
        return_type = "tuple[ndarray, ndarray]"
        return_desc = "Pareto-optimal solutions and their fitness values"
    else:
        return_type = "tuple[np.ndarray, float]"
        return_desc = "Best solution found and its fitness value"

    template = f'''r"""FIXME: [Algorithm Full Name] ([ACRONYM]) optimization algorithm.

    Algorithm Metadata:
        | Property          | Value                                    |
        |-------------------|------------------------------------------|
        | Algorithm Name    | FIXME: [Full algorithm name]             |
        | Acronym           | FIXME: [SHORT]                           |
        | Year Introduced   | FIXME: [YYYY]                            |
        | Authors           | FIXME: [Last, First; ...]                |
        | Algorithm Class   | {info.category.replace("_", " ").title()} |
        | Complexity        | FIXME: O([expression])                   |
        | Properties        | FIXME: [Population-based, ...]           |
        | Implementation    | Python 3.10+                             |
        | COCO Compatible   | Yes                                      |

    Mathematical Formulation:
        FIXME: Core update equation:

            $$
            x_{{t+1}} = x_t + v_t
            $$

        where:
            - $x_t$ is the position at iteration $t$
            - $v_t$ is the velocity/step at iteration $t$
            - FIXME: Additional variable definitions...

        Constraint handling:
            - **Boundary conditions**: FIXME: [clamping/reflection/periodic]
            - **Feasibility enforcement**: FIXME: [description]

    Hyperparameters:
        | Parameter              | Default | BBOB Recommended | Description                    |
        |------------------------|---------|------------------|--------------------------------|
        | population_size        | 100     | 10*dim           | Number of individuals          |
        | max_iter               | 1000    | 10000            | Maximum iterations             |
        | FIXME: [param_name]    | [val]   | [bbob_val]       | [description]                  |

        **Sensitivity Analysis**:
            - FIXME: `[param_name]`: **[High/Medium/Low]** impact on convergence
            - Recommended tuning ranges: FIXME: $\\text{{[param]}} \\in [\\text{{min}}, \\text{{max}}]$

    COCO/BBOB Benchmark Settings:
        **Search Space**:
            - Dimensions tested: `2, 3, 5, 10, 20, 40`
            - Bounds: Function-specific (typically `[-5, 5]` or `[-100, 100]`)
            - Instances: **15** per function (BBOB standard)

        **Evaluation Budget**:
            - Budget: $\\text{{dim}} \\times 10000$ function evaluations
            - Independent runs: **15** (for statistical significance)
            - Seeds: `0-14` (reproducibility requirement)

        **Performance Metrics**:
            - Target precision: `1e-8` (BBOB default)
            - Success rate at precision thresholds: `[1e-8, 1e-6, 1e-4, 1e-2]`
            - Expected Running Time (ERT) tracking

    Example:
        Basic usage with BBOB benchmark function:

        >>> from opt.{info.category}.{info.filepath.stem} import {info.class_name}
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = {info.class_name}(
        ...     func=shifted_ackley,
        ...     lower_bound=-2.768,
        ...     upper_bound=2.768,
        ...     dim=2,
        ...     max_iter=100,
        ...     seed=42  # Required for reproducibility
        ... )
        >>> solution, fitness = optimizer.search()
        >>> isinstance(fitness, float) and fitness >= 0
        True

        COCO benchmark example:

        >>> from opt.benchmark.functions import sphere
        >>> optimizer = {info.class_name}(
        ...     func=sphere,
        ...     lower_bound=-5,
        ...     upper_bound=5,
        ...     dim=10,
        ...     max_iter=10000,
        ...     seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> len(solution) == 10
        True

    Args:
        FIXME: Document all parameters with BBOB guidance.
        Detected parameters from __init__ signature: {", ".join(info.parameters)}

        Common parameters (adjust based on actual signature):
        func (Callable[[ndarray], float]): Objective function to minimize. Must accept
            numpy array and return scalar. BBOB functions available in
            `opt.benchmark.functions`.
        lower_bound (float): Lower bound of search space. BBOB typical: -5
            (most functions).
        upper_bound (float): Upper bound of search space. BBOB typical: 5
            (most functions).
        dim (int): Problem dimensionality. BBOB standard dimensions: 2, 3, 5, 10, 20, 40.
        max_iter (int, optional): Maximum iterations. BBOB recommendation: 10000 for
            complete evaluation. Defaults to 1000.
        seed (int | None, optional): Random seed for reproducibility. BBOB requires
            seeds 0-14 for 15 runs. If None, generates random seed. Defaults to None.
        population_size (int, optional): Population size. BBOB recommendation: 10*dim
            for population-based methods. Defaults to 100. (Only for population-based
            algorithms)
        track_history (bool, optional): Enable convergence history tracking for BBOB
            post-processing. Defaults to False.
        FIXME: [algorithm_specific_params] ([type], optional): FIXME: Document any
            algorithm-specific parameters not listed above. Defaults to [value].

    Attributes:
        func (Callable[[ndarray], float]): The objective function being optimized.
        lower_bound (float): Lower search space boundary.
        upper_bound (float): Upper search space boundary.
        dim (int): Problem dimensionality.
        max_iter (int): Maximum number of iterations.
        seed (int): **REQUIRED** Random seed for reproducibility (BBOB compliance).
        population_size (int): Number of individuals in population.
        track_history (bool): Whether convergence history is tracked.
        history (dict[str, list]): Optimization history if track_history=True. Contains:
            - 'best_fitness': list[float] - Best fitness per iteration
            - 'best_solution': list[ndarray] - Best solution per iteration
            - 'population_fitness': list[ndarray] - All fitness values
            - 'population': list[ndarray] - All solutions
        FIXME: [algorithm_specific_attrs] ([type]): FIXME: [Description]

    Methods:
        search() -> {return_type}:
            Execute optimization algorithm.

            Returns:
                {return_type}:
                    {return_desc}

            Raises:
                ValueError:
                    If search space is invalid or function evaluation fails.

            Notes:
                - Modifies self.history if track_history=True
                - Uses self.seed for all random number generation
                - BBOB: Returns final best solution after max_iter or convergence

    References:
        FIXME: [1] Author1, A., Author2, B. (YEAR). "Algorithm Name: Description."
            _Journal Name_, Volume(Issue), Pages.
            https://doi.org/10.xxxx/xxxxx

        [2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
            "COCO: A platform for comparing continuous optimizers in a black-box setting."
            _Optimization Methods and Software_, 36(1), 114-144.
            https://doi.org/10.1080/10556788.2020.1808977

        **COCO Data Archive**:
            - Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
            - FIXME: Algorithm data: [URL to algorithm-specific COCO results if available]
            - Code repository: https://github.com/Anselmoo/useful-optimizer

        **Implementation**:
            - FIXME: Original paper code: [URL if different from this implementation]
            - This implementation: Based on [1] with modifications for BBOB compliance

    See Also:
        FIXME: [RelatedAlgorithm1]: Similar algorithm with [key difference]
            BBOB Comparison: [Brief performance notes on sphere/rosenbrock/ackley]

        FIXME: [RelatedAlgorithm2]: [Relationship description]
            BBOB Comparison: Generally [faster/slower/more robust] on [function classes]

        AbstractOptimizer: Base class for all optimizers
        opt.benchmark.functions: BBOB-compatible test functions

        Related BBOB Algorithm Classes:
            - Evolutionary: GeneticAlgorithm, DifferentialEvolution
            - Swarm: ParticleSwarm, AntColony
            - Gradient: AdamW, SGDMomentum

    Notes:
        **Computational Complexity**:
            - Time per iteration: FIXME: $O(\\text{{[expression]}})$
            - Space complexity: FIXME: $O(\\text{{[expression]}})$
            - BBOB budget usage: FIXME: _[Typical percentage of dim*10000 budget needed]_

        **BBOB Performance Characteristics**:
            - **Best function classes**: FIXME: [Unimodal/Multimodal/Ill-conditioned/...]
            - **Weak function classes**: FIXME: [Function types where algorithm struggles]
            - Typical success rate at 1e-8 precision: FIXME: **[X]%** (dim=5)
            - Expected Running Time (ERT): FIXME: [Comparative notes vs other algorithms]

        **Convergence Properties**:
            - Convergence rate: FIXME: [Linear/Quadratic/Exponential]
            - Local vs Global: FIXME: [Tendency for local/global optima]
            - Premature convergence risk: FIXME: **[High/Medium/Low]**

        **Reproducibility**:
            - **Deterministic**: FIXME: [Yes/No] - Same seed guarantees same results
            - **BBOB compliance**: seed parameter required for 15 independent runs
            - Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
            - RNG usage: `numpy.random.default_rng(self.seed)` throughout

        **Implementation Details**:
            - Parallelization: FIXME: [Not supported/Supported via `[method]`]
            - Constraint handling: FIXME: [Clamping to bounds/Penalty/Repair]
            - Numerical stability: FIXME: [Considerations for floating-point arithmetic]

        **Known Limitations**:
            - FIXME: [Any known issues or limitations specific to this implementation]
            - FIXME: BBOB known issues: [Any BBOB-specific challenges]

        **Version History**:
            - v0.1.0: Initial implementation
            - FIXME: [vX.X.X]: [Changes relevant to BBOB compliance]
"""'''

    return template


def find_optimizer_files(base_path: Path, category: str | None = None) -> list[Path]:
    """Find all optimizer Python files, excluding abstract base classes.

    Args:
        base_path: Base directory containing optimizer categories.
        category: Optional category name to filter by.

    Returns:
        List of Path objects for optimizer files to process.
    """
    optimizer_files = []

    categories = [category] if category else OPTIMIZER_CATEGORIES

    for cat in categories:
        cat_path = base_path / cat
        if not cat_path.exists():
            continue

        # Find all .py files in category, excluding abstract_*.py and __init__.py
        optimizer_files.extend(
            py_file
            for py_file in cat_path.glob("*.py")
            if not py_file.name.startswith("abstract_")
            and py_file.name != "__init__.py"
        )
    return sorted(optimizer_files)


def write_template_to_file(filepath: Path, info: OptimizerInfo, template: str) -> bool:
    """Write the generated template to the optimizer file.

    Args:
        filepath: Path to the optimizer file.
        info: OptimizerInfo object with class metadata.
        template: Generated docstring template.

    Returns:
        True if file was successfully written, False otherwise.
    """
    try:
        # Read the original file
        with filepath.open("r", encoding="utf-8") as f:
            original_content = f.read()

        # Parse to find the class definition
        tree = ast.parse(original_content)

        # Find the class node and its docstring location
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == info.class_name:
                # Calculate line numbers for replacement
                class_start_line = node.lineno

                # Find where to insert/replace the docstring
                lines = original_content.splitlines(keepends=True)

                if existing_docstring_node := ast.get_docstring(node, clean=False):
                    if not node.body or not isinstance(node.body[0], ast.Expr):
                        # Shouldn't happen if get_docstring returned something
                        return False
                    docstring_node = node.body[0]
                    # Replace existing docstring
                    start_line = docstring_node.lineno - 1
                    end_line = docstring_node.end_lineno

                    # Build new content
                    new_lines = [
                        *lines[:start_line],
                        f"    {template}\n",
                        *lines[end_line:],
                    ]
                else:
                    # No existing docstring - insert after class definition line
                    # Find the line after "class ClassName(...):"
                    insert_line = class_start_line  # This is 1-indexed
                    new_lines = [
                        *lines[:insert_line],
                        f"    {template}\n\n",
                        *lines[insert_line:],
                    ]

                # Write the modified content
                new_content = "".join(new_lines)
                with filepath.open("w", encoding="utf-8") as f:
                    f.write(new_content)

                return True

        # No matching class found - this is in else block of the for loop
        return False  # noqa: TRY300

    except Exception as e:
        print(f"❌ Error writing to {filepath}: {e}", file=sys.stderr)
        return False


def process_optimizer(filepath: Path, *, dry_run: bool = False) -> OptimizerInfo | None:
    """Process a single optimizer file.

    Args:
        filepath: Path to the optimizer file.
        dry_run: If True, only preview changes without modifying files.

    Returns:
        OptimizerInfo object if processing was successful, None otherwise.
    """
    info = extract_optimizer_info(filepath)
    if not info:
        return None

    # Generate template
    template = generate_bbob_docstring_template(info)

    # Output processing information
    print(f"\n📝 Generated BBOB template for {info.class_name}")

    # Safely compute relative path
    try:
        display_path = filepath.relative_to(filepath.parents[2])
    except (IndexError, ValueError):
        display_path = filepath
    print(f"   File: {display_path}")

    print(f"   Category: {info.category}")
    print(f"   Parameters: {', '.join(info.parameters)}")

    if dry_run:
        print("   Action: DRY RUN - No changes made")
        print("   Template preview (first 300 chars):")
        print(f"   {template[:300]}...")
    # Write template to file
    elif write_template_to_file(filepath, info, template):
        print("   Action: ✅ Docstring template written to file")
        print(
            "   ⚠️  Note: Review and complete FIXME markers in the generated template."
        )
    else:
        print("   Action: ❌ Failed to write template to file")
        return None

    return info


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the batch docstring update script.

    Args:
        argv: Command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(
        description="Batch update optimizer docstrings with COCO/BBOB compliance templates."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without modifying files"
    )
    parser.add_argument(
        "--category",
        choices=OPTIMIZER_CATEGORIES,
        help="Process only the specified category",
    )

    args = parser.parse_args(argv)

    # Find the opt directory
    script_dir = Path(__file__).parent
    opt_dir = script_dir.parent / "opt"

    if not opt_dir.exists():
        print(f"❌ Error: opt directory not found at {opt_dir}", file=sys.stderr)
        return 1

    # Find all optimizer files
    optimizer_files = find_optimizer_files(opt_dir, args.category)

    if not optimizer_files:
        print("❌ No optimizer files found to process.", file=sys.stderr)
        return 1

    print(f"🔍 Found {len(optimizer_files)} optimizer files to process")
    if args.dry_run:
        print("🔄 Running in DRY RUN mode - no files will be modified")
    if args.category:
        print(f"📂 Processing category: {args.category}")

    # Process each optimizer
    processed = []
    failed = []

    for filepath in optimizer_files:
        try:
            info = process_optimizer(filepath, dry_run=args.dry_run)
            if info:
                processed.append(info)
            else:
                failed.append(filepath)
        except Exception as e:  # noqa: PERF203
            # Exception handling in loop is necessary for graceful error handling
            print(f"❌ Error processing {filepath}: {e}", file=sys.stderr)
            failed.append(filepath)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"✅ Successfully processed {len(processed)} optimizer files")

    if failed:
        print(f"⚠️  Failed to process {len(failed)} files:")
        for filepath in failed:
            try:
                rel_path = filepath.relative_to(opt_dir.parent)
            except ValueError:
                rel_path = filepath
            print(f"   - {rel_path}")

    print("\n📊 Summary:")
    print(f"   Total files scanned: {len(optimizer_files)}")
    print(f"   Successfully processed: {len(processed)}")
    print(f"   Failed: {len(failed)}")

    if args.dry_run:
        print("\n💡 This was a dry run. To apply changes, run without --dry-run flag.")
    else:
        print("\n⚠️  Templates generated with FIXME markers. Manual review required!")
        print("   See .github/prompts/optimizer-docs-template.prompt.md for guidance.")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
