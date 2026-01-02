#!/usr/bin/env python3
"""Documentation generator for optimization algorithms.

This script parses Python source files to extract structured information
from docstrings and generates standardized Markdown documentation.

Usage:
    uv run python scripts/generate_docs.py --all
    uv run python scripts/generate_docs.py --file opt/swarm_intelligence/particle_swarm.py
    uv run python scripts/generate_docs.py --category swarm_intelligence
    uv run python scripts/generate_docs.py --dry-run  # Preview without writing
    uv run python scripts/generate_docs.py --sidebar  # Generate sidebar config

The generator extracts:
    - Module description and algorithm overview
    - Academic references with DOIs
    - Class parameters with types and defaults
    - Example usage code
    - Attributes documentation
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import textwrap

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence


# Category mappings for badge styling and navigation
CATEGORY_INFO = {
    "swarm_intelligence": {
        "name": "Swarm Intelligence",
        "badge": "badge-swarm",
        "slug": "swarm-intelligence",
        "icon": "🦋",
        "description": "Nature-inspired algorithms based on collective behavior",
    },
    "evolutionary": {
        "name": "Evolutionary",
        "badge": "badge-evolutionary",
        "slug": "evolutionary",
        "icon": "🧬",
        "description": "Algorithms inspired by biological evolution",
    },
    "gradient_based": {
        "name": "Gradient-Based",
        "badge": "badge-gradient",
        "slug": "gradient-based",
        "icon": "🧠",
        "description": "Optimization using gradient information",
    },
    "classical": {
        "name": "Classical",
        "badge": "badge-classical",
        "slug": "classical",
        "icon": "🎯",
        "description": "Traditional mathematical optimization methods",
    },
    "metaheuristic": {
        "name": "Metaheuristic",
        "badge": "badge-metaheuristic",
        "slug": "metaheuristic",
        "icon": "🔧",
        "description": "High-level problem-independent algorithms",
    },
    "physics_inspired": {
        "name": "Physics-Inspired",
        "badge": "badge-physics",
        "slug": "physics-inspired",
        "icon": "⚛️",
        "description": "Algorithms based on physical phenomena",
    },
    "probabilistic": {
        "name": "Probabilistic",
        "badge": "badge-probabilistic",
        "slug": "probabilistic",
        "icon": "📊",
        "description": "Stochastic and Bayesian optimization methods",
    },
    "social_inspired": {
        "name": "Social-Inspired",
        "badge": "badge-social",
        "slug": "social-inspired",
        "icon": "👥",
        "description": "Algorithms inspired by social behaviors",
    },
    "constrained": {
        "name": "Constrained",
        "badge": "badge-constrained",
        "slug": "constrained",
        "icon": "🔒",
        "description": "Optimization with constraints",
    },
    "multi_objective": {
        "name": "Multi-Objective",
        "badge": "badge-multi",
        "slug": "multi-objective",
        "icon": "📈",
        "description": "Pareto-optimal solutions for multiple objectives",
    },
}

# Special name mappings for acronyms and known algorithms
NAME_MAPPINGS = {
    # Acronym fixes - exact class name to slug
    "BFGS": "bfgs",
    "LBFGS": "lbfgs",
    "Bfgs": "bfgs",
    "Lbfgs": "lbfgs",
    "L-BFGS": "lbfgs",
    "CMAESAlgorithm": "cma-es",
    "CmaEsAlgorithm": "cma-es",
    "CmaEs": "cma-es",
    "Cmaes": "cma-es",
    "SGD": "sgd",
    "Sgd": "sgd",
    "SGDMomentum": "sgd-momentum",
    "SgdMomentum": "sgd-momentum",
    "StochasticGradientDescent": "sgd",
    "RMSprop": "rmsprop",
    "RMSProp": "rmsprop",
    "Rmsprop": "rmsprop",
    "ADAGrad": "adagrad",
    "AdaGrad": "adagrad",
    "Adagrad": "adagrad",
    "AdaDelta": "adadelta",
    "Adadelta": "adadelta",
    "AdaMax": "adamax",
    "Adamax": "adamax",
    "AdamW": "adamw",
    "Adamw": "adamw",
    "AMSGrad": "amsgrad",
    "Amsgrad": "amsgrad",
    "NAdam": "nadam",
    "Nadam": "nadam",
    "ADAMOptimization": "adam",
    "AdaptiveMomentEstimation": "adam",
    "Adam": "adam",
    "NesterovAcceleratedGradient": "nesterov",
    "NSGAII": "nsga-ii",
    "NsgaII": "nsga-ii",
    "NsgaIi": "nsga-ii",
    "SPEA2": "spea2",
    "Spea2": "spea2",
    "MOEAD": "moead",
    "Moead": "moead",
}

# Human-readable name mappings
DISPLAY_NAMES = {
    "bfgs": "BFGS",
    "lbfgs": "L-BFGS",
    "cma-es": "CMA-ES",
    "sgd": "Stochastic Gradient Descent",
    "sgd-momentum": "SGD with Momentum",
    "rmsprop": "RMSprop",
    "adagrad": "Adagrad",
    "adadelta": "Adadelta",
    "adamax": "Adamax",
    "adamw": "AdamW",
    "adam": "Adam",
    "amsgrad": "AMSGrad",
    "nadam": "NAdam",
    "nesterov": "Nesterov Accelerated Gradient",
    "nsga-ii": "NSGA-II",
    "spea2": "SPEA2",
    "moead": "MOEA/D",
    "particle-swarm": "Particle Swarm Optimization",
    "ant-colony": "Ant Colony Optimization",
    "grey-wolf": "Grey Wolf Optimizer",
    "whale": "Whale Optimization Algorithm",
    "firefly": "Firefly Algorithm",
    "bat": "Bat Algorithm",
    "cuckoo-search": "Cuckoo Search",
    "harmony-search": "Harmony Search",
    "simulated-annealing": "Simulated Annealing",
    "genetic": "Genetic Algorithm",
    "differential-evolution": "Differential Evolution",
    "nelder-mead": "Nelder-Mead",
    "hill-climbing": "Hill Climbing",
    "trust-region": "Trust Region",
    "conjugate-gradient": "Conjugate Gradient",
    "tabu-search": "Tabu Search",
    "powell": "Powell's Method",
}


@dataclass
class ParameterInfo:
    """Information about a function/method parameter."""

    name: str
    type_hint: str = "Any"
    default: str | None = None
    description: str = ""

    @property
    def is_required(self) -> bool:
        """Check if parameter is required (no default value)."""
        return self.default is None


@dataclass
class DocstringInfo:
    """Parsed information from a docstring."""

    short_description: str = ""
    long_description: str = ""
    reference: str = ""
    reference_doi: str = ""
    example_code: str = ""
    attributes: list[tuple[str, str]] = field(default_factory=list)
    parameters: list[ParameterInfo] = field(default_factory=list)
    returns: str = ""


@dataclass
class OptimizerDoc:
    """Complete documentation for an optimizer."""

    name: str
    class_name: str
    category: str
    module_path: str
    file_slug: str
    docstring_info: DocstringInfo
    init_params: list[ParameterInfo] = field(default_factory=list)


def parse_google_docstring(docstring: str | None) -> DocstringInfo:
    """Parse a Google-style docstring into structured information."""
    if not docstring:
        return DocstringInfo()

    info = DocstringInfo()
    docstring = textwrap.dedent(docstring).strip()

    lines = docstring.split("\n")
    current_section = "description"
    section_content: dict[str, list[str]] = {
        "description": [],
        "reference": [],
        "example": [],
        "attributes": [],
        "args": [],
        "returns": [],
    }

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Detect section headers
        if stripped.lower().startswith("reference:"):
            current_section = "reference"
            remaining = stripped[len("reference:") :].strip()
            if remaining:
                section_content["reference"].append(remaining)
            i += 1
            continue
        if stripped.lower().startswith("example:"):
            current_section = "example"
            i += 1
            continue
        if stripped == "Attributes:":
            current_section = "attributes"
            i += 1
            continue
        if stripped == "Args:":
            current_section = "args"
            i += 1
            continue
        if stripped == "Returns:":
            current_section = "returns"
            i += 1
            continue
        if stripped == "Methods:":
            current_section = "skip"
            i += 1
            continue
        if stripped.startswith(">>>"):
            current_section = "example"
            section_content["example"].append(line)
            i += 1
            continue

        if current_section != "skip":
            section_content[current_section].append(line)
        i += 1

    # Process description
    desc_lines = section_content["description"]
    if desc_lines:
        paragraphs = "\n".join(desc_lines).split("\n\n")
        if paragraphs:
            info.short_description = paragraphs[0].strip()
            if len(paragraphs) > 1:
                info.long_description = "\n\n".join(paragraphs[1:]).strip()

    # Process reference
    ref_text = "\n".join(section_content["reference"]).strip()
    if ref_text:
        info.reference = ref_text
        doi_match = re.search(r"DOI:\s*(10\.\d+/[^\s]+)", ref_text)
        if doi_match:
            info.reference_doi = doi_match.group(1)

    # Process example
    example_lines = section_content["example"]
    if example_lines:
        cleaned = []
        for line in example_lines:
            if line.strip().startswith(">>>") or line.strip().startswith("..."):
                cleaned.append(line.strip()[4:])
            else:
                cleaned.append(line)
        info.example_code = "\n".join(cleaned).strip()

    # Process attributes
    attr_lines = section_content["attributes"]
    attr_text = "\n".join(attr_lines)
    attr_pattern = r"(\w+)\s*\(([^)]+)\):\s*(.+?)(?=\n\s*\w+\s*\(|\Z)"
    for match in re.finditer(attr_pattern, attr_text, re.DOTALL):
        name, _, desc = match.groups()
        info.attributes.append((name, desc.strip()))

    # Process args/parameters
    args_text = "\n".join(section_content["args"])
    parse_args_section(args_text, info.parameters)

    # Process returns
    info.returns = "\n".join(section_content["returns"]).strip()

    return info


def parse_args_section(args_text: str, params: list[ParameterInfo]) -> None:
    """Parse the Args section of a docstring."""
    pattern = r"(\w+)\s*\(([^)]+)\):\s*(.+?)(?=\n\s*\w+\s*\(|\Z)"

    for match in re.finditer(pattern, args_text, re.DOTALL):
        name, type_info, desc = match.groups()
        type_hint = type_info.split(",")[0].strip()
        is_optional = "optional" in type_info.lower()

        default = None
        default_match = re.search(r"[Dd]efaults?\s+to\s+(\S+)", desc)
        if default_match:
            default = default_match.group(1).rstrip(".")

        params.append(
            ParameterInfo(
                name=name,
                type_hint=type_hint,
                default=default if is_optional else None,
                description=desc.strip().split(".")[0] + ".",
            )
        )


# Base class parameters for classes without custom __init__
_ABSTRACT_OPTIMIZER_PARAMS = [
    ParameterInfo(
        "func", "Callable[[ndarray], float]", None, "Objective function to minimize"
    ),
    ParameterInfo("lower_bound", "float", None, "Lower bound of search space"),
    ParameterInfo("upper_bound", "float", None, "Upper bound of search space"),
    ParameterInfo("dim", "int", None, "Problem dimensionality"),
    ParameterInfo("max_iter", "int", "1000", "Maximum number of iterations"),
    ParameterInfo("seed", "int | None", "None", "Random seed for reproducibility"),
    ParameterInfo(
        "population_size", "int", "100", "Number of individuals in population"
    ),
    ParameterInfo(
        "track_history", "bool", "False", "Track optimization history for visualization"
    ),
]


def extract_init_signature(class_node: ast.ClassDef) -> list[ParameterInfo]:
    """Extract parameter information from __init__ method."""
    params: list[ParameterInfo] = []
    has_init = False

    for node in ast.walk(class_node):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            has_init = True
            args = node.args
            defaults = args.defaults
            num_defaults = len(defaults)
            num_args = len(args.args)

            for i, arg in enumerate(args.args):
                if arg.arg == "self":
                    continue

                type_hint = "Any"
                if arg.annotation:
                    type_hint = ast.unparse(arg.annotation)

                default = None
                default_idx = i - (num_args - num_defaults)
                if default_idx >= 0:
                    default = ast.unparse(defaults[default_idx])

                params.append(
                    ParameterInfo(name=arg.arg, type_hint=type_hint, default=default)
                )
            break

    if not has_init:
        return list(_ABSTRACT_OPTIMIZER_PARAMS)

    return params


def class_name_to_slug(class_name: str) -> str:
    """Convert a class name to a URL-friendly slug."""
    # Check for known mappings first
    if class_name in NAME_MAPPINGS:
        return NAME_MAPPINGS[class_name]

    # Handle common patterns
    slug = class_name

    # Replace known acronyms before splitting
    acronym_replacements = [
        ("BFGS", "-bfgs-"),
        ("LBFGS", "-lbfgs-"),
        ("SGD", "-sgd-"),
        ("RMSprop", "-rmsprop-"),
        ("RMSProp", "-rmsprop-"),
        ("CMA", "-cma-"),
        ("NSGA", "-nsga-"),
        ("SPEA", "-spea-"),
        ("MOEAD", "-moead-"),
    ]
    for acronym, replacement in acronym_replacements:
        if acronym in slug:
            slug = slug.replace(acronym, replacement)

    # Split CamelCase to kebab-case
    slug = re.sub(r"(?<!^)(?=[A-Z])", "-", slug).lower()

    # Clean up
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")

    # Remove common suffixes for cleaner URLs
    suffixes_to_remove = [
        "-optimizer",
        "-algorithm",
        "-optimization",
        "-method",
        "-search",
    ]
    for suffix in suffixes_to_remove:
        if slug.endswith(suffix) and len(slug) > len(suffix) + 2:
            slug = slug[: -len(suffix)]

    return slug


def slug_to_display_name(slug: str, class_name: str) -> str:
    """Convert a slug to a human-readable display name."""
    if slug in DISPLAY_NAMES:
        return DISPLAY_NAMES[slug]

    # Generate from class name
    name = re.sub(r"(?<!^)(?=[A-Z])", " ", class_name)

    # Fix common acronym patterns
    fixes = [
        ("Cma Es", "CMA-ES"),
        ("Bfgs", "BFGS"),
        ("Lbfgs", "L-BFGS"),
        ("Sgd", "SGD"),
        ("Rms Prop", "RMSprop"),
        ("R M Sprop", "RMSprop"),
        ("Nsga I I", "NSGA-II"),
        ("Nsga Ii", "NSGA-II"),
        ("Spea 2", "SPEA2"),
        ("Moead", "MOEA/D"),
        ("A D A Grad", "Adagrad"),
        ("Ada Grad", "Adagrad"),
        ("Ada Max", "Adamax"),
        ("Ada Delta", "Adadelta"),
        ("A M S Grad", "AMSGrad"),
        ("Adam W", "AdamW"),
        ("N Adam", "NAdam"),
    ]
    for pattern, replacement in fixes:
        name = name.replace(pattern, replacement)

    return name


def parse_optimizer_file(file_path: Path) -> OptimizerDoc | None:
    """Parse a Python file and extract optimizer documentation."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Error parsing {file_path}: {e}")
        return None

    module_docstring = ast.get_docstring(tree)
    module_info = parse_google_docstring(module_docstring)

    optimizer_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and "Optimizer" in base.id:
                    optimizer_class = node
                    break
                if isinstance(base, ast.Attribute) and "Optimizer" in base.attr:
                    optimizer_class = node
                    break
            if optimizer_class:
                break

    if not optimizer_class:
        return None

    class_docstring = ast.get_docstring(optimizer_class)
    class_info = parse_google_docstring(class_docstring)

    init_params = extract_init_signature(optimizer_class)

    merged_info = DocstringInfo(
        short_description=module_info.short_description or class_info.short_description,
        long_description=module_info.long_description or class_info.long_description,
        reference=module_info.reference or class_info.reference,
        reference_doi=module_info.reference_doi or class_info.reference_doi,
        example_code=module_info.example_code or class_info.example_code,
        attributes=module_info.attributes or class_info.attributes,
        parameters=class_info.parameters or module_info.parameters,
        returns=class_info.returns or module_info.returns,
    )

    parts = file_path.parts
    category = "unknown"
    for i, part in enumerate(parts):
        if part == "opt" and i + 1 < len(parts):
            category = parts[i + 1]
            break

    file_slug = class_name_to_slug(optimizer_class.name)
    display_name = slug_to_display_name(file_slug, optimizer_class.name)

    return OptimizerDoc(
        name=display_name,
        class_name=optimizer_class.name,
        category=category,
        module_path=str(file_path),
        file_slug=file_slug,
        docstring_info=merged_info,
        init_params=init_params,
    )


def generate_markdown(doc: OptimizerDoc) -> str:
    """Generate Markdown documentation from parsed optimizer info."""
    cat_info = CATEGORY_INFO.get(doc.category, {})
    badge_name = cat_info.get("name", doc.category.replace("_", " ").title())
    badge_class = cat_info.get("badge", "badge-default")
    cat_slug = cat_info.get("slug", doc.category.replace("_", "-"))

    lines: list[str] = []

    # Title and badge
    lines.append(f"# {doc.name}")
    lines.append("")
    lines.append(f'<span class="badge {badge_class}">{badge_name}</span>')
    lines.append("")

    # Short description - clean it up
    if doc.docstring_info.short_description:
        desc = doc.docstring_info.short_description
        # Remove redundant module-level descriptions
        desc = re.sub(
            r"^This module (?:provides|implements|contains)\s+",
            "",
            desc,
            flags=re.IGNORECASE,
        )
        desc = desc[0].upper() + desc[1:] if desc else desc
        lines.append(desc)
        lines.append("")

    # Algorithm Overview
    if doc.docstring_info.long_description:
        lines.append("## Algorithm Overview")
        lines.append("")
        lines.append(doc.docstring_info.long_description)
        lines.append("")

    # Reference with proper citation format
    if doc.docstring_info.reference:
        lines.append("## Reference")
        lines.append("")
        ref = doc.docstring_info.reference
        ref = re.sub(r"\s+", " ", ref)  # Normalize whitespace
        lines.append(f"> {ref}")
        if doc.docstring_info.reference_doi:
            lines.append("")
            lines.append(
                f"[📄 View Paper (DOI: {doc.docstring_info.reference_doi})](https://doi.org/{doc.docstring_info.reference_doi})"
            )
        lines.append("")

    # Usage section with import statement
    lines.append("## Usage")
    lines.append("")

    module_name = Path(doc.module_path).stem

    # Always generate a standardized example with correct imports
    lines.append("```python")
    lines.append(f"from opt.{doc.category}.{module_name} import {doc.class_name}")
    lines.append("from opt.benchmark.functions import sphere")
    lines.append("")
    lines.append(f"optimizer = {doc.class_name}(")
    lines.append("    func=sphere,")
    lines.append("    lower_bound=-5.12,")
    lines.append("    upper_bound=5.12,")
    lines.append("    dim=10,")
    lines.append("    max_iter=500,")
    if any(p.name == "population_size" for p in doc.init_params):
        lines.append("    population_size=50,")
    lines.append(")")
    lines.append("")
    lines.append("best_solution, best_fitness = optimizer.search()")
    lines.append('print(f"Best solution: {best_solution}")')
    lines.append('print(f"Best fitness: {best_fitness:.6e}")')
    lines.append("```")
    lines.append("")

    # Parameters table
    lines.append("## Parameters")
    lines.append("")
    lines.append("| Parameter | Type | Default | Description |")
    lines.append("|-----------|------|---------|-------------|")

    for param in doc.init_params:
        default_str = "Required" if param.is_required else f"`{param.default}`"

        desc = ""
        for doc_param in doc.docstring_info.parameters:
            if doc_param.name == param.name:
                desc = doc_param.description
                break
        if not desc:
            desc = _get_default_param_description(param.name)

        type_display = param.type_hint
        type_display = type_display.replace("|", " \\| ")
        type_display = re.sub(r"Callable\[\[.*?\],\s*float\]", "Callable", type_display)

        lines.append(f"| `{param.name}` | `{type_display}` | {default_str} | {desc} |")

    lines.append("")

    # See Also section
    lines.append("## See Also")
    lines.append("")
    lines.append(f"- [{badge_name} Algorithms](/algorithms/{cat_slug}/)")
    lines.append("- [All Algorithms](/algorithms/)")
    lines.append("- [Benchmark Functions](/api/benchmark-functions)")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("::: tip Source Code")
    lines.append(
        f"View the implementation: [`{module_name}.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/{doc.module_path})"
    )
    lines.append(":::")
    lines.append("")

    return "\n".join(lines)


def _get_default_param_description(name: str) -> str:
    """Get a default description for common parameter names."""
    defaults = {
        "func": "Objective function to minimize",
        "lower_bound": "Lower bound of search space",
        "upper_bound": "Upper bound of search space",
        "dim": "Problem dimensionality",
        "max_iter": "Maximum number of iterations",
        "population_size": "Number of individuals in population",
        "seed": "Random seed for reproducibility",
        "learning_rate": "Step size for parameter updates",
        "beta1": "First moment decay rate",
        "beta2": "Second moment decay rate",
        "epsilon": "Numerical stability constant",
        "weight_decay": "L2 regularization coefficient",
        "c1": "Cognitive coefficient (personal best attraction)",
        "c2": "Social coefficient (global best attraction)",
        "w": "Inertia weight",
        "F": "Mutation/scaling factor",
        "CR": "Crossover probability",
        "track_history": "Track optimization history for visualization",
        "track_convergence": "Track convergence history",
        "mutation_rate": "Probability of mutation",
        "crossover_rate": "Probability of crossover",
        "temperature": "Initial temperature for annealing",
        "cooling_rate": "Rate of temperature decrease",
        "alpha": "Step size or learning rate",
        "gamma": "Discount or decay factor",
        "rho": "Pheromone evaporation rate",
        "q0": "Exploitation vs exploration parameter",
    }
    return defaults.get(name, "Algorithm-specific parameter")


def get_output_path(doc: OptimizerDoc, output_dir: Path) -> Path:
    """Determine the output path for a documentation file."""
    cat_slug = CATEGORY_INFO.get(doc.category, {}).get(
        "slug", doc.category.replace("_", "-")
    )
    return output_dir / "algorithms" / cat_slug / f"{doc.file_slug}.md"


def generate_sidebar_config(docs: list[OptimizerDoc], output_path: Path) -> None:
    """Generate VitePress sidebar configuration TypeScript file."""
    by_category: dict[str, list[OptimizerDoc]] = {}
    for doc in docs:
        if doc.category not in by_category:
            by_category[doc.category] = []
        by_category[doc.category].append(doc)

    for category, docs_list in by_category.items():
        docs_list.sort(key=lambda d: d.name)

    sidebar_items = []
    sidebar_items.append(
        {
            "text": "Overview",
            "items": [{"text": "Introduction", "link": "/algorithms/"}],
        }
    )

    category_order = [
        "swarm_intelligence",
        "evolutionary",
        "gradient_based",
        "classical",
        "metaheuristic",
        "physics_inspired",
        "probabilistic",
        "social_inspired",
        "constrained",
        "multi_objective",
    ]

    for category in category_order:
        if category not in by_category:
            continue

        cat_info = CATEGORY_INFO.get(category, {})
        cat_name = cat_info.get("name", category.replace("_", " ").title())
        cat_slug = cat_info.get("slug", category.replace("_", "-"))
        icon = cat_info.get("icon", "")

        items = []
        for doc in by_category[category]:
            items.append(
                {"text": doc.name, "link": f"/algorithms/{cat_slug}/{doc.file_slug}"}
            )

        sidebar_items.append(
            {
                "text": f"{icon} {cat_name} ({len(items)})",
                "collapsed": category != "swarm_intelligence",
                "items": items,
            }
        )

    ts_content = f"""// Auto-generated sidebar configuration
// Run: uv run python scripts/generate_docs.py --all --sidebar

export const algorithmsSidebar = {json.dumps(sidebar_items, indent=2)}
"""

    output_path.write_text(ts_content, encoding="utf-8")
    print(f"Generated sidebar config: {output_path}")


def generate_json_metadata(docs: list[OptimizerDoc], output_path: Path) -> None:
    """Generate JSON metadata file for VitePress data loader."""
    optimizers = []

    for doc in docs:
        cat_info = CATEGORY_INFO.get(doc.category, {})
        cat_slug = cat_info.get("slug", doc.category.replace("_", "-"))

        optimizer_data = {
            "name": doc.name,
            "class_name": doc.class_name,
            "category": doc.category,
            "category_display": cat_info.get("name", doc.category.replace("_", " ").title()),
            "slug": doc.file_slug,
            "link": f"/algorithms/{cat_slug}/{doc.file_slug}",
            "description": doc.docstring_info.short_description,
            "reference": doc.docstring_info.reference,
            "reference_doi": doc.docstring_info.reference_doi,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type_hint,
                    "default": p.default,
                    "required": p.is_required,
                    "description": next(
                        (dp.description for dp in doc.docstring_info.parameters if dp.name == p.name),
                        _get_default_param_description(p.name)
                    )
                }
                for p in doc.init_params
            ],
            "example_code": doc.docstring_info.example_code,
        }
        optimizers.append(optimizer_data)

    # Sort by category then name
    optimizers.sort(key=lambda x: (x["category"], x["name"]))

    output_data = {
        "version": "1.0.0",
        "generated": "auto-generated by scripts/generate_docs.py",
        "total_count": len(optimizers),
        "optimizers": optimizers
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(f"Generated JSON metadata: {output_path}")


def generate_griffe_json(category: str, output_dir: Path, *, verbose: bool = False) -> bool:
    """Generate Griffe JSON API documentation for a specific category.

    Args:
        category: Category name (e.g., 'swarm_intelligence')
        output_dir: Output directory for JSON files
        verbose: Print detailed progress information

    Returns:
        True if successful, False otherwise
    """
    output_path = output_dir / "api" / f"{category}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cmd = [
            "uv", "run", "griffe", "dump",
            f"opt.{category}",
            "-d", "google",
            "-r",  # Resolve aliases
            "-o", str(output_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        if verbose:
            print(f"    Griffe: {output_path}")
            if result.stderr:
                print(f"    Info: {result.stderr.strip()}")

    except subprocess.CalledProcessError as e:
        print(f"Error generating Griffe JSON for {category}: {e}")
        if e.stderr:
            print(f"  {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: griffe command not found. Please install griffe.")
        return False
    else:
        return True


def generate_full_api_json(output_dir: Path, *, verbose: bool = False) -> bool:
    """Generate full API JSON using Griffe for the entire opt package.

    Args:
        output_dir: Output directory for JSON file
        verbose: Print detailed progress information

    Returns:
        True if successful, False otherwise
    """
    output_path = output_dir / "api" / "full_api.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cmd = [
            "uv", "run", "griffe", "dump",
            "opt",
            "-d", "google",
            "-r",  # Resolve aliases
            "-o", str(output_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        if verbose:
            print(f"  Generated full API JSON: {output_path}")
            if result.stderr:
                print(f"  Info: {result.stderr.strip()}")

    except subprocess.CalledProcessError as e:
        print(f"Error generating full API JSON: {e}")
        if e.stderr:
            print(f"  {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: griffe command not found. Please install griffe.")
        return False
    else:
        return True


def main(args: Sequence[str] | None = None) -> None:
    """Main entry point for the documentation generator."""
    parser = argparse.ArgumentParser(
        description="Generate Markdown documentation from Python optimizer source files."
    )
    parser.add_argument(
        "--all", action="store_true", help="Generate docs for all optimizer files"
    )
    parser.add_argument("--file", type=Path, help="Generate docs for a specific file")
    parser.add_argument(
        "--category",
        type=str,
        choices=list(CATEGORY_INFO.keys()),
        help="Generate docs for a specific category",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs"),
        help="Output directory for generated docs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated docs without writing files",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress information"
    )
    parser.add_argument(
        "--sidebar", action="store_true", help="Generate sidebar configuration file"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove old generated docs before generating",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Generate JSON metadata for VitePress data loader",
    )
    parser.add_argument(
        "--griffe",
        action="store_true",
        help="Generate Griffe JSON API documentation per category",
    )
    parser.add_argument(
        "--full-api",
        action="store_true",
        help="Generate full API JSON using Griffe for entire opt package",
    )

    parsed = parser.parse_args(args)

    opt_dir = Path("opt")
    files_to_process: list[Path] = []

    if parsed.file:
        if parsed.file.exists():
            files_to_process.append(parsed.file)
        else:
            print(f"Error: File not found: {parsed.file}")
            return

    elif parsed.category:
        category_dir = opt_dir / parsed.category
        if category_dir.exists():
            files_to_process.extend(
                f for f in category_dir.glob("*.py") if f.name != "__init__.py"
            )
        else:
            print(f"Error: Category directory not found: {category_dir}")
            return

    elif parsed.all or parsed.sidebar or parsed.json or parsed.griffe or parsed.full_api:
        for category in CATEGORY_INFO:
            category_dir = opt_dir / category
            if category_dir.exists():
                files_to_process.extend(
                    f for f in category_dir.glob("*.py") if f.name != "__init__.py"
                )
    else:
        parser.print_help()
        return

    if not files_to_process:
        print("No files found to process.")
        return

    if parsed.clean and not parsed.dry_run:
        for category, cat_info in CATEGORY_INFO.items():
            cat_slug = cat_info.get("slug", category.replace("_", "-"))
            cat_dir = parsed.output_dir / "algorithms" / cat_slug
            if cat_dir.exists():
                for md_file in cat_dir.glob("*.md"):
                    if md_file.name != "index.md":
                        md_file.unlink()
                        if parsed.verbose:
                            print(f"  Removed: {md_file}")

    print(f"Processing {len(files_to_process)} files...")

    generated = 0
    skipped = 0
    all_docs: list[OptimizerDoc] = []

    for file_path in sorted(files_to_process):
        if parsed.verbose:
            print(f"  Parsing: {file_path}")

        doc = parse_optimizer_file(file_path)
        if doc is None:
            if parsed.verbose:
                print("    Skipped: No optimizer class found")
            skipped += 1
            continue

        all_docs.append(doc)
        markdown = generate_markdown(doc)

        if parsed.dry_run:
            output_path = get_output_path(doc, parsed.output_dir)
            print(f"\n{'=' * 60}")
            print(f"Would write to: {output_path}")
            print("=" * 60)
            print(markdown[:800] + "..." if len(markdown) > 800 else markdown)
        else:
            output_path = get_output_path(doc, parsed.output_dir)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown, encoding="utf-8")
            if parsed.verbose:
                print(f"    Generated: {output_path}")

        generated += 1

    if parsed.sidebar and not parsed.dry_run:
        sidebar_path = parsed.output_dir / ".vitepress" / "algorithmsSidebar.ts"
        generate_sidebar_config(all_docs, sidebar_path)

    if parsed.json and not parsed.dry_run:
        json_path = parsed.output_dir / "public" / "optimizers" / "optimizers.json"
        generate_json_metadata(all_docs, json_path)

    if parsed.griffe and not parsed.dry_run:
        print("\nGenerating Griffe JSON API documentation per category...")
        for category in CATEGORY_INFO:
            category_dir = opt_dir / category
            if category_dir.exists():
                generate_griffe_json(category, parsed.output_dir, parsed.verbose)

    if parsed.full_api and not parsed.dry_run:
        print("\nGenerating full API JSON...")
        generate_full_api_json(parsed.output_dir, parsed.verbose)

    print(f"\nDone! Generated: {generated}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
