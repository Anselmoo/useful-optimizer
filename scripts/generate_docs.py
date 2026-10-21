#!/usr/bin/env python3
"""Documentation generator for optimization algorithms.

This script parses Python source files to extract structured information
from docstrings and generates standardized Markdown documentation.

Usage:
    uv run python scripts/generate_docs.py --all
    uv run python scripts/generate_docs.py --file src/opt/swarm_intelligence/particle_swarm.py
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
import operator
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


PLACEHOLDER_REFERENCE = "Available in scientific literature"


@dataclass
class ReferenceInfo:
    """One entry of a docstring ``References`` section."""

    citation: str
    doi: str = ""
    # True for the "Available in scientific literature" stand-in that some
    # docstrings carry instead of a real citation.
    placeholder: bool = False


@dataclass
class DocstringInfo:
    """Parsed information from a docstring."""

    short_description: str = ""
    long_description: str = ""
    reference: str = ""
    reference_doi: str = ""
    references: list[ReferenceInfo] = field(default_factory=list)
    example_code: str = ""
    attributes: list[tuple[str, str]] = field(default_factory=list)
    parameters: list[ParameterInfo] = field(default_factory=list)
    returns: str = ""
    # Rich narrative sections (Algorithm Metadata, Mathematical Formulation,
    # Hyperparameters, COCO/BBOB Benchmark Settings, Notes, References, See Also, …)
    # captured verbatim as (title, body) so their markdown/LaTeX renders as-is.
    raw_sections: list[tuple[str, str]] = field(default_factory=list)


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
    # Rows of the "Algorithm Metadata" table (Algorithm Name, Acronym, Year, ...).
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def acronym(self) -> str:
        """Acronym from the metadata table, empty if absent."""
        return self.metadata.get("Acronym", "")

    @property
    def year(self) -> str:
        """Year introduced from the metadata table, empty if absent."""
        return self.metadata.get("Year Introduced", "")


# Section headers with dedicated parsing (drive the Parameters table, JSON
# metadata, references, and usage example). Matched case-insensitively.
_SPECIAL_SECTIONS = {
    "reference": "reference",
    "example": "example",
    "attributes": "attributes",
    "args": "args",
    "arguments": "args",
    "parameters": "args",
    "returns": "returns",
    "methods": "skip",
}

# Rich narrative sections captured verbatim and rendered as Markdown ## sections.
# These carry the algorithm's tables, LaTeX formulas, and COCO/BBOB metadata.
_RICH_SECTIONS = {
    "algorithm metadata",
    "mathematical formulation",
    "hyperparameters",
    "coco/bbob benchmark settings",
    "notes",
    "references",
    "see also",
    "raises",
    "results",
}

# A top-level section header: begins at column 0, a short Title-like phrase
# ending in a colon (optionally followed by inline content on the same line).
_SECTION_HEADER_RE = re.compile(r"^([A-Z][A-Za-z0-9 /()_.-]*):[ \t]*(.*)$")


def _classify_header(line: str) -> tuple[str, str, str] | None:
    """Return (title, lowercase_key, inline_remainder) if line is a section header."""
    if not line or line[0].isspace():
        return None
    match = _SECTION_HEADER_RE.match(line)
    if not match:
        return None
    title = match.group(1).strip()
    return title, title.lower(), match.group(2).strip()


def _normalize_section_body(body_lines: list[str]) -> str:
    """Normalize a rich section body for Markdown rendering.

    Google-style docstrings over-indent block content (4+ spaces) which Markdown
    would misinterpret as code blocks. Left-align every line so tables, lists,
    and math render, and guarantee blank lines around ``$$…$$`` display math so
    it is parsed as a block rather than inline text.
    """
    out: list[str] = []
    in_math = False
    for raw in body_lines:
        line = raw.strip()
        if line == "$$":
            if not in_math:
                if out and out[-1]:
                    out.append("")  # blank line before opening delimiter
                out.append("$$")
                in_math = True
            else:
                out.extend(("$$", ""))  # blank line after closing delimiter
                in_math = False
            continue
        out.append(line)

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse runs of blank lines
    return text.strip("\n")


_REF_ENTRY_RE = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")
_DOI_URL_RE = re.compile(
    r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s)]+)", re.IGNORECASE
)
_DOI_LABEL_RE = re.compile(r"DOI:\s*(10\.\d{4,9}/[^\s)]+)", re.IGNORECASE)
_PLACEHOLDER_ENTRY_RE = re.compile(
    r'\[1\] (.+?) \((\d{4})\)\. "Original publication\."\s*\n\s*'
    r"_Journal/Conference_, Available in scientific literature\."
)


def parse_references(body: str) -> list[ReferenceInfo]:
    """Parse the entries (``[1] ...``) of a ``References`` section body.

    Parsing stops at the first bold sub-heading (``**COCO Data Archive**`` and
    similar), which introduces resources rather than citations.

    Args:
        body: The normalised body of the ``References`` section.

    Returns:
        One entry per numbered reference, in order, with DOI and placeholder flag.
    """
    chunks: list[list[str]] = []
    current: list[str] | None = None
    for line in body.splitlines():
        if line.strip().startswith("**"):
            break
        entry = _REF_ENTRY_RE.match(line)
        if entry:
            current = [entry.group(2)]
            chunks.append(current)
        elif not line.strip():
            current = None
        elif current is not None:
            current.append(line.strip())

    references: list[ReferenceInfo] = []
    for chunk in chunks:
        text = re.sub(r"\s+", " ", " ".join(chunk)).strip()
        doi_match = _DOI_URL_RE.search(text) or _DOI_LABEL_RE.search(text)
        doi = doi_match.group(1).rstrip(".,;") if doi_match else ""
        references.append(
            ReferenceInfo(
                citation=text, doi=doi, placeholder=PLACEHOLDER_REFERENCE in text
            )
        )
    return references


def render_references_body(body: str) -> str:
    """Make placeholder citations explicit instead of pretending they are real."""
    return _PLACEHOLDER_ENTRY_RE.sub(
        r"[1] \1 (\2): *original publication not yet cited here.*", body
    )


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
    raw_sections: list[tuple[str, list[str]]] = []
    current_raw: list[str] | None = None

    for line in lines:
        header = _classify_header(line)
        if header is not None:
            title, key, remainder = header
            if key in _SPECIAL_SECTIONS:
                current_section = _SPECIAL_SECTIONS[key]
                current_raw = None
                if key == "reference" and remainder:
                    section_content["reference"].append(remainder)
                continue
            if key in _RICH_SECTIONS:
                current_section = "raw"
                current_raw = []
                raw_sections.append((title, current_raw))
                if remainder:
                    current_raw.append(remainder)
                continue
            # Unrecognized header: fall through and keep it as narrative text.

        if current_section == "raw" and current_raw is not None:
            current_raw.append(line)
            continue
        if line.strip().startswith(">>>"):
            current_section = "example"
            current_raw = None
            section_content["example"].append(line)
            continue
        if current_section != "skip":
            section_content[current_section].append(line)

    # Assemble rich narrative sections (normalize indentation; drop empties;
    # merge repeated titles such as a second "Notes:" block into the first).
    merged: dict[str, str] = {}
    order: list[str] = []
    for title, body_lines in raw_sections:
        body = _normalize_section_body(body_lines)
        if not body.strip():
            continue
        key = title.lower()
        if key in merged:
            merged[key] = f"{merged[key]}\n\n{body}"
        else:
            merged[key] = body
            order.append(key)
    title_by_key = {title.lower(): title for title, _ in raw_sections}
    info.raw_sections = [(title_by_key[key], merged[key]) for key in order]
    if "references" in merged:
        info.references = parse_references(merged["references"])

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
    elif info.references:
        # No singular "Reference:" section: entry [1] of "References:" is the
        # algorithm's own publication (later entries are COCO and background).
        first = info.references[0]
        if not first.placeholder:
            info.reference = first.citation
            info.reference_doi = first.doi

    # Process example. For doctest-style examples keep only the executable
    # code (the >>> / ... lines), dropping expected-output lines and trailing
    # prose so the snippet is clean. Plain code blocks are kept verbatim.
    example_lines = section_content["example"]
    if example_lines:
        prompt_lines = [
            line for line in example_lines if line.strip().startswith((">>>", "..."))
        ]
        if prompt_lines:
            info.example_code = "\n".join(
                line.strip()[4:] for line in prompt_lines
            ).strip()
        else:
            info.example_code = "\n".join(example_lines).strip()

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

        # Collapse internal whitespace (multi-line docstring descriptions carry
        # newlines that would otherwise break the generated Markdown table).
        clean = re.sub(r"\s+", " ", desc.strip())
        first_sentence = clean.split(".")[0].strip()
        description = f"{first_sentence}." if first_sentence else clean

        params.append(
            ParameterInfo(
                name=name,
                type_hint=type_hint,
                default=default if is_optional else None,
                description=description,
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


def file_stem_to_slug(stem: str) -> str:
    """Return the page slug for a source file: the file stem with ``_`` as ``-``.

    The slug is the one naming rule that links source file, docs URL and sidebar,
    so ``whale_optimization_algorithm.py`` is documented at
    ``/algorithms/swarm-intelligence/whale-optimization-algorithm``.
    """
    return stem.replace("_", "-")


_METADATA_ROW_RE = re.compile(r"^\|\s*([^|]+?)\s*\|\s*(.*?)\s*\|\s*$", re.MULTILINE)


def parse_algorithm_metadata(raw_sections: list[tuple[str, str]]) -> dict[str, str]:
    """Parse the ``Algorithm Metadata`` table into a dict.

    Args:
        raw_sections: ``(title, body)`` pairs captured from the class docstring.

    Returns:
        Mapping of property name to value, e.g. ``{"Acronym": "WOA", ...}``;
        empty when the section is missing.
    """
    for title, body in raw_sections:
        if title.lower() != "algorithm metadata":
            continue
        rows: dict[str, str] = {}
        for key, value in _METADATA_ROW_RE.findall(body):
            if key == "Property" or set(key) <= set("-: "):
                continue
            rows[key] = value
        return rows
    return {}


def _class_name_to_title(class_name: str) -> str:
    """Fallback title from a CamelCase class name when metadata has none."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", class_name)


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

    # Prefer the class docstring: it carries the rich algorithm sections
    # (metadata, formulation, hyperparameters, COCO/BBOB settings). Fall back to
    # the module docstring for anything the class docstring omits (e.g. overview).
    merged_info = DocstringInfo(
        short_description=class_info.short_description or module_info.short_description,
        long_description=class_info.long_description or module_info.long_description,
        reference=class_info.reference or module_info.reference,
        reference_doi=class_info.reference_doi or module_info.reference_doi,
        references=class_info.references or module_info.references,
        example_code=class_info.example_code or module_info.example_code,
        attributes=class_info.attributes or module_info.attributes,
        parameters=class_info.parameters or module_info.parameters,
        returns=class_info.returns or module_info.returns,
        raw_sections=class_info.raw_sections or module_info.raw_sections,
    )

    parts = file_path.parts
    category = "unknown"
    for i, part in enumerate(parts):
        if part == "opt" and i + 1 < len(parts):
            category = parts[i + 1]
            break

    metadata = parse_algorithm_metadata(merged_info.raw_sections)
    file_slug = file_stem_to_slug(file_path.stem)
    display_name = metadata.get("Algorithm Name") or _class_name_to_title(
        optimizer_class.name
    )

    return OptimizerDoc(
        name=display_name,
        class_name=optimizer_class.name,
        category=category,
        module_path=str(file_path),
        file_slug=file_slug,
        docstring_info=merged_info,
        init_params=init_params,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Benchmark chart integration
# ---------------------------------------------------------------------------


def _benchmark_section_lines(doc: OptimizerDoc) -> list[str]:
    """Build the '## Benchmark Performance' Markdown block for an optimizer.

    Every page embeds the same component, which reads the published benchmark
    data in the browser. The page therefore does not depend on the data files,
    and refreshing the benchmark data does not change any generated page.
    """
    lines = ["## Benchmark Performance", ""]
    if doc.category == "multi_objective":
        lines.extend(
            (
                "::: info Not covered by the benchmark suite",
                (
                    "The benchmark suite compares single-objective optimizers on one "
                    "scalar fitness. Multi-objective optimizers return a Pareto front "
                    "and are not part of it."
                ),
                ":::",
                "",
            )
        )
        return lines

    lines.extend(
        (
            (
                "Results of the benchmark suite for this optimizer, on the same "
                "evaluation budget as every other optimizer and compared with a random "
                "search baseline. See the [methodology](/benchmarks/methodology) for the "
                "protocol and statistics."
            ),
            "",
            "<ClientOnly>",
            f'  <BenchmarkCharts algorithm="{doc.class_name}" />',
            "</ClientOnly>",
            "",
        )
    )
    return lines


def generate_markdown(doc: OptimizerDoc) -> str:
    """Generate Markdown documentation from parsed optimizer info."""
    cat_info = CATEGORY_INFO.get(doc.category, {})
    badge_name = cat_info.get("name", doc.category.replace("_", " ").title())
    badge_class = cat_info.get("badge", "badge-default")
    cat_slug = cat_info.get("slug", doc.category.replace("_", "-"))

    lines: list[str] = []

    # Title and badge
    lines.extend(
        (
            f"# {doc.name}",
            "",
            f'<span class="badge {badge_class}">{badge_name}</span>',
            "",
        )
    )

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
        lines.extend((desc, ""))

    # Algorithm Overview
    if doc.docstring_info.long_description:
        lines.extend(
            ("## Algorithm Overview", "", doc.docstring_info.long_description, "")
        )

    # Titles of the rich narrative sections captured from the class docstring.
    raw_titles = {title.lower() for title, _ in doc.docstring_info.raw_sections}

    # Reference with proper citation format (skip when the class docstring
    # already provides a richer "References" section rendered further below).
    if doc.docstring_info.reference and "references" not in raw_titles:
        lines.extend(("## Reference", ""))
        ref = doc.docstring_info.reference
        ref = re.sub(r"\s+", " ", ref)  # Normalize whitespace
        lines.append(f"> {ref}")
        if doc.docstring_info.reference_doi:
            lines.extend(
                (
                    "",
                    f"[📄 View Paper (DOI: {doc.docstring_info.reference_doi})](https://doi.org/{doc.docstring_info.reference_doi})",
                )
            )
        lines.append("")

    # Usage section with import statement
    lines.extend(("## Usage", ""))

    module_name = Path(doc.module_path).stem

    # Always generate a standardized example with correct imports
    lines.extend(
        (
            "```python",
            f"from opt.{doc.category}.{module_name} import {doc.class_name}",
            "from opt.benchmark.functions import sphere",
            "",
            f"optimizer = {doc.class_name}(",
            "    func=sphere,",
            "    lower_bound=-5.12,",
            "    upper_bound=5.12,",
            "    dim=10,",
            "    max_iter=500,",
        )
    )
    if any(p.name == "population_size" for p in doc.init_params):
        lines.append("    population_size=50,")
    lines.extend(
        (
            ")",
            "",
            "best_solution, best_fitness = optimizer.search()",
            'print(f"Best solution: {best_solution}")',
            'print(f"Best fitness: {best_fitness:.6e}")',
            "```",
            "",
        )
    )

    # Parameters table
    lines.extend(
        (
            "## Parameters",
            "",
            "| Parameter | Type | Default | Description |",
            "|-----------|------|---------|-------------|",
        )
    )

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

        # Guard against any residual newlines/pipes breaking the Markdown table.
        safe_desc = re.sub(r"\s+", " ", desc).replace("|", "\\|").strip()

        lines.append(
            f"| `{param.name}` | `{type_display}` | {default_str} | {safe_desc} |"
        )

    lines.append("")

    # Rich narrative sections from the class docstring (Algorithm Metadata,
    # Mathematical Formulation, Hyperparameters, COCO/BBOB Benchmark Settings,
    # Notes, References, See Also, …). Content is already Markdown/LaTeX and
    # renders natively in VitePress.
    for section_title, raw_body in doc.docstring_info.raw_sections:
        section_body = (
            render_references_body(raw_body)
            if section_title.lower() == "references"
            else raw_body
        )
        lines.extend((f"## {section_title}", "", section_body, ""))

    # Benchmark performance charts: analytic fitness landscape on every page,
    # plus real convergence/distribution/ECDF charts when run data exists.
    lines.extend(_benchmark_section_lines(doc))

    # Navigation links. When the class docstring supplies its own "See Also"
    # section (rendered above), use a distinct heading to avoid a duplicate.
    nav_heading = "Related Pages" if "see also" in raw_titles else "See Also"
    lines.extend(
        (
            f"## {nav_heading}",
            "",
            f"- [{badge_name} Algorithms](/algorithms/{cat_slug}/)",
            "- [All Algorithms](/algorithms/)",
            "- [Benchmark Functions](/api/benchmark-functions)",
            "",
        )
    )

    # Footer
    lines.extend(
        (
            "---",
            "",
            "::: tip Source Code",
            f"View the implementation: [`{module_name}.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/{doc.module_path})",
            ":::",
            "",
        )
    )

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


CATEGORY_ORDER = [
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

COMMON_PARAMS = frozenset(
    {
        "func",
        "lower_bound",
        "upper_bound",
        "dim",
        "max_iter",
        "seed",
        "population_size",
        "track_history",
        "target_precision",
        "f_opt",
    }
)


def _markers(name: str) -> tuple[str, str]:
    return (
        f"<!-- BEGIN GENERATED: {name} (scripts/generate_docs.py) -->",
        f"<!-- END GENERATED: {name} -->",
    )


def _group_by_category(docs: list[OptimizerDoc]) -> dict[str, list[OptimizerDoc]]:
    """Group docs by category in display order, each group sorted by name."""
    grouped: dict[str, list[OptimizerDoc]] = {}
    for category in CATEGORY_ORDER:
        members = sorted(
            (d for d in docs if d.category == category), key=lambda d: d.name.lower()
        )
        if members:
            grouped[category] = members
    return grouped


def _cat_slug(category: str) -> str:
    return CATEGORY_INFO.get(category, {}).get("slug", category.replace("_", "-"))


def _first_sentence(text: str, limit: int = 140) -> str:
    """First sentence of ``text``, table-safe and at most ``limit`` characters."""
    flat = re.sub(r"\s+", " ", text).strip()
    match = re.match(r"(.+?[.!?])(?:\s|$)", flat)
    sentence = match.group(1) if match else flat
    if len(sentence) > limit:
        sentence = sentence[: limit - 1].rstrip() + "…"
    return sentence.replace("|", "\\|")


def generate_category_index(category: str, docs: list[OptimizerDoc]) -> str:
    """Generate the index page of one category from the parsed optimizers."""
    info = CATEGORY_INFO.get(category, {})
    name = info.get("name", category.replace("_", " ").title())
    icon = info.get("icon", "")
    first = docs[0]

    lines = [
        f"# {icon} {name} Algorithms".replace("#  ", "# "),
        "",
        f"{info.get('description', '')}.",
        "",
        f"This category contains **{len(docs)} algorithm{'s' if len(docs) != 1 else ''}**.",
        "",
        "| Algorithm | Acronym | Year | Description |",
        "|-----------|---------|------|-------------|",
    ]
    for doc in docs:
        description = _first_sentence(doc.docstring_info.short_description)
        lines.append(
            f"| [{doc.name}](./{doc.file_slug}) | {doc.acronym or '-'} "
            f"| {doc.year or '-'} | {description} |"
        )
    lines += [
        "",
        "## Usage",
        "",
        "Every optimizer shares the same interface:",
        "",
        "```python",
        f"from opt.{category}.{Path(first.module_path).stem} import {first.class_name}",
        "from opt.benchmark.functions import sphere",
        "",
        f"optimizer = {first.class_name}(",
        "    func=sphere,",
        "    lower_bound=-5.12,",
        "    upper_bound=5.12,",
        "    dim=10,",
        "    max_iter=500,",
        ")",
        "best_solution, best_fitness = optimizer.search()",
        "```",
        "",
        "## See Also",
        "",
        "- [All Algorithms](/algorithms/)",
        "- [Benchmark Functions](/api/benchmark-functions)",
        "",
    ]
    return "\n".join(lines)


def generate_overview_catalog(docs: list[OptimizerDoc]) -> str:
    """Generate the catalog block of ``algorithms/index.md``."""
    grouped = _group_by_category(docs)
    lines = [
        (
            f"Useful Optimizer provides **{len(docs)} optimization algorithms** in "
            f"{len(grouped)} categories. Each one shares the same interface and is "
            "documented with its mathematical formulation, parameters, references and "
            "benchmark results."
        ),
        "",
        "## Algorithm Categories",
        "",
        "| Category | Algorithms | Description |",
        "|----------|-----------|-------------|",
    ]
    for category, members in grouped.items():
        info = CATEGORY_INFO.get(category, {})
        lines.append(
            f"| [{info.get('icon', '')} {info.get('name', category)}]"
            f"(./{_cat_slug(category)}/) | {len(members)} | {info.get('description', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def replace_generated_block(path: Path, name: str, content: str) -> bool:
    """Rewrite the generated block ``name`` of a hand-written page in place.

    The text around the markers is preserved. If the markers are missing the
    page is left untouched and a warning is printed.

    Returns:
        True when the markers were found.
    """
    begin, end = _markers(name)
    text = path.read_text(encoding="utf-8")
    if begin not in text or end not in text:
        print(f"Warning: '{name}' markers missing in {path}; page not updated")
        return False
    head, rest = text.split(begin, 1)
    tail = rest.split(end, 1)[1]
    new_text = f"{head}{begin}\n\n{content}\n{end}{tail}"
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
    print(f"Updated {name}: {path}")
    return True


def generate_api_page(category: str, docs: list[OptimizerDoc]) -> str:
    """Generate the API reference page of one category from the parsed optimizers."""
    info = CATEGORY_INFO.get(category, {})
    name = info.get("name", category.replace("_", " ").title())
    slug = _cat_slug(category)

    lines = [
        f"# {name} API",
        "",
        (
            f"API reference for the {len(docs)} classes of `opt.{category}`. "
            "This page is generated from the source; see the "
            f"[algorithm pages](/algorithms/{slug}/) for formulations and references."
        ),
        "",
        "## Imports",
        "",
        "```python",
        f"from opt.{category} import (",
        *[f"    {d.class_name}," for d in docs],
        ")",
        "```",
        "",
        "## Common interface",
        "",
        (
            "Every class takes the objective `func`, the search-space `lower_bound`, "
            "`upper_bound` and `dim`, usually `max_iter` and `seed`, and implements "
            "`search() -> tuple[np.ndarray, float]`. See "
            "[AbstractOptimizer](./abstract-optimizer)."
        ),
        "",
        "## Classes",
        "",
        "| Class | Algorithm | Additional parameters |",
        "|-------|-----------|-----------------------|",
    ]
    for doc in docs:
        extra = [p for p in doc.init_params if p.name not in COMMON_PARAMS]
        shown = ", ".join(
            f"`{p.name}`" if p.is_required else f"`{p.name}={p.default}`"
            for p in extra[:8]
        )
        if len(extra) > 8:
            shown += f", … (+{len(extra) - 8})"
        lines.append(
            f"| [`{doc.class_name}`](/algorithms/{slug}/{doc.file_slug}) "
            f"| {doc.name} | {shown or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def generate_api_overview(docs: list[OptimizerDoc]) -> str:
    """Generate the module structure and module list of ``api/index.md``."""
    grouped = _group_by_category(docs)
    lines = ["## Module Structure", "", "```", "opt/"]
    lines.append(
        "├── abstract/                # Base classes (AbstractOptimizer, history)"
    )
    for category, members in grouped.items():
        lines.append(f"├── {category + '/':<24} # {len(members)} algorithms")
    lines += ["└── benchmark/               # Benchmark functions", "```", ""]
    lines += ["## Algorithm Modules", ""]
    for category, members in grouped.items():
        info = CATEGORY_INFO.get(category, {})
        lines.append(
            f"- [{info.get('name', category)}](./{_cat_slug(category)}): "
            f"{len(members)} classes. {info.get('description', '')}."
        )
    lines.append("")
    return "\n".join(lines)


def prune_stale_pages(output_dir: Path, expected: set[Path]) -> list[Path]:
    """Delete algorithm pages that no optimizer generates any more.

    Index pages are generated separately and never pruned here.
    """
    removed: list[Path] = []
    for category in CATEGORY_INFO:
        cat_dir = output_dir / "algorithms" / _cat_slug(category)
        if not cat_dir.exists():
            continue
        for md_file in sorted(cat_dir.glob("*.md")):
            if md_file.name != "index.md" and md_file not in expected:
                md_file.unlink()
                removed.append(md_file)
    return removed


DOCS_URL = "https://anselmoo.github.io/useful-optimizer"
REPO_BLOB_URL = "https://github.com/Anselmoo/useful-optimizer/blob/main"


def generate_readme_structure(docs: list[OptimizerDoc]) -> str:
    """Generate the package layout tree of the README."""
    grouped = _group_by_category(docs)
    lines = [
        "```",
        "src/opt/",
        "├── abstract/            # Base classes and history tracking",
    ]
    for category, members in grouped.items():
        lines.append(f"├── {category + '/':<20} # {len(members)} algorithms")
    lines += ["└── benchmark/           # Benchmark functions", "```", ""]
    return "\n".join(lines)


def generate_readme_optimizers(docs: list[OptimizerDoc]) -> str:
    """Generate the per-category optimizer lists of the README."""
    grouped = _group_by_category(docs)
    lines = [
        (
            f"Useful Optimizer includes {len(docs)} optimization algorithms in "
            f"{len(grouped)} categories, each implemented as a separate module. Every "
            "optimizer links to its documentation page, which carries the formulation, "
            "parameters, references and benchmark charts, and to its source code."
        ),
        "",
    ]
    for category, members in grouped.items():
        info = CATEGORY_INFO.get(category, {})
        lines += [
            "<details>",
            (
                f"<summary><strong>{info.get('icon', '')} {info.get('name', category)} "
                f"({len(members)})</strong></summary>"
            ),
            "",
            f"{info.get('description', '')}.",
            "",
        ]
        for doc in members:
            page = f"{DOCS_URL}/algorithms/{_cat_slug(category)}/{doc.file_slug}"
            source = f"{REPO_BLOB_URL}/{doc.module_path}"
            description = _first_sentence(doc.docstring_info.short_description, 160)
            lines.append(
                f"- **[{doc.name}]({page})** ([source]({source})) - {description}"
            )
        lines += ["", "</details>", ""]
    return "\n".join(lines)


def generate_sidebar_config(docs: list[OptimizerDoc], output_path: Path) -> None:
    """Generate VitePress sidebar configuration TypeScript file."""
    by_category = _group_by_category(docs)

    sidebar_items = []
    sidebar_items.append(
        {
            "text": "Overview",
            "items": [{"text": "Introduction", "link": "/algorithms/"}],
        }
    )

    for category in by_category:
        cat_info = CATEGORY_INFO.get(category, {})
        cat_name = cat_info.get("name", category.replace("_", " ").title())
        cat_slug = cat_info.get("slug", category.replace("_", "-"))
        icon = cat_info.get("icon", "")

        items = [
            {"text": doc.name, "link": f"/algorithms/{cat_slug}/{doc.file_slug}"}
            for doc in by_category[category]
        ]
        # The category page itself comes first so it stays reachable.
        items.insert(0, {"text": "Overview", "link": f"/algorithms/{cat_slug}/"})

        sidebar_items.append(
            {
                "text": f"{icon} {cat_name} ({len(by_category[category])})",
                "collapsed": category != "swarm_intelligence",
                "items": items,
            }
        )

    api_links = [
        {
            "text": CATEGORY_INFO.get(category, {}).get("name", category),
            "link": f"/api/{_cat_slug(category)}",
        }
        for category in by_category
    ]

    ts_content = f"""// Auto-generated sidebar configuration
// Run: uv run python scripts/generate_docs.py --all --sidebar

export const algorithmsSidebar = {json.dumps(sidebar_items, indent=2)}

export const apiModuleLinks = {json.dumps(api_links, indent=2)}
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
            "category_display": cat_info.get(
                "name", doc.category.replace("_", " ").title()
            ),
            "slug": doc.file_slug,
            "module": Path(doc.module_path).stem,
            "acronym": doc.acronym,
            "year": doc.year,
            "link": f"/algorithms/{cat_slug}/{doc.file_slug}",
            "description": doc.docstring_info.short_description,
            "reference": doc.docstring_info.reference,
            "reference_doi": doc.docstring_info.reference_doi,
            "references": [
                {"citation": r.citation, "doi": r.doi, "placeholder": r.placeholder}
                for r in doc.docstring_info.references
            ],
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type_hint,
                    "default": p.default,
                    "required": p.is_required,
                    "description": next(
                        (
                            dp.description
                            for dp in doc.docstring_info.parameters
                            if dp.name == p.name
                        ),
                        _get_default_param_description(p.name),
                    ),
                }
                for p in doc.init_params
            ],
            "example_code": doc.docstring_info.example_code,
        }
        optimizers.append(optimizer_data)

    # Sort by category then name
    optimizers.sort(key=operator.itemgetter("category", "name"))

    output_data = {
        "version": "1.0.0",
        "generated": "auto-generated by scripts/generate_docs.py",
        "total_count": len(optimizers),
        "optimizers": optimizers,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Trailing newline keeps end-of-file-fixer from re-touching the file (which
    # would otherwise ping-pong with the regenerate-optimizer-docs hook).
    output_path.write_text(json.dumps(output_data, indent=2) + "\n", encoding="utf-8")
    print(f"Generated JSON metadata: {output_path}")


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
            "uv",
            "run",
            "griffe",
            "dump",
            "opt",
            "-d",
            "google",
            "-r",  # Resolve aliases
            "-o",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603

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
        help="Deprecated alias of --full-api (one Griffe dump holds every category)",
    )
    parser.add_argument(
        "--full-api",
        action="store_true",
        help="Generate full API JSON using Griffe for entire opt package",
    )

    parsed = parser.parse_args(args)

    opt_dir = Path("src/opt")
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

    elif (
        parsed.all or parsed.sidebar or parsed.json or parsed.griffe or parsed.full_api
    ):
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

    full_run = bool(parsed.all and not parsed.file and not parsed.category)
    if full_run and not parsed.dry_run:
        expected = {get_output_path(d, parsed.output_dir) for d in all_docs}
        for removed in prune_stale_pages(parsed.output_dir, expected):
            print(f"  Pruned stale page: {removed}")

        for category, members in _group_by_category(all_docs).items():
            index_path = (
                parsed.output_dir / "algorithms" / _cat_slug(category) / "index.md"
            )
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(
                generate_category_index(category, members), encoding="utf-8"
            )
        overview_path = parsed.output_dir / "algorithms" / "index.md"
        if overview_path.exists():
            replace_generated_block(
                overview_path, "catalog", generate_overview_catalog(all_docs)
            )

        readme_path = parsed.output_dir.parent / "README.md"
        if readme_path.exists():
            replace_generated_block(
                readme_path, "readme-structure", generate_readme_structure(all_docs)
            )
            replace_generated_block(
                readme_path, "readme-optimizers", generate_readme_optimizers(all_docs)
            )

        for category, members in _group_by_category(all_docs).items():
            api_path = parsed.output_dir / "api" / f"{_cat_slug(category)}.md"
            api_path.parent.mkdir(parents=True, exist_ok=True)
            api_path.write_text(generate_api_page(category, members), encoding="utf-8")
        api_index = parsed.output_dir / "api" / "index.md"
        if api_index.exists():
            replace_generated_block(
                api_index, "modules", generate_api_overview(all_docs)
            )

    if parsed.sidebar and not parsed.dry_run:
        sidebar_path = parsed.output_dir / ".vitepress" / "algorithmsSidebar.ts"
        generate_sidebar_config(all_docs, sidebar_path)

    if parsed.json and not parsed.dry_run:
        json_path = parsed.output_dir / "public" / "optimizers" / "optimizers.json"
        generate_json_metadata(all_docs, json_path)

    if (parsed.griffe or parsed.full_api) and not parsed.dry_run:
        print("\nGenerating full API JSON...")
        generate_full_api_json(parsed.output_dir, verbose=parsed.verbose)

    print(f"\nDone! Generated: {generated}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
