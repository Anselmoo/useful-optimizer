"""Consistency of the documentation with the code it documents.

Checks that every optimizer source file has exactly one docs page, JSON entry and
sidebar link under the one naming rule (slug = file stem), that internal links of
the generated pages resolve, and that every ``from opt... import ...`` shown in
the Markdown docs and README actually works.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import sys

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_docs as gd  # noqa: E402


DOCS = ROOT / "docs"
SKIP_DIRS = {"node_modules", "dist", "cache", ".git"}


def _source_files() -> list[Path]:
    files: list[Path] = []
    for category in gd.CATEGORY_INFO:
        files.extend(
            sorted(
                f
                for f in (ROOT / "src" / "opt" / category).glob("*.py")
                if f.name != "__init__.py"
            )
        )
    return files


def _optimizer_docs() -> list[gd.OptimizerDoc]:
    docs = [gd.parse_optimizer_file(f) for f in _source_files()]
    return [d for d in docs if d is not None]


def _markdown_files() -> list[Path]:
    files = [ROOT / "README.md"]
    files += [
        p for p in DOCS.rglob("*.md") if not SKIP_DIRS & set(p.relative_to(DOCS).parts)
    ]
    return sorted(files)


def _page_path(doc: gd.OptimizerDoc) -> Path:
    return gd.get_output_path(doc, DOCS)


def test_every_source_file_is_an_optimizer() -> None:
    assert len(_optimizer_docs()) == len(_source_files())


def test_slug_is_the_file_stem() -> None:
    for doc in _optimizer_docs():
        assert doc.file_slug == Path(doc.module_path).stem.replace("_", "-")


def test_slugs_are_unique_per_category() -> None:
    seen: set[tuple[str, str]] = set()
    for doc in _optimizer_docs():
        key = (doc.category, doc.file_slug)
        assert key not in seen, f"duplicate slug {key}"
        seen.add(key)


def test_source_pages_json_and_sidebar_agree() -> None:
    docs = _optimizer_docs()
    expected_links = {
        f"/algorithms/{gd.CATEGORY_INFO[d.category]['slug']}/{d.file_slug}"
        for d in docs
    }

    pages = {
        f"/algorithms/{p.parent.name}/{p.stem}"
        for p in (DOCS / "algorithms").glob("*/*.md")
        if p.stem != "index"
    }
    assert pages == expected_links, "docs pages differ from source files"

    data = json.loads((DOCS / "public" / "optimizers" / "optimizers.json").read_text())
    assert data["total_count"] == len(docs)
    assert {o["link"] for o in data["optimizers"]} == expected_links

    sidebar = (DOCS / ".vitepress" / "algorithmsSidebar.ts").read_text()
    sidebar_links = set(re.findall(r'"link": "(/algorithms/[^"]+)"', sidebar))
    leaf_links = {link for link in sidebar_links if not link.endswith("/")}
    assert leaf_links == expected_links, "sidebar differs from source files"


def test_every_category_has_an_index_page() -> None:
    for info in gd.CATEGORY_INFO.values():
        assert (DOCS / "algorithms" / info["slug"] / "index.md").exists()


def test_titles_come_from_metadata() -> None:
    for doc in _optimizer_docs():
        assert doc.name == doc.metadata.get("Algorithm Name"), doc.module_path


LINK_RE = re.compile(r"\]\((/(?:algorithms|api|benchmarks|guide)/[^)#\s]*)")


def _link_target_exists(link: str) -> bool:
    relative = link.strip("/")
    candidates = [
        DOCS / f"{relative}.md",
        DOCS / relative / "index.md",
        DOCS / relative,
    ]
    if not relative:
        candidates = [DOCS / "index.md"]
    return any(c.exists() for c in candidates)


def test_internal_links_in_generated_pages_resolve() -> None:
    broken = []
    for page in (DOCS / "algorithms").glob("**/*.md"):
        broken.extend(
            f"{page.relative_to(DOCS)} -> {link}"
            for link in LINK_RE.findall(page.read_text(encoding="utf-8"))
            if not _link_target_exists(link)
        )
    assert not broken, "\n".join(broken)


FENCE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _opt_imports(markdown: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for block in FENCE_RE.findall(markdown):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue  # intentionally partial snippet
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0] == "opt"
            ):
                found.extend((node.module, alias.name) for alias in node.names)
            elif isinstance(node, ast.Import):
                found.extend(
                    (alias.name, "")
                    for alias in node.names
                    if alias.name.split(".")[0] == "opt"
                )
    return found


def _resolves(module: str, name: str) -> bool:
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return False
    if not name or hasattr(mod, name):
        return True
    try:
        importlib.import_module(f"{module}.{name}")
    except ImportError:
        return False
    return True


@pytest.mark.parametrize(
    "path", _markdown_files(), ids=lambda p: str(p.relative_to(ROOT))
)
def test_documented_imports_resolve(path: Path) -> None:
    broken = [
        f"from {module} import {name}" if name else f"import {module}"
        for module, name in _opt_imports(path.read_text(encoding="utf-8"))
        if not _resolves(module, name)
    ]
    assert not broken, f"{path.relative_to(ROOT)}: " + "; ".join(sorted(set(broken)))


COUNT_RE = re.compile(
    r"\b(\d+)\+?\s+(?:optimization\s+)?(?:algorithms|optimizers)\b", re.IGNORECASE
)


def test_advertised_algorithm_count_matches() -> None:
    total = len(_optimizer_docs())
    files = [
        ROOT / "README.md",
        ROOT / "CLAUDE.md",
        ROOT / "src" / "opt" / "__init__.py",
        DOCS / "index.md",
        *sorted((DOCS / "guide").glob("*.md")),
    ]
    generated = re.compile(
        r"<!-- BEGIN GENERATED.*?<!-- END GENERATED[^>]*-->", re.DOTALL
    )
    wrong = []
    for path in files:
        # Generated blocks cannot drift, and list per-category counts.
        text = generated.sub("", path.read_text(encoding="utf-8"))
        wrong.extend(
            f"{path.relative_to(ROOT)}: says {number}, there are {total}"
            for number in COUNT_RE.findall(text)
            if int(number) > 20 and int(number) != total
        )
    assert not wrong, "\n".join(wrong)
