#!/usr/bin/env python3
"""Test script for documentation generator functionality.

Validates that the generate_docs.py script produces expected outputs.
"""

from __future__ import annotations

import json
import sys

from pathlib import Path


def test_json_metadata() -> bool:  # noqa: PLR0911
    """Test that optimizers.json has correct structure."""
    json_path = Path("docs/public/optimizers/optimizers.json")

    if not json_path.exists():
        print(f"❌ FAIL: {json_path} not found")
        return False

    try:
        data = json.loads(json_path.read_text())
    except json.JSONDecodeError as e:
        print(f"❌ FAIL: Invalid JSON in {json_path}: {e}")
        return False

    # Check required fields
    required_fields = ["version", "generated", "total_count", "optimizers"]
    for field in required_fields:
        if field not in data:
            print(f"❌ FAIL: Missing field '{field}' in {json_path}")
            return False

    # Check optimizer count
    if data["total_count"] != len(data["optimizers"]):
        print(
            f"❌ FAIL: total_count ({data['total_count']}) != actual count ({len(data['optimizers'])})"
        )
        return False

    # Check optimizer structure
    if not data["optimizers"]:
        print("❌ FAIL: No optimizers in metadata")
        return False

    first_opt = data["optimizers"][0]
    required_opt_fields = [
        "name",
        "class_name",
        "category",
        "slug",
        "link",
        "parameters",
    ]
    for field in required_opt_fields:
        if field not in first_opt:
            print(f"❌ FAIL: Missing field '{field}' in optimizer entry")
            return False

    print(f"✅ PASS: optimizers.json valid ({data['total_count']} optimizers)")
    return True


def test_griffe_json_files() -> bool:
    """Test that Griffe JSON files exist for all categories."""
    categories = [
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

    api_dir = Path("docs/api")
    if not api_dir.exists():
        print(f"❌ FAIL: {api_dir} not found")
        return False

    missing = []
    for category in categories:
        json_file = api_dir / f"{category}.json"
        if not json_file.exists():
            missing.append(category)

    if missing:
        print(f"❌ FAIL: Missing Griffe JSON for: {', '.join(missing)}")
        return False

    # Check full API
    full_api = api_dir / "full_api.json"
    if not full_api.exists():
        print(f"❌ FAIL: {full_api} not found")
        return False

    print(
        f"✅ PASS: All Griffe JSON files present ({len(categories)} categories + full_api.json)"
    )
    return True


def test_sidebar_config() -> bool:
    """Test that sidebar configuration exists and is valid."""
    sidebar_path = Path("docs/.vitepress/algorithmsSidebar.ts")

    if not sidebar_path.exists():
        print(f"❌ FAIL: {sidebar_path} not found")
        return False

    content = sidebar_path.read_text()

    # Check that it exports algorithmsSidebar
    if "export const algorithmsSidebar" not in content:
        print("❌ FAIL: Missing 'export const algorithmsSidebar' in sidebar config")
        return False

    # Check that it contains category names
    categories_found = 0
    for category in ["Swarm Intelligence", "Evolutionary", "Gradient-Based"]:
        if category in content:
            categories_found += 1

    if categories_found < 3:
        print("❌ FAIL: Sidebar config missing expected categories")
        return False

    print("✅ PASS: Sidebar configuration valid")
    return True


def test_vitepress_files() -> bool:
    """Test that VitePress integration files exist."""
    files = [
        "docs/.vitepress/loaders/api.data.ts",
        "docs/.vitepress/theme/components/APIDoc.vue",
    ]

    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            print(f"❌ FAIL: {file_path} not found")
            return False

    print("✅ PASS: VitePress integration files present")
    return True


def test_markdown_docs() -> bool:
    """Test that markdown documentation files exist."""
    algo_dir = Path("docs/algorithms")

    if not algo_dir.exists():
        print(f"❌ FAIL: {algo_dir} not found")
        return False

    # Count markdown files
    md_files = list(algo_dir.rglob("*.md"))
    # Filter out index.md files
    md_files = [f for f in md_files if f.name != "index.md"]

    if len(md_files) < 100:  # Should have 117+
        print(f"❌ FAIL: Only {len(md_files)} markdown files found (expected 117+)")
        return False

    print(f"✅ PASS: Markdown documentation present ({len(md_files)} files)")
    return True


def main() -> int:
    """Run all tests and return exit code."""
    print("Testing documentation generator outputs...\n")

    tests = [
        ("JSON Metadata", test_json_metadata),
        ("Griffe JSON Files", test_griffe_json_files),
        ("Sidebar Config", test_sidebar_config),
        ("VitePress Files", test_vitepress_files),
        ("Markdown Docs", test_markdown_docs),
    ]

    results = []
    for name, test_func in tests:
        print(f"\nTest: {name}")
        results.append(test_func())

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
