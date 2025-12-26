#!/usr/bin/env python3
"""Validate benchmark JSON files against the repository schema.

Usage:
    python scripts/validate_benchmark_json.py [--schema PATH] file1.json file2.json ...

If no files are provided, the script will search `benchmarks/output/*.json`.
Exits with non-zero status on validation errors.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path


def _load_schema(schema_path: Path) -> dict:
    with schema_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _validate_file(path: Path, schema: dict) -> bool:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        print(f"[ERROR] Failed to read JSON {path}: {exc}")
        return False

    try:
        from jsonschema import Draft7Validator

        Draft7Validator(schema).validate(data)
    except ModuleNotFoundError:
        print(
            "[WARN] jsonschema not installed; cannot validate schema. Install 'jsonschema' to enable validation."
        )
        return True
    except Exception as exc:
        print(f"[INVALID] {path}: {exc}")
        return False
    else:
        print(f"[OK] {path}")
        return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="JSON files to validate")
    parser.add_argument(
        "--schema",
        default="docs/schemas/benchmark-data-schema.json",
        help="Path to JSON schema",
    )
    args = parser.parse_args(argv)

    schema_path = Path(args.schema)
    if not schema_path.exists():
        print(f"[ERROR] Schema not found at {schema_path}")
        return 2

    schema = _load_schema(schema_path)

    files = (
        [Path(f) for f in args.files]
        if args.files
        else list(Path("benchmarks/output").glob("*.json"))
    )
    if not files:
        print("[INFO] No JSON files found to validate.")
        return 0

    all_ok = True
    for f in files:
        ok = _validate_file(f, schema)
        all_ok = all_ok and ok

    return 0 if all_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
