"""Export the JSON schema and the TypeScript types of the published benchmark data.

The Pydantic models in :mod:`benchmarks.models` are the source of truth. This script
writes ``docs/schemas/benchmark-data-schema.json`` from them and, with ``--types``,
regenerates ``docs/.vitepress/theme/types/benchmark.ts`` through the docs site's
``json-schema-to-typescript`` (needs ``npm ci`` in ``docs/``).

    uv run python benchmarks/export_schema.py --types
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.models import BenchmarkBundle  # noqa: E402


SCHEMA_PATH = _ROOT / "docs" / "schemas" / "benchmark-data-schema.json"
TYPES_PATH = _ROOT / "docs" / ".vitepress" / "theme" / "types" / "benchmark.ts"
JSON2TS = _ROOT / "docs" / "node_modules" / ".bin" / "json2ts"


def build_schema() -> dict:
    """Return the JSON schema of every published file, as one document."""
    schema = BenchmarkBundle.model_json_schema()
    schema["title"] = "BenchmarkBundle"
    schema["description"] = (
        "Published benchmark data, schema version 2. Generated from "
        "benchmarks/models.py by benchmarks/export_schema.py; do not edit."
    )
    return schema


def schema_text() -> str:
    """The schema serialised exactly as it is committed."""
    return json.dumps(build_schema(), indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the schema, and optionally the TypeScript types."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--types", action="store_true", help="also regenerate benchmark.ts"
    )
    args = parser.parse_args(argv)

    SCHEMA_PATH.write_text(schema_text(), encoding="utf-8")
    print(f"Wrote {SCHEMA_PATH.relative_to(_ROOT)}")

    if args.types:
        if not JSON2TS.exists():
            print(
                "error: run `npm ci` in docs/ first (json2ts missing)", file=sys.stderr
            )
            return 1
        subprocess.run(  # noqa: S603
            [
                str(JSON2TS),
                "-i",
                str(SCHEMA_PATH),
                "-o",
                str(TYPES_PATH),
                "--additionalProperties=false",
                "--bannerComment",
                "/* Generated from benchmarks/models.py by benchmarks/export_schema.py; do not edit. */",
            ],
            check=True,
        )
        print(f"Wrote {TYPES_PATH.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
