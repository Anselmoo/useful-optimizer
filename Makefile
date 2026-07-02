.PHONY: docs docs-generate docs-snapshot

# Regenerate all VitePress docs from source, then refresh the rrt artifact
# fingerprint lock so `rrt-artifacts-check` stays green. Run this after editing
# optimizer docstrings or publishing new benchmark data.
docs: docs-generate docs-snapshot

# Regenerate the Griffe API JSON, per-algorithm Markdown pages (with embedded
# benchmark charts) and the sidebar config from the Python source.
docs-generate:
	uv run python scripts/generate_docs.py --all --json --griffe --full-api --sidebar

# Refresh .rrt/artifacts.lock.toml to match the freshly generated artifacts.
docs-snapshot:
	pre-commit run rrt-artifacts-snapshot --hook-stage manual --all-files
