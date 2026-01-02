# Documentation Generator Usage

This guide explains how to use the enhanced documentation generator with Griffe integration.

## Overview

The `scripts/generate_docs.py` script now supports:
- Markdown documentation generation (existing)
- JSON metadata export for VitePress
- Griffe-based API documentation
- Automatic sidebar configuration

## Quick Start

### Generate Everything

```bash
# From project root
uv run python scripts/generate_docs.py --all --json --griffe --full-api --sidebar

# Or from docs directory via npm
cd docs && npm run docs:api:all
```

### Generate Specific Components

**Markdown documentation only:**
```bash
uv run python scripts/generate_docs.py --all
```

**JSON metadata only:**
```bash
uv run python scripts/generate_docs.py --all --json
```

**Griffe API docs (per category):**
```bash
uv run python scripts/generate_docs.py --griffe
```

**Full API documentation:**
```bash
uv run python scripts/generate_docs.py --full-api
```

**Sidebar configuration:**
```bash
uv run python scripts/generate_docs.py --all --sidebar
```

### Generate for Specific Category

```bash
uv run python scripts/generate_docs.py --category swarm_intelligence --json --griffe
```

## Output Files

### 1. Markdown Documentation
- **Location**: `docs/algorithms/{category}/{algorithm}.md`
- **Count**: 117 files (one per optimizer)
- **Format**: VitePress-compatible markdown with frontmatter

### 2. JSON Metadata
- **Location**: `docs/public/optimizers/optimizers.json`
- **Size**: ~135KB
- **Contains**: All optimizer metadata for Vue components
- **Structure**:
  ```json
  {
    "version": "1.0.0",
    "total_count": 117,
    "optimizers": [...]
  }
  ```

### 3. Griffe API Documentation
- **Per-category**: `docs/api/{category}.json` (10 files)
- **Full API**: `docs/api/full_api.json`
- **Size**: ~4MB each
- **Format**: Griffe JSON output with class/function signatures

### 4. Sidebar Configuration
- **Location**: `docs/.vitepress/algorithmsSidebar.ts`
- **Format**: TypeScript export for VitePress config
- **Auto-generated**: Navigation structure for all algorithms

## VitePress Integration

### Data Loader

The VitePress data loader transforms Griffe JSON into structured data:

```typescript
// docs/.vitepress/loaders/api.data.ts
import { data } from './loaders/api.data'

// Access in Vue components
data.modules.swarm_intelligence.classes
```

### APIDoc Component

Render optimizer documentation in Vue components:

```vue
<script setup>
import { data } from '@/loaders/api.data'
import APIDoc from '@/components/APIDoc.vue'

const particleSwarm = data.modules.swarm_intelligence.classes
  .find(c => c.name === 'ParticleSwarm')
</script>

<template>
  <APIDoc :classDoc="particleSwarm" />
</template>
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--all` | Process all optimizer files |
| `--file PATH` | Process specific file |
| `--category NAME` | Process specific category |
| `--output-dir DIR` | Output directory (default: `docs`) |
| `--dry-run` | Preview without writing files |
| `--verbose` | Show detailed progress |
| `--sidebar` | Generate sidebar config |
| `--clean` | Remove old docs before generating |
| `--json` | Generate JSON metadata |
| `--griffe` | Generate Griffe API docs per category |
| `--full-api` | Generate full API JSON |

## Performance

- **Full generation**: ~6 seconds
- **Markdown only**: ~4 seconds
- **JSON only**: ~2 seconds
- **Griffe per-category**: ~4 seconds
- **Full API**: ~2 seconds

## Troubleshooting

### Griffe not found
```bash
# Install Griffe
uv add griffe --group dev
uv sync
```

### VitePress build fails
```bash
# Install npm dependencies
cd docs
npm install
npm run docs:build
```

### Missing JSON files
```bash
# Regenerate all JSON
uv run python scripts/generate_docs.py --all --json --griffe --full-api
```

## Development Workflow

1. **Update optimizer docstrings** following Google style
2. **Generate documentation**:
   ```bash
   uv run python scripts/generate_docs.py --all --json --griffe --full-api --sidebar
   ```
3. **Test VitePress**:
   ```bash
   cd docs
   npm run docs:dev  # Preview locally
   npm run docs:build  # Production build
   ```
4. **Commit changes** including generated files

## Files Modified by Generator

### Always Generated
- `docs/algorithms/**/*.md` - Algorithm pages
- `docs/public/optimizers/optimizers.json` - Metadata

### With `--sidebar`
- `docs/.vitepress/algorithmsSidebar.ts`

### With `--griffe`
- `docs/api/{category}.json` (10 files)

### With `--full-api`
- `docs/api/full_api.json`

## CI/CD Integration

Add to GitHub Actions workflow:

```yaml
- name: Generate Documentation
  run: |
    uv run python scripts/generate_docs.py --all --json --griffe --full-api --sidebar
    
- name: Build VitePress
  run: |
    cd docs
    npm install
    npm run docs:build
```

## See Also

- [VitePress Documentation](https://vitepress.dev/)
- [Griffe Documentation](https://mkdocstrings.github.io/griffe/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
