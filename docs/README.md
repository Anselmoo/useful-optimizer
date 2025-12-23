# useful-optimizer Documentation

VitePress documentation site for the useful-optimizer library.

## Quick Start

### Installation

```bash
cd docs
npm install
```

**Note**: Installation no longer requires `--legacy-peer-deps` flag. Dependencies have been updated for compatibility.

### Development

Start the local development server:

```bash
npm run docs:dev
```

The site will be available at `http://localhost:5173/useful-optimizer/`

**Important**: The site uses a base path of `/useful-optimizer/` for GitHub Pages deployment. This means:
- Links in markdown should use absolute paths: `/guide/installation` not `./installation`
- The dev server will serve content at `/useful-optimizer/` not `/`

### Building

Build the production site:

```bash
npm run docs:build
```

Output will be in `.vitepress/dist/`

### Preview

Preview the production build locally:

```bash
npm run docs:preview
```

## Project Structure

```
docs/
├── .vitepress/
│   ├── config.ts              # VitePress configuration
│   ├── theme/
│   │   ├── index.ts           # Custom theme entry
│   │   └── style.css          # Catppuccin Mocha theme
│   └── components/            # Vue components
├── algorithms/                # Algorithm documentation
│   ├── swarm-intelligence/    # 56+ swarm algorithms
│   ├── evolutionary/          # Evolutionary algorithms
│   ├── gradient-based/        # Gradient optimizers
│   ├── classical/             # Classical methods
│   ├── metaheuristic/         # Metaheuristic algorithms
│   ├── constrained/           # Constrained optimization
│   ├── probabilistic/         # Probabilistic methods
│   ├── multi-objective/       # Multi-objective optimization
│   ├── physics-inspired/      # Physics-based algorithms
│   └── social-inspired/       # Social behavior algorithms
├── api/                       # API reference docs
├── benchmarks/                # Benchmark documentation
├── guide/                     # Getting started guides
├── public/                    # Static assets
│   ├── favicon.ico
│   └── logo.svg
├── index.md                   # Landing page
├── package.json
└── README.md                  # This file
```

## Theme

The site uses the **Catppuccin Mocha** color scheme, a soothing pastel theme with:
- Base: `#1e1e2e` (dark background)
- Mauve: `#cba6f7` (primary brand color)
- Blue: `#89b4fa` (links, secondary brand)
- Full color palette in `.vitepress/theme/style.css`

## Features

### KaTeX Math Rendering

LaTeX equations are supported via KaTeX (fast, lightweight math renderer):

**Inline math**: `$f(x) = x^2$`

**Block math**:
```markdown
$$
\min_{x \in \mathbb{R}^n} f(x) = \sum_{i=1}^{n} x_i^2
$$
```

### Local Search

Built-in VitePress search is enabled. Press `Ctrl+K` or `Cmd+K` to search.

### Code Highlighting

Syntax highlighting with line numbers is enabled for all code blocks.

## Dependencies

- **vitepress**: `^1.5.0` - Documentation framework
- **vue**: `^3.4.0` - Required by VitePress
- **echarts**: `^6.0.0` - Charting library (actively maintained)
- **vue-echarts**: `^8.0.0` - Vue wrapper for ECharts (actively maintained)
- **@catppuccin/vitepress**: `^0.1.2` - Catppuccin theme support
- **markdown-it-katex**: `^2.0.3` - Fast LaTeX math rendering with KaTeX

### Dependency Notes

- **Math rendering**: Migrated from `markdown-it-mathjax3` to `markdown-it-katex` for better performance, smaller bundle size, and active maintenance.
- **Charts**: Using modern `echarts` v6.0.0 and `vue-echarts` v8.0.0, both actively maintained packages.
- **Note**: `echarts-gl` was removed from the documentation dependencies due to lack of maintenance and incompatibility with recent ECharts releases (ECharts v6+).

### TODO: 3D visualization (non-visible)

- **Goal**: Provide 3D benchmark visualizations without shipping `echarts-gl` in the docs build.
  - Consider server-side rendering or an export pipeline to produce PNG/SVG/GLTF assets for the docs.
  - Alternatively, implement an optional, lazily-loaded 3D viewer (Three.js / Deck.gl / Plotly) that is not required for the default docs build.
- **Tracking**: Create an issue `docs: implement non-visible 3D visualization pipeline` to track effort, tests, and compatibility requirements.
- **CI**: Add a docs-specific npm check to catch dependency/peer-dependency issues early (see workflow).
- **vue-echarts v7**: Compatible with echarts@5. Version 8 requires echarts@6.

## Dead Links

The configuration currently sets `ignoreDeadLinks: true` because not all algorithm pages have been created yet. This allows the build to succeed while pages are being developed.

**To check for dead links**, temporarily set `ignoreDeadLinks: false` in `.vitepress/config.ts` and run:

```bash
npm run docs:build
```

The build will fail and list all dead links that need to be addressed.

## Contributing

### Adding Algorithm Pages

1. Create a new markdown file in the appropriate category directory
2. Follow the template structure of existing algorithm pages
3. Include: description, mathematical formulation, parameters, usage example
4. Reference benchmark functions from `/benchmarks/functions`

### Adding API Documentation

1. Create markdown files in `api/` directory
2. Use consistent structure: overview, interface, examples
3. Cross-reference with algorithm pages

## Testing

### Validate Migration

To verify the documentation migration to KaTeX is working correctly:

```bash
npm run validate:migration
```

This runs a comprehensive validation script that checks:
- Correct dependencies in package.json
- VitePress configuration for KaTeX
- Built output contains KaTeX rendering
- No MathJax traces remain

### Full Integration Test

To run a complete test including clean install and build:

```bash
./test-docs-migration.sh
```

This script:
1. Performs a clean npm install (without `--legacy-peer-deps`)
2. Validates all dependencies
3. Builds the documentation
4. Verifies KaTeX rendering in output
5. Runs the validation script

## CI/CD

A GitHub Actions workflow builds and deploys the documentation to GitHub Pages on pushes to the main branch.

The workflow includes:
- Dependency installation validation (no legacy flags)
- Migration validation check
- Documentation build
- Deployment to GitHub Pages

## Troubleshooting

### Build Failures

If the build fails:
1. Check for syntax errors in markdown files
2. Verify all internal links are valid
3. Ensure code blocks have valid syntax
4. Check that frontmatter is properly formatted

### Dev Server Issues

If the dev server shows 404 errors:
- Remember the base path is `/useful-optimizer/`
- Use absolute paths in links: `/guide/` not `./guide/`
- Restart the dev server if hot reload isn't working

### Dependency Conflicts

If `npm install` fails:
- Ensure you're not using `--legacy-peer-deps` (no longer needed after KaTeX migration)
- Delete `node_modules/` and `package-lock.json`
- Run `npm install` again
- Run `npm run validate:migration` to verify the setup

## License

Documentation is released under the MIT License, same as the useful-optimizer library.
