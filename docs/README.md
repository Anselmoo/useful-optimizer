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
│   ├── swarm-intelligence/    # 56 swarm algorithms
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

### Math Rendering

LaTeX equations are rendered with MathJax (`markdown-it-mathjax3`):

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

The authoritative list is `package.json`; the packages that shape the site are:

- **vitepress** - Documentation framework
- **echarts** (v5) - 2D benchmark charts, wrapped by `theme/utils/useEChart.ts`
- **three**, **@tresjs/core**, **@tresjs/cientos** - 3D fitness landscape
- **d3** - Colour scales
- **markdown-it-mathjax3** - LaTeX math rendering

### Dependency Notes

- **Charts**: ECharts is used directly through the `useEChart` composable (`vue-echarts` is not a dependency).
- **3D**: `echarts-gl` is not used (unmaintained); the landscape is rendered with TresJS / Three.js.
- **Math**: rendered with MathJax through `markdown-it-mathjax3` (see `validate:migration`).

## Dead Links

`ignoreDeadLinks` is off: `npm run docs:build` fails and lists every dead link, so a broken
link blocks CI instead of reaching the site.

## Contributing

### Adding Algorithm Pages

Algorithm pages, category indexes, API pages and the sidebar are generated from the optimizer
source files; do not edit them by hand. Add or change the optimizer (and its docstring) and run

```bash
uv run python scripts/generate_docs.py --all --json --sidebar
```

See the [contributing guide](./contributing.md) for the naming rules.

## Testing

### Validate Migration

To verify the math rendering setup (MathJax) is working:

```bash
npm run validate:migration
```

This validation script checks the dependencies in `package.json`, the VitePress configuration and
that the built output contains rendered MathJax.

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
- `.npmrc` sets `legacy-peer-deps=true`, which hides peer-dependency conflicts; remove it temporarily to see them
- Delete `node_modules/` and `package-lock.json`
- Run `npm install` again
- Run `npm run validate:migration` to verify the setup

## License

Documentation is released under the MIT License, same as the useful-optimizer library.
