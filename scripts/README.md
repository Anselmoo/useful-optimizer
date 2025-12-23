# Batch Docstring Update Script

## Overview

The `batch_update_docstrings.py` script automates the generation of COCO/BBOB-compliant docstring templates for all optimizer classes in the useful-optimizer library. This eliminates the error-prone and time-consuming manual process of updating 117+ optimizer docstrings.

## Features

- ✅ **AST Parsing**: Extracts class names, parameters, and existing docstrings using Python's AST module
- ✅ **Template Generation**: Generates COCO/BBOB-compliant templates with FIXME markers for manual completion
- ✅ **Category Detection**: Auto-detects optimizer category from directory structure
- ✅ **Skip Abstract Classes**: Automatically excludes `abstract_*.py` files
- ✅ **Dry Run Mode**: Preview changes before writing to files
- ✅ **File Writing**: Automatically writes generated templates to optimizer files (when not in dry-run mode)
- ✅ **Multi-Objective Support**: Handles both single-objective and multi-objective optimizers
- ✅ **High Performance**: Processes 117 files in ~0.15 seconds

## Usage

### Basic Commands

```bash
# Preview changes for all optimizers (recommended first step)
uv run python scripts/batch_update_docstrings.py --dry-run

# Process specific category only
uv run python scripts/batch_update_docstrings.py --category swarm_intelligence

# Process all optimizers (writes templates to files - CAUTION!)
uv run python scripts/batch_update_docstrings.py

# Show help
uv run python scripts/batch_update_docstrings.py --help
```

### Command-Line Options

- `--dry-run`: Preview changes without writing to files (recommended for first run)
- `--category {category}`: Process only the specified category
  - Available categories: `classical`, `constrained`, `evolutionary`, `gradient_based`, `metaheuristic`, `multi_objective`, `physics_inspired`, `probabilistic`, `social_inspired`, `swarm_intelligence`

## Output Example

```
🔍 Found 117 optimizer files to process

📝 Generated BBOB template for ParticleSwarm
   File: opt/swarm_intelligence/particle_swarm.py
   Category: swarm_intelligence
   Parameters: func, lower_bound, upper_bound, dim, population_size, max_iter, seed
   Action: ✅ Docstring template written to file
   ⚠️  Note: Review and complete FIXME markers in the generated template.
   ⚠️  Note: This is a template. Manual review and completion needed.

...

======================================================================
✅ Successfully processed 117 optimizer files

📊 Summary:
   Total files scanned: 117
   Successfully processed: 117
   Failed: 0

⚠️  Templates generated with FIXME markers. Manual review required!
   See .github/prompts/optimizer-docs-template.md for guidance.
```

## Implementation Details

### How It Works

1. **File Discovery**: Scans optimizer directories and finds all Python files
2. **AST Parsing**: Uses Python's `ast` module to extract:
   - Class names
   - Base classes (AbstractOptimizer or AbstractMultiObjectiveOptimizer)
   - `__init__` parameters
   - Existing docstrings
3. **Category Detection**: Extracts category from parent directory name
4. **Template Generation**: Creates COCO/BBOB-compliant templates with:
   - 11 required sections (Algorithm Metadata, Mathematical Formulation, etc.)
   - FIXME markers for manual completion
   - Algorithm-specific parameter listings
   - Proper return types for single/multi-objective optimizers
5. **Output**: Displays processing results with actionable information

### Key Functions

- `extract_optimizer_info(filepath)`: Parse file with AST and extract class information
- `generate_bbob_docstring_template(info)`: Create compliant template with FIXME markers
- `find_optimizer_files(base_path, category)`: Locate all optimizer files to process
- `process_optimizer(filepath, dry_run)`: Process a single optimizer file
- `main(argv)`: Entry point that orchestrates the batch processing

### File Structure

```
scripts/
  └── batch_update_docstrings.py    # Main script
opt/
  ├── classical/                     # 9 optimizers
  ├── constrained/                   # 5 optimizers
  ├── evolutionary/                  # 6 optimizers
  ├── gradient_based/                # 11 optimizers
  ├── metaheuristic/                 # 14 optimizers
  ├── multi_objective/               # 3 optimizers
  ├── physics_inspired/              # 4 optimizers
  ├── probabilistic/                 # 5 optimizers
  ├── social_inspired/               # 4 optimizers
  └── swarm_intelligence/            # 56 optimizers
  └── test/
      └── test_batch_update_docstrings.py  # Comprehensive tests
```

## Template Sections

The script generates templates with all 11 COCO/BBOB-required sections:

1. **Algorithm Metadata**: Name, acronym, year, authors, class, complexity, properties
2. **Mathematical Formulation**: Core equations, constraint handling
3. **Hyperparameters**: Default values, BBOB recommendations, sensitivity analysis
4. **COCO/BBOB Benchmark Settings**: Search space, evaluation budget, performance metrics
5. **Example**: Working doctest examples with `seed=42`
6. **Args**: All parameters with BBOB guidance
7. **Attributes**: All instance variables including `self.seed`
8. **Methods**: `search()` method signature and documentation
9. **References**: Citations with DOI and COCO data archive links
10. **See Also**: Related algorithms with BBOB comparisons
11. **Notes**: Complexity, performance characteristics, convergence properties, reproducibility

## FIXME Markers

The script adds FIXME markers for sections that require manual completion:

- Algorithm name, acronym, year, authors
- Mathematical formulation equations
- Hyperparameter tables and sensitivity analysis
- Algorithm-specific parameters
- Performance characteristics
- References and citations
- Related algorithms

## Testing

The script includes comprehensive test coverage:

```bash
# Run all tests
uv run pytest opt/test/test_batch_update_docstrings.py -v

# Test coverage includes:
# - AST parsing (basic and multi-objective)
# - Template generation
# - File discovery
# - Real optimizer extraction
# - BBOB compliance verification
```

## Performance

- **Processing Time**: ~0.157 seconds for 117 files
- **Throughput**: ~745 files/second
- **Performance Target**: < 5 seconds ✅ (31x faster than requirement)

## Next Steps After Running Script

1. Review the generated templates with FIXME markers in the modified optimizer files
2. Complete algorithm-specific information:
   - Fill in metadata (year, authors, acronym)
   - Add mathematical formulation
   - Document hyperparameters
   - Add performance characteristics
   - Include references
3. Validate generated docstrings:
   - Run doctests: `uv run python -m doctest opt/[category]/[module].py`
   - Verify formatting: `uv run ruff check opt/`
4. Refer to `.github/prompts/optimizer-docs-template.md` for detailed guidance

**Note:** The script writes templates directly to optimizer files when run without `--dry-run`. 
Always use `--dry-run` first to preview changes before applying them to avoid accidental overwrites.

## Troubleshooting

### Common Issues

**Issue**: Script can't find opt directory  
**Solution**: Run from repository root: `cd /path/to/useful-optimizer`

**Issue**: Import errors  
**Solution**: Ensure virtual environment is activated: `uv sync`

**Issue**: AST parsing fails for a file  
**Solution**: Check file for syntax errors: `uv run python -m py_compile opt/[category]/[file].py`

## Dependencies

- Python 3.10+
- Standard library only (no external dependencies)
  - `ast`: AST parsing
  - `argparse`: Command-line interface
  - `pathlib`: File path handling
  - `sys`: System utilities

## Contributing

When modifying the script:

1. Update tests in `opt/test/test_batch_update_docstrings.py`
2. Run linter: `uv run ruff check scripts/batch_update_docstrings.py`
3. Verify performance: `time uv run python scripts/batch_update_docstrings.py --dry-run`
4. Test on real optimizers: `uv run python scripts/batch_update_docstrings.py --category classical --dry-run`

## Related Documentation

- [COCO/BBOB Template](../.github/prompts/optimizer-docs-template.md): Complete docstring template guide
- [COCO Platform](https://coco-platform.org/): Official COCO/BBOB documentation
- [BBOB Test Suite](https://coco-platform.org/testsuites/bbob/overview.html): Benchmark suite overview

## License

MIT License - See repository root for details.
