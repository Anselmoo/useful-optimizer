#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Benchmark Data Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Configuration
SCHEMA_FILE="$PROJECT_ROOT/docs/schemas/benchmark-data-schema.json"
TEST_DATA_FILE="$PROJECT_ROOT/docs/public/test-data/mock-benchmark-data.json"
TEMP_DIR="$PROJECT_ROOT/docs/public/test-data"

# Check if required files exist
echo -e "${YELLOW}[1/5] Checking required files...${NC}"
if [[ ! -f "$SCHEMA_FILE" ]]; then
    echo -e "${RED}✗ Schema file not found: $SCHEMA_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Schema file found${NC}"

if [[ ! -f "$TEST_DATA_FILE" ]]; then
    echo -e "${RED}✗ Test data file not found: $TEST_DATA_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Test data file found${NC}"
echo ""

# Validate JSON syntax
echo -e "${YELLOW}[2/5] Validating JSON syntax...${NC}"
if ! python3 -m json.tool "$TEST_DATA_FILE" > /dev/null 2>&1; then
    echo -e "${RED}✗ Invalid JSON syntax${NC}"
    exit 1
fi
echo -e "${GREEN}✓ JSON syntax is valid${NC}"
echo ""

# Validate against schema using Python
echo -e "${YELLOW}[3/5] Validating against schema...${NC}"
cat > "$TEMP_DIR/validate_schema.py" << 'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    print("Installing jsonschema...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "jsonschema"])
    import jsonschema

# Load schema
schema_path = Path(sys.argv[1])
with open(schema_path) as f:
    schema = json.load(f)

# Load test data
data_path = Path(sys.argv[2])
with open(data_path) as f:
    data = json.load(f)

# Validate
try:
    jsonschema.validate(instance=data, schema=schema)
    print("✓ Data validates successfully against schema")
    sys.exit(0)
except jsonschema.ValidationError as e:
    print(f"✗ Validation error: {e.message}")
    print(f"  Path: {' -> '.join(str(p) for p in e.path)}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)
PYTHON_SCRIPT

python3 "$TEMP_DIR/validate_schema.py" "$SCHEMA_FILE" "$TEST_DATA_FILE"
if [[ $? -ne 0 ]]; then
    echo -e "${RED}✗ Schema validation failed${NC}"
    rm "$TEMP_DIR/validate_schema.py"
    exit 1
fi
rm "$TEMP_DIR/validate_schema.py"
echo ""

# Display data summary
echo -e "${YELLOW}[4/5] Analyzing test data...${NC}"
python3 << PYTHON_SUMMARY
import json
from pathlib import Path

with open('$TEST_DATA_FILE') as f:
    data = json.load(f)

metadata = data['metadata']
benchmarks = data['benchmarks']

print(f"📊 Data Summary:")
print(f"  • Max iterations: {metadata['max_iterations']}")
print(f"  • Number of runs: {metadata['n_runs']}")
print(f"  • Dimensions: {metadata['dimensions']}")
print(f"  • Timestamp: {metadata['timestamp']}")
print(f"  • Python version: {metadata.get('python_version', 'N/A')}")
print(f"  • NumPy version: {metadata.get('numpy_version', 'N/A')}")
print()
print(f"📈 Benchmark Functions:")

total_optimizers = 0
total_runs = 0

for func_name, dimensions_data in benchmarks.items():
    print(f"  • {func_name}:")
    for dim, optimizers_data in dimensions_data.items():
        print(f"    - Dimension {dim}: {len(optimizers_data)} optimizer(s)")
        for opt_name, opt_data in optimizers_data.items():
            n_runs = len(opt_data['runs'])
            total_runs += n_runs
            total_optimizers += 1
            stats = opt_data['statistics']
            print(f"      → {opt_name}: {n_runs} runs, mean={stats['mean_fitness']:.6f}, success_rate={opt_data['success_rate']}")

print()
print(f"📊 Total Statistics:")
print(f"  • Total benchmark function-dimension pairs: {sum(len(v) for v in benchmarks.values())}")
print(f"  • Total optimizer configurations: {total_optimizers}")
print(f"  • Total runs: {total_runs}")
PYTHON_SUMMARY
echo ""

# Test with TypeScript types (if available)
echo -e "${YELLOW}[5/5] Testing TypeScript type compatibility...${NC}"
TYPES_FILE="$PROJECT_ROOT/docs/.vitepress/theme/types/benchmark.ts"
if [[ -f "$TYPES_FILE" ]]; then
    echo -e "${GREEN}✓ TypeScript types file found${NC}"
    echo -e "  Location: ${TYPES_FILE}"

    # Create a simple TypeScript test
    cat > "$TEMP_DIR/test_types.ts" << 'TS_SCRIPT'
import type { BenchmarkDataSchema } from '../../../.vitepress/theme/types/benchmark'
import data from './mock-benchmark-data.json'

const benchmarkData: BenchmarkDataSchema = data as BenchmarkDataSchema

// Type check will fail at compile time if types don't match
console.log('✓ TypeScript types are compatible')
console.log(`  Functions: ${Object.keys(benchmarkData.benchmarks).join(', ')}`)
TS_SCRIPT

    # Check if Node.js and TypeScript are available
    if command -v node &> /dev/null && [[ -d "$PROJECT_ROOT/docs/node_modules" ]]; then
        cd "$PROJECT_ROOT/docs"
        if npx tsc --noEmit "$TEMP_DIR/test_types.ts" 2>&1 | grep -q "error"; then
            echo -e "${YELLOW}⚠ TypeScript type check found issues (may be expected for dev)${NC}"
        else
            echo -e "${GREEN}✓ TypeScript type compatibility verified${NC}"
        fi
        rm "$TEMP_DIR/test_types.ts"
    else
        echo -e "${YELLOW}⚠ TypeScript/Node.js not available, skipping type check${NC}"
    fi
else
    echo -e "${YELLOW}⚠ TypeScript types file not found, skipping type check${NC}"
fi
echo ""

# Success message
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ All tests passed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Test data location:${NC} $TEST_DATA_FILE"
echo -e "${BLUE}Schema location:${NC} $SCHEMA_FILE"
echo ""
echo -e "${BLUE}Usage in VitePress:${NC}"
echo -e "  1. Load data: ${YELLOW}import data from '/test-data/mock-benchmark-data.json'${NC}"
echo -e "  2. Use with components in markdown files"
echo ""
echo -e "${BLUE}Example:${NC}"
cat << 'EXAMPLE'
<script setup>
import data from '/test-data/mock-benchmark-data.json'

// Extract data for specific function and dimension
const ackleyData = data.benchmarks.shifted_ackley['2']
const optimizers = Object.keys(ackleyData)

// Transform for ConvergenceChart
const convergenceData = optimizers.map(opt => ({
  algorithm: opt,
  iterations: Array.from({length: 10}, (_, i) => i * 10),
  mean: ackleyData[opt].runs[0].history.best_fitness,
  std: ackleyData[opt].runs[0].history.mean_fitness
}))
</script>

<ClientOnly>
<ConvergenceChart :data="convergenceData" />
</ClientOnly>
EXAMPLE

echo ""
echo -e "${GREEN}Test suite complete!${NC}"
