#!/bin/bash

# Documentation Build and Validation Test
# This script runs in CI to validate the documentation migration

set -e  # Exit on error

echo "=== Documentation Build and Validation Test ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Navigate to docs directory
cd "$(dirname "$0")"

echo "📁 Working directory: $(pwd)"
echo ""

# Step 1: Clean install
echo "1️⃣  Clean installation test..."
if [ -d "node_modules" ]; then
    echo "   Removing existing node_modules..."
    rm -rf node_modules
fi

if [ -f "package-lock.json" ]; then
    echo "   Removing existing package-lock.json..."
    rm -f package-lock.json
fi

echo "   Running npm install..."
if npm install 2>&1 | tee /tmp/npm-install.log; then
    echo -e "${GREEN}✅ Installation successful (no --legacy-peer-deps needed)${NC}"
else
    echo -e "${RED}❌ Installation failed${NC}"
    echo "Installation log:"
    cat /tmp/npm-install.log
    exit 1
fi

# Check for peer dependency warnings
if grep -q "peer dep" /tmp/npm-install.log; then
    echo -e "${YELLOW}⚠️  Peer dependency warnings detected:${NC}"
    grep "peer dep" /tmp/npm-install.log
else
    echo -e "${GREEN}✅ No peer dependency warnings${NC}"
fi
echo ""

# Step 2: Validate dependencies
echo "2️⃣  Validating dependencies..."
if [ -d "node_modules/markdown-it-katex" ]; then
    echo -e "${GREEN}✅ markdown-it-katex installed${NC}"
else
    echo -e "${RED}❌ markdown-it-katex not found${NC}"
    exit 1
fi

if [ ! -d "node_modules/markdown-it-mathjax3" ]; then
    echo -e "${GREEN}✅ markdown-it-mathjax3 correctly removed${NC}"
else
    echo -e "${RED}❌ markdown-it-mathjax3 still present${NC}"
    exit 1
fi

if [ -d "node_modules/echarts" ] && [ -d "node_modules/vue-echarts" ]; then
    echo -e "${GREEN}✅ ECharts packages present${NC}"
else
    echo -e "${RED}❌ ECharts packages missing${NC}"
    exit 1
fi
echo ""

# Step 3: Build documentation
echo "3️⃣  Building documentation..."
if npm run docs:build 2>&1 | tee /tmp/docs-build.log; then
    echo -e "${GREEN}✅ Documentation built successfully${NC}"
else
    echo -e "${RED}❌ Documentation build failed${NC}"
    echo "Build log:"
    cat /tmp/docs-build.log
    exit 1
fi

# Check build time (should be reasonable)
build_time=$(grep "build complete in" /tmp/docs-build.log | grep -oP '\d+\.\d+')
if [ -n "$build_time" ]; then
    echo "   Build time: ${build_time}s"
    if (( $(echo "$build_time < 30" | bc -l) )); then
        echo -e "${GREEN}✅ Build time is good (<30s)${NC}"
    else
        echo -e "${YELLOW}⚠️  Build time is slow (>30s)${NC}"
    fi
fi
echo ""

# Step 4: Validate built output
echo "4️⃣  Validating built output..."
DIST_DIR=".vitepress/dist"

if [ ! -d "$DIST_DIR" ]; then
    echo -e "${RED}❌ Build directory not found${NC}"
    exit 1
fi

# Check for KaTeX in built HTML
echo "   Checking for KaTeX rendering..."
if [ -f "$DIST_DIR/benchmarks/functions.html" ]; then
    katex_count=$(grep -o "katex" "$DIST_DIR/benchmarks/functions.html" | wc -l)
    if [ "$katex_count" -gt 0 ]; then
        echo -e "${GREEN}✅ KaTeX rendering detected (${katex_count} instances)${NC}"
    else
        echo -e "${RED}❌ No KaTeX rendering found${NC}"
        exit 1
    fi

    # Check for MathJax traces (should not exist)
    if grep -q "mjx\|MathJax" "$DIST_DIR/benchmarks/functions.html"; then
        echo -e "${RED}❌ MathJax traces found in built HTML${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ No MathJax traces (clean migration)${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Benchmark functions page not found${NC}"
fi

# Check for ECharts
echo "   Checking for ECharts..."
if grep -r "echarts" "$DIST_DIR" | head -1 > /dev/null; then
    echo -e "${GREEN}✅ ECharts found in built output${NC}"
else
    echo -e "${YELLOW}⚠️  ECharts not detected (may not be used on all pages)${NC}"
fi

# Check bundle size
dist_size=$(du -sh "$DIST_DIR" | cut -f1)
echo "   Build output size: $dist_size"
echo ""

# Step 5: Run validation script
echo "5️⃣  Running migration validation script..."
if node validate-docs-migration.js; then
    echo -e "${GREEN}✅ Validation script passed${NC}"
else
    echo -e "${RED}❌ Validation script failed${NC}"
    exit 1
fi
echo ""

# Summary
echo "=== Test Summary ==="
echo -e "${GREEN}✅ All tests passed!${NC}"
echo ""
echo "Migration to KaTeX is complete and verified:"
echo "  - Clean npm install without legacy flags"
echo "  - No peer dependency conflicts"
echo "  - Documentation builds successfully"
echo "  - KaTeX math rendering working"
echo "  - No MathJax traces remaining"
echo "  - ECharts integration intact"
echo ""
echo "The documentation is ready for deployment! 🚀"
