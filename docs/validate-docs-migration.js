#!/usr/bin/env node

/**
 * Documentation Dependency Validation Script
 *
 * Validates that the documentation math rendering uses VitePress' built-in
 * MathJax support (markdown-it-mathjax3) rather than the unmaintained
 * markdown-it-katex plugin, whose markup was incompatible with modern KaTeX
 * CSS and rendered sub/superscripts incorrectly.
 *
 * Usage:
 *   node validate-docs-migration.js
 */

import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const docsDir = __dirname;

// ANSI color codes
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  bold: '\x1b[1m',
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function success(message) {
  log(`✅ ${message}`, 'green');
}

function error(message) {
  log(`❌ ${message}`, 'red');
}

function warning(message) {
  log(`⚠️  ${message}`, 'yellow');
}

function info(message) {
  log(`ℹ️  ${message}`, 'blue');
}

function section(title) {
  log(`\n${colors.bold}=== ${title} ===${colors.reset}`, 'blue');
}

// Validation checks
const checks = {
  packageJson: false,
  viteConfig: false,
  mathRendering: false,
  noLegacyDeps: false,
  echartsVersion: false,
};

section('Documentation Math Rendering Validation');

// Check 1: package.json has correct dependencies
section('1. Checking package.json');
try {
  const packagePath = join(docsDir, 'package.json');
  const packageJson = JSON.parse(readFileSync(packagePath, 'utf-8'));

  // Check for markdown-it-mathjax3
  if (packageJson.devDependencies['markdown-it-mathjax3']) {
    success(`markdown-it-mathjax3 found: ${packageJson.devDependencies['markdown-it-mathjax3']}`);
    checks.packageJson = true;
  } else {
    error('markdown-it-mathjax3 not found in devDependencies');
  }

  // Check that the legacy markdown-it-katex is removed
  if (!packageJson.devDependencies['markdown-it-katex']) {
    success('legacy markdown-it-katex correctly removed');
    checks.noLegacyDeps = true;
  } else {
    error('markdown-it-katex still present - should be removed');
  }

  // Verify ECharts versions
  const echartsVersion = packageJson.devDependencies['echarts'];
  const vueEchartsVersion = packageJson.devDependencies['vue-echarts'];

  if (echartsVersion && vueEchartsVersion) {
    success(`echarts: ${echartsVersion}`);
    success(`vue-echarts: ${vueEchartsVersion}`);
    checks.echartsVersion = true;
  } else {
    error('ECharts packages missing');
  }

} catch (err) {
  error(`Failed to read package.json: ${err.message}`);
}

// Check 2: VitePress config uses built-in MathJax
section('2. Checking VitePress Configuration');
try {
  const configPath = join(docsDir, '.vitepress', 'config.ts');
  const configContent = readFileSync(configPath, 'utf-8');

  // Check for the built-in math option
  if (/math:\s*true/.test(configContent)) {
    success('Built-in "math: true" option configured');
    checks.viteConfig = true;
  } else {
    error('"math: true" not found - enable VitePress built-in MathJax in markdown config');
  }

  // Check that the legacy KaTeX plugin is gone
  if (!configContent.includes('markdown-it-katex')) {
    success('No legacy markdown-it-katex import');
  } else {
    error('markdown-it-katex import still present - should be removed');
  }

  // Check that the stale KaTeX CDN stylesheet is gone
  if (!(configContent.includes('katex@') && configContent.includes('.min.css'))) {
    success('Legacy KaTeX CDN stylesheet removed');
  } else {
    warning('Found a KaTeX CDN stylesheet link - no longer needed with MathJax');
  }

} catch (err) {
  error(`Failed to read VitePress config: ${err.message}`);
}

// Check 3: Built output contains MathJax
section('3. Checking Built Documentation');
try {
  const builtFunctionsPath = join(docsDir, '.vitepress', 'dist', 'benchmarks', 'functions.html');

  if (existsSync(builtFunctionsPath)) {
    const htmlContent = readFileSync(builtFunctionsPath, 'utf-8');

    // Check for MathJax containers
    const mathjaxMatches = htmlContent.match(/mjx-container/g);
    if (mathjaxMatches && mathjaxMatches.length > 0) {
      success(`MathJax rendering detected (${mathjaxMatches.length} instances)`);
      checks.mathRendering = true;
    } else {
      error('No MathJax rendering found in built HTML');
    }

    // Check for legacy KaTeX traces (should not exist)
    if (htmlContent.includes('class="katex')) {
      error('Legacy KaTeX traces found - migration incomplete');
    } else {
      success('No legacy KaTeX traces found (clean migration)');
    }

    // Check for math content
    if (htmlContent.includes('MathJax') || htmlContent.includes('<mjx-')) {
      success('Mathematical content found in HTML');
    } else {
      warning('No mathematical content detected - might need to verify');
    }
  } else {
    warning('Built documentation not found. Run "npm run docs:build" first.');
    info(`Expected path: ${builtFunctionsPath}`);
  }
} catch (err) {
  warning(`Could not verify built output: ${err.message}`);
  info('Run "npm run docs:build" to generate documentation');
}

// Check 4: node_modules integrity
section('4. Checking Node Modules');
try {
  const nodeModulesKatex = join(docsDir, 'node_modules', 'markdown-it-katex');
  const nodeModulesMathjax = join(docsDir, 'node_modules', 'markdown-it-mathjax3');

  if (existsSync(nodeModulesMathjax)) {
    success('markdown-it-mathjax3 installed in node_modules');
  } else {
    error('markdown-it-mathjax3 not found in node_modules - run npm install');
  }

  if (!existsSync(nodeModulesKatex)) {
    success('markdown-it-katex not in node_modules (correctly removed)');
  } else {
    warning('markdown-it-katex still in node_modules - run: rm -rf node_modules && npm install');
  }
} catch (err) {
  warning(`Could not verify node_modules: ${err.message}`);
}

// Summary
section('Validation Summary');
const totalChecks = Object.keys(checks).length;
const passedChecks = Object.values(checks).filter(Boolean).length;
const percentage = Math.round((passedChecks / totalChecks) * 100);

console.log(`\nPassed: ${passedChecks}/${totalChecks} (${percentage}%)\n`);

if (passedChecks === totalChecks) {
  success('All critical checks passed! ✨');
  success('Math rendering via built-in MathJax is complete and working correctly.');
  console.log('\nNext steps:');
  info('1. Run "npm run docs:dev" to test locally');
  info('2. Run "npm run docs:build" to verify production build');
  info('3. Commit and push changes if everything looks good');
  process.exit(0);
} else {
  error('Some checks failed. Please review the output above.');
  console.log('\nFailed checks:');
  Object.entries(checks).forEach(([check, passed]) => {
    if (!passed) {
      error(`  - ${check}`);
    }
  });
  console.log('\nRecommended actions:');
  if (!checks.packageJson) {
    info('1. Verify package.json has "markdown-it-mathjax3": "^4"');
  }
  if (!checks.viteConfig) {
    info('2. Set markdown: { math: true } in .vitepress/config.ts');
  }
  if (!checks.noLegacyDeps) {
    info('3. Remove markdown-it-katex from package.json');
    info('4. Run: rm -rf node_modules package-lock.json && npm install');
  }
  process.exit(1);
}
