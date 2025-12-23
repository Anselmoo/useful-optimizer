#!/usr/bin/env node

/**
 * Documentation Dependency Validation Script
 * 
 * Validates that the documentation migration to KaTeX is successful.
 * Run this script to verify math rendering and dependency health.
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

section('Documentation Migration Validation');

// Check 1: package.json has correct dependencies
section('1. Checking package.json');
try {
  const packagePath = join(docsDir, 'package.json');
  const packageJson = JSON.parse(readFileSync(packagePath, 'utf-8'));
  
  // Check for markdown-it-katex
  if (packageJson.devDependencies['markdown-it-katex']) {
    success(`markdown-it-katex found: ${packageJson.devDependencies['markdown-it-katex']}`);
    checks.packageJson = true;
  } else {
    error('markdown-it-katex not found in devDependencies');
  }
  
  // Check that markdown-it-mathjax3 is removed
  if (!packageJson.devDependencies['markdown-it-mathjax3']) {
    success('markdown-it-mathjax3 correctly removed');
    checks.noLegacyDeps = true;
  } else {
    error('markdown-it-mathjax3 still present - should be removed');
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

// Check 2: VitePress config uses KaTeX
section('2. Checking VitePress Configuration');
try {
  const configPath = join(docsDir, '.vitepress', 'config.ts');
  const configContent = readFileSync(configPath, 'utf-8');
  
  // Check for KaTeX import
  if (configContent.includes('markdown-it-katex')) {
    success('markdown-it-katex import found');
  } else {
    error('markdown-it-katex import missing');
  }
  
  // Check for KaTeX CSS
  if (configContent.includes('katex@') && configContent.includes('.min.css')) {
    success('KaTeX CSS stylesheet configured');
  } else {
    warning('KaTeX CSS might be missing from head configuration');
  }
  
  // Check for plugin configuration
  if (configContent.includes('md.use(markdownItKatex)')) {
    success('KaTeX plugin configured in markdown config');
    checks.viteConfig = true;
  } else {
    error('KaTeX plugin not configured - check markdown.config section');
  }
  
  // Check that old math: true option is removed
  if (configContent.includes('math: true')) {
    warning('Found "math: true" - this option does nothing and should be removed');
  } else {
    success('Legacy "math: true" option removed');
  }
  
} catch (err) {
  error(`Failed to read VitePress config: ${err.message}`);
}

// Check 3: Built output contains KaTeX
section('3. Checking Built Documentation');
try {
  const builtFunctionsPath = join(docsDir, '.vitepress', 'dist', 'benchmarks', 'functions.html');
  
  if (existsSync(builtFunctionsPath)) {
    const htmlContent = readFileSync(builtFunctionsPath, 'utf-8');
    
    // Check for KaTeX classes
    const katexMatches = htmlContent.match(/katex/g);
    if (katexMatches && katexMatches.length > 0) {
      success(`KaTeX rendering detected (${katexMatches.length} instances)`);
      checks.mathRendering = true;
    } else {
      error('No KaTeX rendering found in built HTML');
    }
    
    // Check for MathJax traces (should not exist)
    if (htmlContent.includes('mjx') || htmlContent.includes('mathjax')) {
      error('MathJax traces found - migration incomplete');
    } else {
      success('No MathJax traces found (clean migration)');
    }
    
    // Check for math content
    if (htmlContent.includes('mathbf') || htmlContent.includes('sum')) {
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
  
  if (existsSync(nodeModulesKatex)) {
    success('markdown-it-katex installed in node_modules');
  } else {
    error('markdown-it-katex not found in node_modules - run npm install');
  }
  
  if (!existsSync(nodeModulesMathjax)) {
    success('markdown-it-mathjax3 not in node_modules (correctly removed)');
  } else {
    warning('markdown-it-mathjax3 still in node_modules - run: rm -rf node_modules && npm install');
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
  success('Migration to KaTeX is complete and working correctly.');
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
    info('1. Verify package.json has "markdown-it-katex": "^2.0.3"');
  }
  if (!checks.viteConfig) {
    info('2. Check .vitepress/config.ts for KaTeX configuration');
  }
  if (!checks.noLegacyDeps) {
    info('3. Remove markdown-it-mathjax3 from package.json');
    info('4. Run: rm -rf node_modules package-lock.json && npm install');
  }
  process.exit(1);
}
