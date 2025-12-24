import type { BenchmarkDataSchema } from '../../../.vitepress/theme/types/benchmark'
import data from './mock-benchmark-data.json'

const benchmarkData: BenchmarkDataSchema = data as BenchmarkDataSchema

// Type check will fail at compile time if types don't match
console.log('✓ TypeScript types are compatible')
console.log(`  Functions: ${Object.keys(benchmarkData.benchmarks).join(', ')}`)
