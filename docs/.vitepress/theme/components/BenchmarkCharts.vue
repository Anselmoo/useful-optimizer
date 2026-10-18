<script setup lang="ts">
/**
 * BenchmarkCharts.vue
 *
 * Benchmark results of one optimizer on every algorithm page. Loads the
 * published data (see utils/benchmarkData.ts) and shows, for one function and
 * dimension, tabs for convergence, final-error distribution, runtime ECDF,
 * scaling with dimension, statistics against the baseline, dispersion of the
 * evaluated points and the path through the 2D landscape.
 *
 * Usage:
 *   <ClientOnly><BenchmarkCharts algorithm="WhaleOptimizationAlgorithm" /></ClientOnly>
 */
import { computed, onMounted, ref, shallowRef, watch } from 'vue'
import { withBase } from 'vitepress'
import type { Cell, Index, Ranks, Trajectory } from '../types/benchmark'
import { catppuccinColors } from '../../themes/catppuccin'
import { algorithmColors } from '../utils/useEChart'
import { loadCell, loadIndex, loadRanks, loadTrajectory } from '../utils/benchmarkData'
import {
  convergenceSeries,
  dispersionSeries,
  ecdfSeries,
  formatNumber,
  scalingSeries,
  statsRows,
  violinSeries,
  type ScalingMetric
} from '../utils/benchmarkTransforms'
import ConvergenceChart from './ConvergenceChart.vue'
import ECDFChart from './ECDFChart.vue'
import ViolinPlot from './ViolinPlot.vue'
import LineChart from './LineChart.vue'
import StatsTable from './StatsTable.vue'
import FitnessLandscape3D from './FitnessLandscape3D.vue'

interface Props {
  /** Class name of the optimizer, e.g. "ParticleSwarm". */
  algorithm: string
  /** Initial function; defaults to the first one of the protocol. */
  functionName?: string
  /** Initial dimension; defaults to 10 when available. */
  dimension?: number
  /** Optimizers to compare with; defaults to the baseline plus the best ranked. */
  compareWith?: string[]
}

const props = defineProps<Props>()

const TABS = [
  { id: 'convergence', label: 'Convergence' },
  { id: 'distribution', label: 'Distribution' },
  { id: 'ecdf', label: 'ECDF' },
  { id: 'scaling', label: 'Scaling' },
  { id: 'statistics', label: 'Statistics' },
  { id: 'dispersion', label: 'Dispersion' },
  { id: 'landscape', label: 'Landscape (2D)' }
] as const
type TabId = (typeof TABS)[number]['id']

const MAX_COMPARED = 7
const DEFAULT_RIVALS = 3

const index = ref<Index | null>(null)
const ranks = ref<Ranks | null>(null)
const cell = shallowRef<Cell | null>(null)
const cells = shallowRef<Record<number, Cell>>({})
const trajectory = shallowRef<Trajectory | null>(null)

const error = ref<string | null>(null)
const cellError = ref<string | null>(null)
const loading = ref(true)

const tab = ref<TabId>('convergence')
const functionName = ref('')
const dim = ref(0)
const compared = ref<string[]>([])
const edited = ref(false)
const targetIndex = ref(0)
const scalingMetric = ref<ScalingMetric>('median')
const pathOwner = ref(props.algorithm)

const baseline = computed(() => index.value?.protocol.baseline ?? '')
const covered = computed(() => index.value?.optimizers.some(o => o.name === props.algorithm) ?? false)

const functions = computed(() => index.value?.protocol.functions ?? [])
const functionInfo = computed(() => functions.value.find(f => f.name === functionName.value))

const dimensions = computed(() =>
  (index.value?.cells ?? [])
    .filter(c => c.function === functionName.value)
    .map(c => c.dim)
    .sort((a, b) => a - b)
)

const names = computed(() => [props.algorithm, ...compared.value.filter(n => n !== props.algorithm)])

const colorByName = computed(() => {
  const map: Record<string, string> = {}
  let next = 1 // 0 is reserved for the page's own optimizer
  map[props.algorithm] = algorithmColors[0]
  if (baseline.value) map[baseline.value] = catppuccinColors.overlay2
  for (const name of names.value) {
    if (!map[name]) {
      map[name] = algorithmColors[next % algorithmColors.length]
      next += 1
    }
  }
  return map
})
const colors = computed(() => names.value.map(n => colorByName.value[n]))

const rivals = computed(() => {
  const best = (ranks.value?.entries ?? [])
    .map(e => e.name)
    .filter(n => n !== props.algorithm && n !== baseline.value)
    .slice(0, DEFAULT_RIVALS)
  return [baseline.value, ...best].filter(Boolean)
})

const addable = computed(() => {
  const groups = new Map<string, string[]>()
  for (const o of index.value?.optimizers ?? []) {
    if (names.value.includes(o.name)) continue
    groups.set(o.category, [...(groups.get(o.category) ?? []), o.name])
  }
  return [...groups.entries()].sort(([a], [b]) => a.localeCompare(b))
})

const label = (text: string) => text.replace(/_/g, ' ')

const targets = computed(() => index.value?.protocol.targets ?? [])
const target = computed(() => targets.value[targetIndex.value] ?? 1e-3)

const missing = computed(() =>
  cell.value ? names.value.filter(n => !cell.value!.optimizers[n]) : []
)
const own = computed(() => cell.value?.optimizers[props.algorithm])
const truncatedRuns = computed(() => own.value?.truncated ?? 0)

const convergence = computed(() => (cell.value ? convergenceSeries(cell.value, names.value) : []))
const distribution = computed(() => (cell.value ? violinSeries(cell.value, names.value) : []))
const ecdf = computed(() => (cell.value ? ecdfSeries(cell.value, names.value) : []))
const dispersion = computed(() => (cell.value ? dispersionSeries(cell.value, names.value) : []))
const stats = computed(() => (cell.value ? statsRows(cell.value, names.value, targetIndex.value) : []))
const scaling = computed(() =>
  scalingSeries(cells.value, names.value, scalingMetric.value, targetIndex.value)
)

/** Series colors in the order of the series that actually have data. */
const colorsFor = (series: { algorithm: string }[]) => series.map(s => colorByName.value[s.algorithm])

const SCALING = {
  median: { y: 'Median precision f − f*', log: true, title: 'Final precision vs dimension' },
  ert: { y: 'ERT (evaluations)', log: true, title: 'Expected running time vs dimension' },
  time: { y: 'µs per evaluation', log: true, title: 'Wall-clock cost vs dimension' }
} as const

const landscapeFunctions = ['sphere', 'rosenbrock', 'rastrigin', 'ackley', 'shifted_ackley', 'griewank']
const pathNames = computed(() => Object.keys(trajectory.value?.optimizers ?? {}).sort())
const path = computed(() => {
  const rows = trajectory.value?.optimizers[pathOwner.value] ?? []
  return rows.map(([iteration, x, y, z]) => ({ iteration, x, y, z }))
})

const initial = (data: Index) => {
  const fn = props.functionName && data.protocol.functions.some(f => f.name === props.functionName)
    ? props.functionName
    : data.protocol.functions[0]?.name ?? ''
  functionName.value = fn
  const available = data.cells.filter(c => c.function === fn).map(c => c.dim).sort((a, b) => a - b)
  dim.value = props.dimension && available.includes(props.dimension)
    ? props.dimension
    : available.includes(10) ? 10 : available[0] ?? 0
  const preferred = data.protocol.targets.findIndex(t => t === 1e-3)
  targetIndex.value = preferred >= 0 ? preferred : 0
  compared.value = props.compareWith ?? [data.protocol.baseline]
  edited.value = props.compareWith !== undefined
}

onMounted(async () => {
  try {
    const data = await loadIndex()
    index.value = data
    if (data.optimizers.some(o => o.name === props.algorithm)) initial(data)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
  // The ranks only refine the default comparison; the charts do not wait for them.
  try {
    ranks.value = await loadRanks()
  } catch {
    // keep the baseline-only default
  }
})

watch(ranks, () => {
  if (!edited.value && index.value) compared.value = rivals.value
})

watch([functionName, dim], async ([fn, d]) => {
  if (!fn || !d) return
  cellError.value = null
  try {
    cell.value = await loadCell(fn, d)
  } catch (e) {
    cell.value = null
    cellError.value = e instanceof Error ? e.message : String(e)
  }
}, { immediate: true })

// Dimensions differ between functions only if a cell is missing; keep a valid one.
watch(dimensions, list => {
  if (list.length > 0 && !list.includes(dim.value)) dim.value = list.includes(10) ? 10 : list[0]
})

watch([tab, functionName], async ([id, fn]) => {
  if (!fn) return
  if (id === 'scaling') {
    const loaded: Record<number, Cell> = {}
    await Promise.all(dimensions.value.map(async d => {
      try {
        loaded[d] = await loadCell(fn, d)
      } catch {
        // a missing dimension just leaves a gap in the curve
      }
    }))
    cells.value = loaded
  }
  if (id === 'landscape') {
    trajectory.value = null
    try {
      trajectory.value = await loadTrajectory(fn)
      if (!trajectory.value.optimizers[pathOwner.value]) pathOwner.value = props.algorithm
    } catch {
      trajectory.value = null
    }
  }
})

const addCompared = (event: Event) => {
  const select = event.target as HTMLSelectElement
  if (select.value && names.value.length < MAX_COMPARED + 1) {
    compared.value = [...compared.value, select.value]
    edited.value = true
  }
  select.value = ''
}

const removeCompared = (name: string) => {
  compared.value = compared.value.filter(n => n !== name)
  edited.value = true
}

const resetCompared = () => {
  compared.value = rivals.value
  edited.value = false
}

const selectTab = (event: KeyboardEvent) => {
  const at = TABS.findIndex(t => t.id === tab.value)
  const step = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0
  if (step) tab.value = TABS[(at + step + TABS.length) % TABS.length].id
}

const evaluationBudget = computed(() => (cell.value ? cell.value.budget.toLocaleString() : ''))
</script>

<template>
  <div class="bench">
    <p v-if="loading" class="bench__state">Loading benchmark data…</p>

    <div v-else-if="error" class="bench__state bench__state--error">
      <strong>Benchmark data is not available.</strong>
      <span>{{ error }}</span>
    </div>

    <p v-else-if="!covered" class="bench__state">
      {{ algorithm }} is not part of the single-objective benchmark suite, so there are no results to show.
    </p>

    <template v-else>
      <div class="bench__controls">
        <label>
          Function
          <select v-model="functionName">
            <option v-for="f in functions" :key="f.name" :value="f.name">{{ label(f.name) }}</option>
          </select>
        </label>
        <label v-if="tab !== 'scaling' && tab !== 'landscape'">
          Dimension
          <select v-model.number="dim">
            <option v-for="d in dimensions" :key="d" :value="d">{{ d }}</option>
          </select>
        </label>
        <label v-if="tab === 'statistics' || (tab === 'scaling' && scalingMetric === 'ert')">
          Target
          <select v-model.number="targetIndex">
            <option v-for="(t, i) in targets" :key="t" :value="i">{{ formatNumber(t, 1) }}</option>
          </select>
        </label>
        <label v-if="tab === 'scaling'">
          Metric
          <select v-model="scalingMetric">
            <option value="median">Median precision</option>
            <option value="ert">ERT</option>
            <option value="time">Time per evaluation</option>
          </select>
        </label>
      </div>

      <div v-if="tab !== 'landscape'" class="bench__compare" aria-label="Optimizers compared">
        <span class="chip chip--own" :style="{ borderColor: colorByName[algorithm] }">{{ algorithm }}</span>
        <span
          v-for="name in compared"
          :key="name"
          class="chip"
          :style="{ borderColor: colorByName[name] }"
        >
          {{ name }}<span v-if="name === baseline" class="chip__tag">baseline</span>
          <button type="button" :aria-label="`Remove ${name}`" @click="removeCompared(name)">×</button>
        </span>
        <select
          v-if="addable.length && names.length < MAX_COMPARED + 1"
          aria-label="Add an optimizer to compare"
          @change="addCompared"
        >
          <option value="">Compare with…</option>
          <optgroup v-for="[category, list] in addable" :key="category" :label="label(category)">
            <option v-for="name in list" :key="name" :value="name">{{ name }}</option>
          </optgroup>
        </select>
        <button type="button" class="link" @click="resetCompared">Reset</button>
      </div>

      <div class="bench__tabs" role="tablist" @keydown="selectTab">
        <button
          v-for="t in TABS"
          :id="`bench-tab-${t.id}`"
          :key="t.id"
          type="button"
          role="tab"
          :aria-selected="tab === t.id"
          :tabindex="tab === t.id ? 0 : -1"
          :class="{ active: tab === t.id }"
          @click="tab = t.id"
        >
          {{ t.label }}
        </button>
      </div>

      <div role="tabpanel" :aria-labelledby="`bench-tab-${tab}`">
        <p v-if="cellError && tab !== 'landscape' && tab !== 'scaling'" class="bench__state bench__state--error">
          {{ cellError }}
        </p>

        <template v-else>
          <p v-if="cell && missing.length && tab !== 'landscape'" class="bench__note">
            No result for {{ missing.join(', ') }} on {{ label(functionName) }} in {{ dim }}D.
          </p>
          <p v-if="truncatedRuns && tab !== 'landscape'" class="bench__note">
            {{ truncatedRuns }} of {{ own?.runs_planned }} runs of {{ algorithm }} were cut off by the
            wall-clock cap; they count with the best value found until then.
          </p>

          <ConvergenceChart
            v-if="tab === 'convergence' && convergence.length"
            :data="convergence"
            :colors="colorsFor(convergence)"
            :title="`${label(functionName)}, ${dim}D — convergence`"
            x-axis-label="Function evaluations / dimension"
            y-axis-label="Precision f − f*"
            band-label="interquartile range of the runs"
          />

          <ViolinPlot
            v-else-if="tab === 'distribution' && distribution.length"
            :data="distribution"
            :colors="colorsFor(distribution)"
            :title="`${label(functionName)}, ${dim}D — final precision`"
            y-axis-label="Final precision f − f*"
          />

          <ECDFChart
            v-else-if="tab === 'ecdf' && ecdf.length"
            :data="ecdf"
            :colors="colorsFor(ecdf)"
            :title="`${label(functionName)}, ${dim}D — runtime ECDF`"
            :target-precisions="targets"
          />

          <LineChart
            v-else-if="tab === 'scaling' && scaling.length"
            :data="scaling"
            :colors="colorsFor(scaling)"
            :title="`${label(functionName)} — ${SCALING[scalingMetric].title}`"
            x-axis-label="Dimension"
            :y-axis-label="SCALING[scalingMetric].y"
            log-x
            :log-y="SCALING[scalingMetric].log"
            show-symbols
            :description="scalingMetric === 'ert'
              ? `Expected running time to reach f − f* ≤ ${formatNumber(target, 1)}. Optimizers that never reach it have no point.`
              : 'Every point is the median over the independent runs of one dimension.'"
          />

          <StatsTable
            v-else-if="tab === 'statistics' && stats.length"
            :rows="stats"
            :baseline="baseline"
            :target="target"
            :filename="`${algorithm}-${functionName}-${dim}d`"
          />

          <LineChart
            v-else-if="tab === 'dispersion' && dispersion.length"
            :data="dispersion"
            :colors="colorsFor(dispersion)"
            :title="`${label(functionName)}, ${dim}D — dispersion of evaluated points`"
            x-axis-label="Function evaluations / dimension"
            y-axis-label="Normalised dispersion"
            log-x
            description="Spread of the most recently evaluated points relative to the search domain, a proxy for how widely the optimizer is sampling. It is not the diversity of a population."
          />

          <template v-else-if="tab === 'landscape'">
            <template v-if="trajectory && landscapeFunctions.includes(functionName)">
              <div class="bench__controls">
                <label>
                  Path of
                  <select v-model="pathOwner">
                    <option v-for="name in pathNames" :key="name" :value="name">{{ name }}</option>
                  </select>
                </label>
              </div>
              <FitnessLandscape3D
                :key="functionName"
                :function-name="functionName"
                :x-range="[functionInfo?.lower ?? -5, functionInfo?.upper ?? 5]"
                :y-range="[functionInfo?.lower ?? -5, functionInfo?.upper ?? 5]"
                :trajectory="path"
                lock-function
              />
              <p class="bench__note">
                Best-so-far path of the first run on the 2D function, from its first evaluation to the end of
                the budget.
              </p>
            </template>
            <p v-else class="bench__state">No 2D landscape data for {{ label(functionName) }}.</p>
          </template>

          <p v-else-if="cell" class="bench__state">No data for this selection.</p>
          <p v-else class="bench__state">Loading…</p>
        </template>
      </div>

      <p v-if="cell && index" class="bench__footer">
        {{ evaluationBudget }} evaluations per run ({{ index.protocol.budget_per_dim.toLocaleString() }} × dimension),
        {{ index.protocol.n_runs }} independent runs, protocol and statistics are described in the
        <a :href="withBase('/benchmarks/methodology')">methodology</a>.
      </p>
    </template>
  </div>
</template>

<style scoped>
.bench {
  margin: 16px 0;
}

.bench__state {
  padding: 16px;
  border-radius: 8px;
  background-color: var(--ctp-mocha-surface0, #313244);
  color: var(--ctp-mocha-subtext1, #bac2de);
}

.bench__state--error {
  display: flex;
  flex-direction: column;
  gap: 4px;
  color: var(--ctp-mocha-red, #f38ba8);
}

.bench__controls,
.bench__compare {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px 16px;
  margin-bottom: 12px;
  color: var(--ctp-mocha-text, #cdd6f4);
  font-size: 0.875rem;
}

.bench__compare {
  gap: 8px;
}

select {
  background-color: var(--ctp-mocha-surface1, #45475a);
  border: 1px solid var(--ctp-mocha-surface2, #585b70);
  border-radius: 4px;
  color: var(--ctp-mocha-text, #cdd6f4);
  padding: 4px 8px;
  font-size: 0.875rem;
}

select:focus-visible,
button:focus-visible {
  outline: 2px solid var(--ctp-mocha-mauve, #cba6f7);
  outline-offset: 1px;
}

.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 2px 10px;
  border: 2px solid;
  border-radius: 999px;
  background-color: var(--ctp-mocha-surface0, #313244);
  font-size: 0.8125rem;
}

.chip__tag {
  color: var(--ctp-mocha-overlay1, #7f849c);
  font-size: 0.6875rem;
  text-transform: uppercase;
}

.chip button {
  all: unset;
  cursor: pointer;
  color: var(--ctp-mocha-overlay1, #7f849c);
  line-height: 1;
}

.chip button:hover {
  color: var(--ctp-mocha-red, #f38ba8);
}

.link {
  all: unset;
  cursor: pointer;
  color: var(--ctp-mocha-blue, #89b4fa);
  font-size: 0.8125rem;
}

.bench__tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  border-bottom: 1px solid var(--ctp-mocha-surface1, #45475a);
}

.bench__tabs button {
  all: unset;
  cursor: pointer;
  padding: 8px 14px;
  border-bottom: 2px solid transparent;
  color: var(--ctp-mocha-subtext0, #a6adc8);
  font-size: 0.875rem;
}

.bench__tabs button:hover {
  color: var(--ctp-mocha-text, #cdd6f4);
}

.bench__tabs button.active {
  border-bottom-color: var(--ctp-mocha-mauve, #cba6f7);
  color: var(--ctp-mocha-mauve, #cba6f7);
}

.bench__tabs button:focus-visible {
  outline: 2px solid var(--ctp-mocha-mauve, #cba6f7);
}

.bench__note,
.bench__footer {
  margin: 8px 0;
  color: var(--ctp-mocha-overlay1, #7f849c);
  font-size: 0.75rem;
}
</style>
