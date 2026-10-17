<script setup lang="ts">
/**
 * FitnessLandscape3D.vue
 *
 * Interactive 3D fitness landscape (TresJS / Three.js). The surface is drawn
 * over the function's search domain with its height normalised to the plot
 * width, so every benchmark function is readable without manual tuning.
 * An optional trajectory (e.g. the best-so-far path of a 2D run) is overlaid.
 *
 * Interaction: drag rotates, right-drag pans. Scroll-zoom is enabled after a
 * click on the plot, so scrolling the page over the canvas still works.
 */
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { TresCanvas } from '@tresjs/core'
import { OrbitControls } from '@tresjs/cientos'
import * as THREE from 'three'
import * as d3 from 'd3'
import { catppuccinColors } from '../../themes/catppuccin'

interface TrajectoryPoint {
  x: number
  y: number
  z: number
  iteration: number
}

type Range = [number, number]
type HeightScale = 'linear' | 'log'

interface FunctionSpec {
  label: string
  fn: (x: number, y: number) => number
  xRange: Range
  yRange: Range
  /** Heights span orders of magnitude, so default to a log height. */
  defaultScale: HeightScale
}

const FUNCTIONS: Record<string, FunctionSpec> = {
  sphere: {
    label: 'Sphere',
    fn: (x, y) => x * x + y * y,
    xRange: [-5, 5],
    yRange: [-5, 5],
    defaultScale: 'linear'
  },
  rosenbrock: {
    label: 'Rosenbrock',
    fn: (x, y) => 100 * (y - x * x) ** 2 + (1 - x) ** 2,
    xRange: [-2, 2],
    yRange: [-1, 3],
    defaultScale: 'log'
  },
  rastrigin: {
    label: 'Rastrigin',
    fn: (x, y) =>
      20 + (x * x - 10 * Math.cos(2 * Math.PI * x)) + (y * y - 10 * Math.cos(2 * Math.PI * y)),
    xRange: [-5.12, 5.12],
    yRange: [-5.12, 5.12],
    defaultScale: 'linear'
  },
  ackley: {
    label: 'Ackley',
    fn: (x, y) => {
      const a = 20
      const b = 0.2
      const c = 2 * Math.PI
      const sum1 = (x * x + y * y) / 2
      const sum2 = (Math.cos(c * x) + Math.cos(c * y)) / 2
      return -a * Math.exp(-b * Math.sqrt(sum1)) - Math.exp(sum2) + a + Math.E
    },
    xRange: [-5, 5],
    yRange: [-5, 5],
    defaultScale: 'linear'
  },
  griewank: {
    label: 'Griewank',
    fn: (x, y) => (x * x + y * y) / 4000 - Math.cos(x) * Math.cos(y / Math.SQRT2) + 1,
    xRange: [-10, 10],
    yRange: [-10, 10],
    defaultScale: 'linear'
  },
  himmelblau: {
    label: 'Himmelblau',
    fn: (x, y) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2,
    xRange: [-5, 5],
    yRange: [-5, 5],
    defaultScale: 'log'
  }
}

interface Props {
  functionName: string
  xRange?: Range
  yRange?: Range
  resolution?: number
  trajectory?: TrajectoryPoint[]
  colorScale?: 'viridis' | 'turbo' | 'plasma' | 'inferno' | 'catppuccin'
  height?: string | number
}

const props = withDefaults(defineProps<Props>(), {
  resolution: 100,
  colorScale: 'viridis',
  height: 500
})

const emit = defineEmits<{
  'update:functionName': [value: string]
}>()

// The selector works on its own; a parent can still drive it via v-model.
const current = ref(FUNCTIONS[props.functionName] ? props.functionName : 'sphere')
watch(
  () => props.functionName,
  name => {
    if (FUNCTIONS[name]) current.value = name
  }
)

const spec = computed(() => FUNCTIONS[current.value])
const heightScale = ref<HeightScale>(spec.value.defaultScale)
watch(current, () => {
  heightScale.value = spec.value.defaultScale
})

const xRange = computed<Range>(() => props.xRange ?? spec.value.xRange)
const yRange = computed<Range>(() => props.yRange ?? spec.value.yRange)

const chartHeight = computed(() =>
  typeof props.height === 'number' ? `${props.height}px` : props.height
)

/** Plot is 2 units wide; the surface rises at most SURFACE_HEIGHT above y = 0. */
const SURFACE_HEIGHT = 1

const heightTransform = (z: number) =>
  heightScale.value === 'log' ? Math.log10(1 + Math.max(z, 0)) : z

const getColorScale = (scaleType: string, zMin: number, zMax: number) => {
  const interpolator =
    scaleType === 'turbo'
      ? d3.interpolateTurbo
      : scaleType === 'plasma'
        ? d3.interpolatePlasma
        : scaleType === 'inferno'
          ? d3.interpolateInferno
          : scaleType === 'catppuccin'
            ? d3.interpolateRgbBasis([
                catppuccinColors.green,
                catppuccinColors.teal,
                catppuccinColors.blue,
                catppuccinColors.mauve,
                catppuccinColors.red
              ])
            : d3.interpolateViridis
  return d3.scaleSequential(interpolator).domain([zMin, zMax])
}

const surface = computed(() => {
  const [xMin, xMax] = xRange.value
  const [yMin, yMax] = yRange.value
  const res = Math.max(2, props.resolution)

  // PlaneGeometry lies in the XY plane; heights are written to Z and the
  // geometry is then rotated so that Z becomes the (upward) Y axis.
  const geometry = new THREE.PlaneGeometry(2, 2, res - 1, res - 1)
  const positions = geometry.attributes.position
  const heights = new Float32Array(positions.count)

  let hMin = Infinity
  let hMax = -Infinity
  for (let i = 0; i < positions.count; i += 1) {
    const x = xMin + ((positions.getX(i) + 1) / 2) * (xMax - xMin)
    const y = yMin + ((positions.getY(i) + 1) / 2) * (yMax - yMin)
    const h = heightTransform(spec.value.fn(x, y))
    heights[i] = h
    hMin = Math.min(hMin, h)
    hMax = Math.max(hMax, h)
  }

  const span = hMax - hMin || 1
  const colors = new Float32Array(positions.count * 3)
  const colorScale = getColorScale(props.colorScale, hMin, hMax)
  for (let i = 0; i < positions.count; i += 1) {
    positions.setZ(i, ((heights[i] - hMin) / span) * SURFACE_HEIGHT)
    const rgb = d3.rgb(colorScale(heights[i]))
    colors[i * 3] = rgb.r / 255
    colors[i * 3 + 1] = rgb.g / 255
    colors[i * 3 + 2] = rgb.b / 255
  }

  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  geometry.rotateX(-Math.PI / 2)
  geometry.computeVertexNormals()

  return { geometry, hMin, span }
})

watch(surface, (_next, previous) => previous?.geometry.dispose())
onBeforeUnmount(() => surface.value.geometry.dispose())

/** Map a point in search-space coordinates onto the rendered surface. */
const toScene = (p: TrajectoryPoint) => {
  const [xMin, xMax] = xRange.value
  const [yMin, yMax] = yRange.value
  const u = -1 + ((p.x - xMin) / (xMax - xMin)) * 2
  const v = -1 + ((p.y - yMin) / (yMax - yMin)) * 2
  const { hMin, span } = surface.value
  const h = ((heightTransform(p.z) - hMin) / span) * SURFACE_HEIGHT
  return new THREE.Vector3(u, Math.max(h, 0) + 0.02, -v)
}

const trajectoryPoints = computed(() => (props.trajectory ?? []).map(toScene))

const trajectoryGeometry = computed(() =>
  trajectoryPoints.value.length > 1
    ? new THREE.BufferGeometry().setFromPoints(trajectoryPoints.value)
    : null
)
watch(trajectoryGeometry, (_next, previous) => previous?.dispose())
onBeforeUnmount(() => trajectoryGeometry.value?.dispose())

const startPoint = computed(() => trajectoryPoints.value[0] ?? null)
const endPoint = computed(() => trajectoryPoints.value.at(-1) ?? null)

// Scroll-zoom only while the plot is active, so the page keeps scrolling.
const active = ref(false)
const resetKey = ref(0)
const reset = () => {
  resetKey.value += 1
}

const onFunctionChange = (event: Event) => {
  const value = (event.target as HTMLSelectElement).value
  current.value = value
  emit('update:functionName', value)
}

const CAMERA_POSITION: [number, number, number] = [2.3, 1.8, 2.5]
const CAMERA_TARGET: [number, number, number] = [0, 0.35, 0]
</script>

<template>
  <div class="landscape-container">
    <div class="controls">
      <div class="control-group">
        <label>Function:</label>
        <select :value="current" class="select" @change="onFunctionChange">
          <option v-for="(f, key) in FUNCTIONS" :key="key" :value="key">{{ f.label }}</option>
        </select>
        <label>Height:</label>
        <select v-model="heightScale" class="select">
          <option value="linear">Linear</option>
          <option value="log">Logarithmic</option>
        </select>
        <button type="button" class="reset" @click="reset">Reset view</button>
      </div>
      <div class="control-info">
        <span class="hint">
          {{ active
            ? 'Drag to rotate · scroll to zoom · right-drag to pan'
            : 'Drag to rotate · click the plot to enable scroll-zoom' }}
        </span>
      </div>
    </div>

    <div
      class="chart"
      :style="{ height: chartHeight }"
      @pointerdown="active = true"
      @pointerleave="active = false"
    >
      <TresCanvas :key="resetKey" :clear-color="catppuccinColors.base">
        <TresPerspectiveCamera :position="CAMERA_POSITION" :fov="45" :near="0.1" :far="100" />
        <OrbitControls :target="CAMERA_TARGET" :enable-zoom="active" :min-distance="1.2" :max-distance="12" />

        <TresMesh :geometry="surface.geometry">
          <TresMeshPhongMaterial
            :vertex-colors="true"
            :side="THREE.DoubleSide"
            :shininess="60"
            :specular="0x111111"
          />
        </TresMesh>

        <TresLine v-if="trajectoryGeometry" :geometry="trajectoryGeometry">
          <TresLineBasicMaterial :color="catppuccinColors.yellow" />
        </TresLine>

        <TresMesh v-if="startPoint" :position="startPoint">
          <TresSphereGeometry :args="[0.03, 16, 16]" />
          <TresMeshBasicMaterial :color="catppuccinColors.green" />
        </TresMesh>

        <TresMesh v-if="endPoint" :position="endPoint">
          <TresSphereGeometry :args="[0.03, 16, 16]" />
          <TresMeshBasicMaterial :color="catppuccinColors.red" />
        </TresMesh>

        <TresPointLight :position="[3, 4, 3]" :intensity="40" />
        <TresPointLight :position="[-3, 2, -3]" :intensity="20" />
        <TresAmbientLight :intensity="0.6" />

        <TresGridHelper :args="[2.4, 12]" :position="[0, -0.01, 0]" />
      </TresCanvas>
    </div>

    <p class="axes">
      Search domain x ∈ [{{ xRange[0] }}, {{ xRange[1] }}], y ∈ [{{ yRange[0] }}, {{ yRange[1] }}];
      height is f(x, y){{ heightScale === 'log' ? ' on a log10(1 + f) scale' : '' }}, normalised to the
      plot width.
    </p>

    <div v-if="trajectory && trajectory.length" class="trajectory-info">
      <span class="legend-item"><span class="dot green"></span> Start</span>
      <span class="legend-item"><span class="dot red"></span> End (Best)</span>
      <span class="legend-item"><span class="line yellow"></span> Trajectory</span>
    </div>
  </div>
</template>

<style scoped>
.landscape-container {
  background-color: var(--ctp-mocha-surface0, #313244);
  border-radius: 8px;
  padding: 16px;
  margin: 16px 0;
}

.controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  flex-wrap: wrap;
  gap: 12px;
}

.control-group {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.control-group label {
  color: var(--ctp-mocha-text, #cdd6f4);
  font-weight: 500;
}

.select,
.reset {
  background-color: var(--ctp-mocha-surface1, #45475a);
  border: 1px solid var(--ctp-mocha-surface2, #585b70);
  border-radius: 4px;
  color: var(--ctp-mocha-text, #cdd6f4);
  padding: 6px 12px;
  font-size: 0.875rem;
}

.reset {
  cursor: pointer;
}

.select:focus,
.reset:hover {
  outline: none;
  border-color: var(--ctp-mocha-mauve, #cba6f7);
}

.control-info {
  color: var(--ctp-mocha-subtext0, #a6adc8);
  font-size: 0.8rem;
}

.chart {
  position: relative;
  width: 100%;
  min-height: 400px;
  border-radius: 4px;
  overflow: hidden;
  background-color: var(--ctp-mocha-base, #1e1e2e);
}

.axes {
  margin: 8px 0 0;
  color: var(--ctp-mocha-overlay1, #7f849c);
  font-size: 0.75rem;
}

.trajectory-info {
  display: flex;
  gap: 16px;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--ctp-mocha-surface1, #45475a);
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--ctp-mocha-subtext1, #bac2de);
  font-size: 0.875rem;
}

.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.dot.green {
  background-color: var(--ctp-mocha-green, #a6e3a1);
}

.dot.red {
  background-color: var(--ctp-mocha-red, #f38ba8);
}

.line {
  width: 20px;
  height: 3px;
  border-radius: 2px;
}

.line.yellow {
  background-color: var(--ctp-mocha-yellow, #f9e2af);
}
</style>
