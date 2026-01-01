<script setup lang="ts">
/**
 * FitnessLandscape3D.vue
 *
 * Interactive 3D fitness landscape visualization using TresJS (Vue wrapper for Three.js).
 * Replaces deprecated ECharts-GL with modern, maintained library.
 * Uses PlaneGeometry for surface plots with D3 color scales.
 */
import { computed, ref, watch } from 'vue'
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

interface Props {
  functionName: string
  xRange?: [number, number]
  yRange?: [number, number]
  resolution?: number
  trajectory?: TrajectoryPoint[]
  colorScale?: 'viridis' | 'turbo' | 'plasma' | 'inferno' | 'catppuccin'
  height?: string | number
}

const props = withDefaults(defineProps<Props>(), {
  xRange: () => [-5, 5],
  yRange: () => [-5, 5],
  resolution: 100,
  colorScale: 'viridis',
  height: 500
})

const emit = defineEmits<{
  'update:functionName': [value: string]
}>()

const chartHeight = computed(() => {
  return typeof props.height === 'number' ? `${props.height}px` : props.height
})

// Benchmark function implementations
const benchmarkFunctions: Record<string, (x: number, y: number) => number> = {
  sphere: (x, y) => x * x + y * y,

  rosenbrock: (x, y) => {
    return 100 * Math.pow(y - x * x, 2) + Math.pow(1 - x, 2)
  },

  rastrigin: (x, y) => {
    return 20 + (x * x - 10 * Math.cos(2 * Math.PI * x)) +
           (y * y - 10 * Math.cos(2 * Math.PI * y))
  },

  ackley: (x, y) => {
    const a = 20
    const b = 0.2
    const c = 2 * Math.PI
    const d = 2

    const sum1 = (x * x + y * y) / d
    const sum2 = (Math.cos(c * x) + Math.cos(c * y)) / d

    return -a * Math.exp(-b * Math.sqrt(sum1)) - Math.exp(sum2) + a + Math.E
  },

  griewank: (x, y) => {
    const sum = (x * x + y * y) / 4000
    const prod = Math.cos(x) * Math.cos(y / Math.sqrt(2))
    return sum - prod + 1
  },

  himmelblau: (x, y) => {
    return Math.pow(x * x + y - 11, 2) + Math.pow(x + y * y - 7, 2)
  }
}

// Evaluate function at (x, y)
const evaluateFunction = (x: number, y: number, functionName: string): number => {
  const func = benchmarkFunctions[functionName] || benchmarkFunctions.sphere
  return func(x, y)
}

// Generate surface geometry with colors
const surfaceGeometry = computed(() => {
  const [xMin, xMax] = props.xRange
  const [yMin, yMax] = props.yRange
  const res = props.resolution

  const geometry = new THREE.PlaneGeometry(2, 2, res - 1, res - 1)
  const positions = geometry.attributes.position
  const colors = new Float32Array(positions.count * 3)

  let zMin = Infinity
  let zMax = -Infinity

  // First pass: compute Z values and find min/max
  for (let i = 0; i < positions.count; i++) {
    const u = positions.getX(i) // [-1, 1]
    const v = positions.getY(i) // [-1, 1]

    const x = xMin + (u + 1) / 2 * (xMax - xMin)
    const y = yMin + (v + 1) / 2 * (yMax - yMin)
    const z = evaluateFunction(x, y, props.functionName)

    positions.setZ(i, z)
    zMin = Math.min(zMin, z)
    zMax = Math.max(zMax, z)
  }

  // Second pass: apply colors based on Z values
  const colorScale = getColorScale(props.colorScale, zMin, zMax)

  for (let i = 0; i < positions.count; i++) {
    const z = positions.getZ(i)
    const colorValue = d3.color(colorScale(z))
    if (colorValue) {
      const rgb = colorValue.rgb()
      colors[i * 3] = rgb.r / 255
      colors[i * 3 + 1] = rgb.g / 255
      colors[i * 3 + 2] = rgb.b / 255
    }
  }

  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  geometry.computeVertexNormals()

  return geometry
})

// Get D3 color scale
const getColorScale = (scaleType: string, zMin: number, zMax: number) => {
  switch (scaleType) {
    case 'viridis':
      return d3.scaleSequential(d3.interpolateViridis).domain([zMin, zMax])
    case 'turbo':
      return d3.scaleSequential(d3.interpolateTurbo).domain([zMin, zMax])
    case 'plasma':
      return d3.scaleSequential(d3.interpolatePlasma).domain([zMin, zMax])
    case 'inferno':
      return d3.scaleSequential(d3.interpolateInferno).domain([zMin, zMax])
    case 'catppuccin':
      return d3.scaleSequential(
        d3.interpolateRgbBasis([
          catppuccinColors.green,
          catppuccinColors.teal,
          catppuccinColors.blue,
          catppuccinColors.mauve,
          catppuccinColors.red
        ])
      ).domain([zMin, zMax])
    default:
      return d3.scaleSequential(d3.interpolateViridis).domain([zMin, zMax])
  }
}

// Generate trajectory line geometry
const trajectoryGeometry = computed(() => {
  if (!props.trajectory || props.trajectory.length === 0) return null

  const points = props.trajectory.map(p => {
    const [xMin, xMax] = props.xRange
    const [yMin, yMax] = props.yRange

    // Map world coordinates to [-1, 1] range
    const u = -1 + (p.x - xMin) / (xMax - xMin) * 2
    const v = -1 + (p.y - yMin) / (yMax - yMin) * 2

    return new THREE.Vector3(u, v, p.z)
  })

  return new THREE.BufferGeometry().setFromPoints(points)
})

// Start and end points for trajectory
const startPoint = computed(() => {
  if (!props.trajectory || props.trajectory.length === 0) return null
  const p = props.trajectory[0]
  const [xMin, xMax] = props.xRange
  const [yMin, yMax] = props.yRange

  const u = -1 + (p.x - xMin) / (xMax - xMin) * 2
  const v = -1 + (p.y - yMin) / (yMax - yMin) * 2

  return new THREE.Vector3(u, v, p.z)
})

const endPoint = computed(() => {
  if (!props.trajectory || props.trajectory.length === 0) return null
  const p = props.trajectory[props.trajectory.length - 1]
  const [xMin, xMax] = props.xRange
  const [yMin, yMax] = props.yRange

  const u = -1 + (p.x - xMin) / (xMax - xMin) * 2
  const v = -1 + (p.y - yMin) / (yMax - yMin) * 2

  return new THREE.Vector3(u, v, p.z)
})
</script>

<template>
  <div class="landscape-container">
    <div class="controls">
      <div class="control-group">
        <label>Function:</label>
        <select
          :value="functionName"
          @change="emit('update:functionName', ($event.target as HTMLSelectElement).value)"
          class="select"
        >
          <option value="sphere">Sphere</option>
          <option value="rosenbrock">Rosenbrock</option>
          <option value="rastrigin">Rastrigin</option>
          <option value="ackley">Ackley</option>
          <option value="griewank">Griewank</option>
          <option value="himmelblau">Himmelblau</option>
        </select>
      </div>
      <div class="control-info">
        <span class="hint">🖱️ Drag to rotate • Scroll to zoom • Right-click to pan</span>
      </div>
    </div>

    <div class="chart" :style="{ height: chartHeight }">
      <TresCanvas>
        <TresPerspectiveCamera :position="[3, 3, 5]" :fov="45" />
        <OrbitControls />

        <!-- Surface mesh with vertex colors -->
        <TresMesh :geometry="surfaceGeometry">
          <TresMeshPhongMaterial
            :vertex-colors="true"
            :side="THREE.DoubleSide"
            :shininess="100"
            :specular="0x111111"
          />
        </TresMesh>

        <!-- Trajectory line -->
        <TresLine
          v-if="trajectoryGeometry"
          :geometry="trajectoryGeometry"
        >
          <TresLineBasicMaterial :color="catppuccinColors.yellow" :linewidth="3" />
        </TresLine>

        <!-- Start point marker -->
        <TresMesh v-if="startPoint" :position="startPoint">
          <TresSphereGeometry :args="[0.05, 16, 16]" />
          <TresMeshBasicMaterial :color="catppuccinColors.green" />
        </TresMesh>

        <!-- End point marker -->
        <TresMesh v-if="endPoint" :position="endPoint">
          <TresSphereGeometry :args="[0.05, 16, 16]" />
          <TresMeshBasicMaterial :color="catppuccinColors.red" />
        </TresMesh>

        <!-- Lighting -->
        <TresPointLight :position="[5, 5, 10]" :intensity="0.8" />
        <TresPointLight :position="[-5, -5, -10]" :intensity="0.5" />
        <TresAmbientLight :intensity="0.3" />

        <!-- Grid helper (optional) -->
        <TresGridHelper :args="[10, 10]" />
      </TresCanvas>
    </div>

    <div v-if="trajectory && trajectory.length" class="trajectory-info">
      <span class="legend-item">
        <span class="dot green"></span> Start
      </span>
      <span class="legend-item">
        <span class="dot red"></span> End (Best)
      </span>
      <span class="legend-item">
        <span class="line yellow"></span> Trajectory
      </span>
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
  gap: 8px;
}

.control-group label {
  color: var(--ctp-mocha-text, #cdd6f4);
  font-weight: 500;
}

.select {
  background-color: var(--ctp-mocha-surface1, #45475a);
  border: 1px solid var(--ctp-mocha-surface2, #585b70);
  border-radius: 4px;
  color: var(--ctp-mocha-text, #cdd6f4);
  padding: 6px 12px;
  font-size: 0.875rem;
}

.select:focus {
  outline: none;
  border-color: var(--ctp-mocha-mauve, #cba6f7);
}

.control-info {
  color: var(--ctp-mocha-subtext0, #a6adc8);
  font-size: 0.8rem;
}

.hint {
  opacity: 0.8;
}

.chart {
  width: 100%;
  min-height: 400px;
  border-radius: 4px;
  background-color: var(--ctp-mocha-base, #1e1e2e);
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
