<script setup lang="ts">
/**
 * FitnessLandscape3D.vue
 * 
 * Interactive 3D fitness landscape visualization using ECharts-GL.
 * Allows orbit controls, zoom, and trajectory overlay.
 */
import { ref, computed, onMounted, onUnmounted, watch, shallowRef } from 'vue'
import * as echarts from 'echarts'
import 'echarts-gl'
import { catppuccinMochaTheme, catppuccinColors, getGradientColors } from '../../themes/catppuccin'

interface TrajectoryPoint {
  x: number
  y: number
  z: number
  iteration: number
}

interface Props {
  functionName: string
  bounds?: [number, number]
  resolution?: number
  trajectory?: TrajectoryPoint[]
  showContour?: boolean
  colormap?: 'catppuccin' | 'viridis' | 'plasma'
  height?: string | number
}

const props = withDefaults(defineProps<Props>(), {
  bounds: () => [-5, 5],
  resolution: 50,
  showContour: false,
  colormap: 'catppuccin',
  height: 500
})

const chartRef = ref<HTMLElement | null>(null)
const chart = shallowRef<echarts.ECharts | null>(null)

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

// Generate landscape data
const generateLandscape = () => {
  const func = benchmarkFunctions[props.functionName] || benchmarkFunctions.sphere
  const [min, max] = props.bounds
  const step = (max - min) / props.resolution
  
  const data: number[][] = []
  let minZ = Infinity
  let maxZ = -Infinity
  
  for (let i = 0; i <= props.resolution; i++) {
    for (let j = 0; j <= props.resolution; j++) {
      const x = min + i * step
      const y = min + j * step
      const z = func(x, y)
      
      minZ = Math.min(minZ, z)
      maxZ = Math.max(maxZ, z)
      
      data.push([x, y, z])
    }
  }
  
  return { data, minZ, maxZ }
}

const getColors = () => {
  switch (props.colormap) {
    case 'viridis':
      return getGradientColors('viridis')
    case 'catppuccin':
    default:
      return [
        catppuccinColors.green,
        catppuccinColors.teal,
        catppuccinColors.blue,
        catppuccinColors.mauve,
        catppuccinColors.red
      ]
  }
}

const chartOption = computed(() => {
  const { data, minZ, maxZ } = generateLandscape()
  const colors = getColors()
  
  const series: echarts.SeriesOption[] = [
    {
      type: 'surface',
      wireframe: {
        show: false
      },
      shading: 'realistic',
      realisticMaterial: {
        roughness: 0.5,
        metalness: 0.1
      },
      data: data,
      itemStyle: {
        opacity: 0.95
      }
    } as any
  ]
  
  // Add trajectory if provided
  if (props.trajectory && props.trajectory.length > 0) {
    series.push({
      type: 'line3D',
      data: props.trajectory.map(p => [p.x, p.y, p.z + (maxZ - minZ) * 0.01]),
      lineStyle: {
        color: catppuccinColors.yellow,
        width: 3
      }
    } as any)
    
    // Add start point
    series.push({
      type: 'scatter3D',
      data: [[props.trajectory[0].x, props.trajectory[0].y, props.trajectory[0].z]],
      symbolSize: 12,
      itemStyle: {
        color: catppuccinColors.green
      }
    } as any)
    
    // Add end point
    const last = props.trajectory[props.trajectory.length - 1]
    series.push({
      type: 'scatter3D',
      data: [[last.x, last.y, last.z]],
      symbolSize: 12,
      itemStyle: {
        color: catppuccinColors.red
      }
    } as any)
  }
  
  return {
    backgroundColor: catppuccinColors.base,
    title: {
      text: `${props.functionName.charAt(0).toUpperCase() + props.functionName.slice(1)} Function`,
      left: 'center',
      top: 10,
      textStyle: {
        color: catppuccinColors.text,
        fontSize: 16,
        fontWeight: 600
      }
    },
    tooltip: {
      backgroundColor: catppuccinColors.mantle,
      borderColor: catppuccinColors.surface1,
      textStyle: {
        color: catppuccinColors.text
      },
      formatter: (params: any) => {
        const [x, y, z] = params.value
        return `x: ${x.toFixed(3)}<br>y: ${y.toFixed(3)}<br>f(x,y): ${z.toExponential(3)}`
      }
    },
    visualMap: {
      show: true,
      min: minZ,
      max: maxZ,
      dimension: 2,
      inRange: {
        color: colors
      },
      textStyle: {
        color: catppuccinColors.text
      },
      left: 10,
      bottom: 60
    },
    xAxis3D: {
      type: 'value',
      name: 'x',
      nameTextStyle: { color: catppuccinColors.text },
      axisLine: { lineStyle: { color: catppuccinColors.surface2 } },
      axisLabel: { color: catppuccinColors.subtext0 },
      splitLine: { lineStyle: { color: catppuccinColors.surface0 } }
    },
    yAxis3D: {
      type: 'value',
      name: 'y',
      nameTextStyle: { color: catppuccinColors.text },
      axisLine: { lineStyle: { color: catppuccinColors.surface2 } },
      axisLabel: { color: catppuccinColors.subtext0 },
      splitLine: { lineStyle: { color: catppuccinColors.surface0 } }
    },
    zAxis3D: {
      type: 'value',
      name: 'f(x,y)',
      nameTextStyle: { color: catppuccinColors.text },
      axisLine: { lineStyle: { color: catppuccinColors.surface2 } },
      axisLabel: { 
        color: catppuccinColors.subtext0,
        formatter: (value: number) => value.toExponential(0)
      },
      splitLine: { lineStyle: { color: catppuccinColors.surface0 } }
    },
    grid3D: {
      viewControl: {
        projection: 'perspective',
        autoRotate: false,
        rotateSensitivity: 2,
        zoomSensitivity: 1,
        panSensitivity: 1,
        distance: 200,
        alpha: 30,
        beta: 40
      },
      light: {
        main: {
          intensity: 1.2,
          shadow: true
        },
        ambient: {
          intensity: 0.3
        }
      },
      boxWidth: 100,
      boxHeight: 60,
      boxDepth: 100,
      environment: catppuccinColors.base
    },
    series
  }
})

const initChart = () => {
  if (!chartRef.value) return
  
  echarts.registerTheme('catppuccin-mocha', catppuccinMochaTheme)
  chart.value = echarts.init(chartRef.value, 'catppuccin-mocha')
  chart.value.setOption(chartOption.value)
}

const resizeChart = () => {
  chart.value?.resize()
}

watch(() => [props.functionName, props.bounds, props.resolution, props.trajectory], () => {
  chart.value?.setOption(chartOption.value, true)
}, { deep: true })

onMounted(() => {
  initChart()
  window.addEventListener('resize', resizeChart)
})

onUnmounted(() => {
  window.removeEventListener('resize', resizeChart)
  chart.value?.dispose()
})
</script>

<template>
  <div class="landscape-container">
    <div class="controls">
      <div class="control-group">
        <label>Function:</label>
        <select 
          :value="functionName"
          @change="$emit('update:functionName', ($event.target as HTMLSelectElement).value)"
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
    <div 
      ref="chartRef" 
      class="chart"
      :style="{ height: chartHeight }"
    />
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
