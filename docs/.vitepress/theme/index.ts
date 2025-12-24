// https://vitepress.dev/guide/custom-theme
import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import './style.css'

// Import ECharts components
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, ScatterChart, BoxplotChart, BarChart } from 'echarts/charts'
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  TitleComponent,
  VisualMapComponent,
  Grid3DComponent,
  ToolboxComponent
} from 'echarts/components'

// Import TresJS components
import { TresCanvas } from '@tresjs/core'

// Import chart components
import ECDFChart from './components/ECDFChart.vue'
import ConvergenceChart from './components/ConvergenceChart.vue'
import ViolinPlot from './components/ViolinPlot.vue'
import FitnessLandscape3D from './components/FitnessLandscape3D.vue'

// Register ECharts components
use([
  CanvasRenderer,
  LineChart,
  ScatterChart,
  BoxplotChart,
  BarChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  TitleComponent,
  VisualMapComponent,
  Grid3DComponent,
  ToolboxComponent
])

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      // https://vitepress.dev/guide/extending-default-theme#layout-slots
    })
  },
  enhanceApp({ app }) {
    // Register ECharts
    app.component('VChart', VChart)

    // Register TresJS for 3D
    app.component('TresCanvas', TresCanvas)

    // Register chart components
    app.component('ECDFChart', ECDFChart)
    app.component('ConvergenceChart', ConvergenceChart)
    app.component('ViolinPlot', ViolinPlot)
    app.component('FitnessLandscape3D', FitnessLandscape3D)
  }
} satisfies Theme
