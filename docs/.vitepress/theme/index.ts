import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import './style.css'

// ECharts
import { VChart } from 'vue-echarts'
import 'echarts'

// Custom components
import ConvergenceChart from './components/ConvergenceChart.vue'
import ECDFChart from './components/ECDFChart.vue'
import ViolinPlot from './components/ViolinPlot.vue'
import FitnessLandscape3D from './components/FitnessLandscape3D.vue'

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {})
  },
  enhanceApp({ app }) {
    app.component('VChart', VChart)
    app.component('ConvergenceChart', ConvergenceChart)
    app.component('ECDFChart', ECDFChart)
    app.component('ViolinPlot', ViolinPlot)
    app.component('FitnessLandscape3D', FitnessLandscape3D)
  }
} satisfies Theme
