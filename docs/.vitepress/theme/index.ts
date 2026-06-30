import { h, defineAsyncComponent } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import './style.css'

// APIDoc has no browser globals — safe to register synchronously
import APIDoc from './components/APIDoc.vue'

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {})
  },
  async enhanceApp({ app }) {
    app.component('APIDoc', APIDoc)

    // Chart components import echarts which accesses `document` at module load
    // time — must be deferred to client side only.
    if (typeof window !== 'undefined') {
      app.component('ConvergenceChart', defineAsyncComponent(() =>
        import('./components/ConvergenceChart.vue')
      ))
      app.component('ECDFChart', defineAsyncComponent(() =>
        import('./components/ECDFChart.vue')
      ))
      app.component('ViolinPlot', defineAsyncComponent(() =>
        import('./components/ViolinPlot.vue')
      ))
      app.component('FitnessLandscape3D', defineAsyncComponent(() =>
        import('./components/FitnessLandscape3D.vue')
      ))
    }
  }
} satisfies Theme
