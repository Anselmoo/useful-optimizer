// https://vitepress.dev/guide/custom-theme
import { h, defineAsyncComponent } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      // https://vitepress.dev/guide/extending-default-theme#layout-slots
    })
  },
  async enhanceApp({ app }) {
    // Only register components on client side
    if (typeof window !== 'undefined') {
      // Dynamically import and register ECharts
      const { default: VChart } = await import('vue-echarts')
      const { use } = await import('echarts/core')
      const { CanvasRenderer } = await import('echarts/renderers')
      const { LineChart, ScatterChart, BoxplotChart, BarChart } = await import('echarts/charts')
      const {
        GridComponent,
        TooltipComponent,
        LegendComponent,
        TitleComponent,
        VisualMapComponent,
        ToolboxComponent
      } = await import('echarts/components')

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
        ToolboxComponent
      ])

      app.component('VChart', VChart)

      // Register 2D chart components as async
      app.component('ECDFChart', defineAsyncComponent(() =>
        import('./components/ECDFChart.vue')
      ))
      app.component('ConvergenceChart', defineAsyncComponent(() =>
        import('./components/ConvergenceChart.vue')
      ))
      app.component('ViolinPlot', defineAsyncComponent(() =>
        import('./components/ViolinPlot.vue')
      ))

      // Register 3D components as async (client-only)
      app.component('FitnessLandscape3D', defineAsyncComponent(() =>
        import('./components/FitnessLandscape3D.vue')
      ))

      // Register APIDoc component for API documentation pages
      app.component('APIDoc', defineAsyncComponent(() =>
        import('./components/APIDoc.vue')
      ))
    }
  }
} satisfies Theme
