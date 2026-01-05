# Visualization Architecture

## Why ECharts?

When building the benchmark visualization system for `useful-optimizer`, we evaluated multiple charting libraries to determine the best fit for scientific data visualization in a VitePress environment. Here's our comprehensive comparison:

### ECharts vs Chart.js Comparison

| Feature | ECharts | Chart.js | Winner |
|---------|---------|----------|--------|
| **SSR Support** | ✅ Native server-side rendering | ⚠️ Requires workarounds | ECharts |
| **Large Datasets** | ✅ WebGL acceleration, 30K+ points | ⚠️ Degrades >5K points | ECharts |
| **Vue Integration** | ✅ Official `vue-echarts` package | ❌ Manual wrapper needed | ECharts |
| **3D Support** | ✅ Built-in `echarts-gl` | ❌ No native 3D | ECharts |
| **Scientific Charts** | ✅ Heatmaps, parallel coords, radar | ⚠️ Limited | ECharts |
| **Bundle Size** | ⚠️ ~900KB full, ~300KB tree-shaken | ✅ ~200KB | Chart.js |
| **TypeScript** | ✅ First-class support | ✅ Good support | Tie |
| **Learning Curve** | ⚠️ Complex option API | ✅ Simpler API | Chart.js |
| **Documentation** | ✅ Extensive examples | ✅ Good docs | Tie |

**Decision: ECharts** - SSR support, large dataset handling, and 3D capabilities are critical for scientific visualization of optimization algorithms.

### Why SSR Support Matters

VitePress builds static sites using server-side rendering (SSR). Libraries that access browser-only APIs (like `window`, `document`) during module loading will crash the build process. ECharts, when properly configured with VitePress's `noExternal` option, handles SSR gracefully without requiring component-level workarounds.

## Architecture Overview

Our visualization system follows a layered architecture that integrates VitePress, Vue 3, and ECharts:

```
┌─────────────────────────────────────────────────────────────────┐
│                     VitePress + Vue 3 + ECharts                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  VitePress   │───▶│  Vue 3 SFC   │───▶│  vue-echarts │       │
│  │  Markdown    │    │  Components  │    │  Wrapper     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Static      │    │  Client-side │    │  ECharts     │       │
│  │  Generation  │    │  Hydration   │    │  Instance    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                                       │                │
│         ▼                                       ▼                │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              Browser Rendering                        │       │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │       │
│  │  │  ECDF   │  │ Converg │  │ Violin  │  │   3D    │  │       │
│  │  │  Chart  │  │  Chart  │  │  Plot   │  │Landscape│  │       │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Flow

1. **VitePress Markdown**: Authors write documentation with embedded component references
2. **Vue 3 SFC Components**: Custom chart components receive data props and configure ECharts
3. **vue-echarts Wrapper**: Official Vue 3 wrapper handles ECharts instance lifecycle
4. **Static Generation**: VitePress builds static HTML during SSR phase
5. **Client-side Hydration**: Browser loads and activates interactive charts
6. **ECharts Instance**: Renders interactive, responsive visualizations

## SSR Configuration Pattern

### VitePress Configuration

Our `docs/.vitepress/config.ts` includes essential Vite SSR configuration:

```typescript
// docs/.vitepress/config.ts
import { defineConfig } from 'vitepress'

export default defineConfig({
  // ... other config

  vite: {
    ssr: {
      // Ensure ECharts is bundled for SSR, not treated as external
      noExternal: ['echarts', 'vue-echarts', 'zrender']
    },
    optimizeDeps: {
      // Pre-bundle these dependencies for faster dev server
      include: ['echarts', 'vue-echarts']
    }
  }
})
```

**Key Configuration Points:**
- `noExternal`: Forces Vite to bundle ECharts for SSR instead of treating it as external
- `optimizeDeps`: Improves development server performance by pre-bundling large dependencies

### Component SSR Pattern

While our current implementation works with static imports due to the `noExternal` configuration, here's the recommended pattern for explicit SSR safety:

```typescript
// Pattern for robust SSR handling
import { onMounted, ref, shallowRef } from 'vue'
import type { EChartsOption } from 'echarts'

const chartRef = ref<HTMLElement | null>(null)
const chartInstance = shallowRef<echarts.ECharts | null>(null)

onMounted(async () => {
  // SSR SAFETY: Only run in browser environment
  if (typeof window === 'undefined') return

  // Dynamic import for additional safety (optional with noExternal)
  const echarts = await import('echarts')

  if (chartRef.value) {
    chartInstance.value = echarts.init(chartRef.value)
    chartInstance.value.setOption(chartOption.value)
  }
})
```

**Why This Pattern?**
- `onMounted()` only runs in the browser, never during SSR
- `typeof window === 'undefined'` provides explicit guard
- `shallowRef` for chart instance avoids deep reactivity overhead
- Dynamic import can provide extra isolation (optional with our config)

## Available Chart Components

### ECDFChart - Empirical Cumulative Distribution Function

**Purpose**: The gold standard for optimizer comparison following COCO platform standards.

**Features**:
- Shows proportion of (function, target) pairs solved at each budget
- Log-scale x-axis for budget (function evaluations / dimension)
- Multiple algorithm comparison with distinct colors
- Target precision badges for reference

**Usage**:
```vue
<ECDFChart
  :data="ecdfData"
  title="Algorithm Performance Comparison"
  :log-x-axis="true"
  :target-precisions="[1e-1, 1e-3, 1e-5, 1e-7]"
/>
```

**Data Format**:
```typescript
interface ECDFData {
  algorithm: string
  budget: number[]       // Function evaluations / dimension
  proportion: number[]   // Proportion of targets reached [0, 1]
}
```

### ConvergenceChart - Fitness Over Iterations

**Purpose**: Visualize optimization progress over iterations with confidence bands.

**Features**:
- Mean fitness trajectory with ±σ confidence bands
- Log-scale y-axis for fitness values
- Multiple algorithm comparison
- Customizable axis labels

**Usage**:
```vue
<ConvergenceChart
  :data="convergenceData"
  title="Convergence Curves"
  :log-scale="true"
  :show-confidence-band="true"
/>
```

**Data Format**:
```typescript
interface ConvergenceData {
  algorithm: string
  iterations: number[]
  mean: number[]
  std?: number[]  // For confidence bands
}
```

### ViolinPlot - Final Fitness Distribution

**Purpose**: Statistical distribution visualization for final fitness values across multiple runs.

**Features**:
- Boxplot with quartiles and outliers
- Scatter points for individual runs
- Summary statistics table (mean, std, best, worst)
- Interactive controls for display options

**Usage**:
```vue
<ViolinPlot
  :data="violinData"
  title="Final Fitness Distribution"
  :log-scale="true"
  :show-boxplot="true"
  :show-points="true"
/>
```

**Data Format**:
```typescript
interface ViolinData {
  algorithm: string
  values: number[]  // Fitness values from multiple runs
}
```

### FitnessLandscape3D - Interactive 3D Surface Plots

**Purpose**: Visualize benchmark function landscapes and optimizer trajectories.

**Features**:
- Interactive 3D surface with OrbitControls (rotate, zoom, pan)
- Multiple benchmark functions (Sphere, Rosenbrock, Rastrigin, Ackley, etc.)
- Optional optimization trajectory visualization
- Color scales: Viridis, Turbo, Plasma, Inferno, Catppuccin

**Technology**: Uses **TresJS** (Vue wrapper for Three.js) instead of deprecated `echarts-gl`.

**Usage**:
```vue
<FitnessLandscape3D
  function-name="rosenbrock"
  :x-range="[-2, 2]"
  :y-range="[-1, 3]"
  :resolution="100"
  :trajectory="trajectoryPoints"
  color-scale="viridis"
/>
```

**Data Format**:
```typescript
interface TrajectoryPoint {
  x: number
  y: number
  z: number  // Fitness value
  iteration: number
}
```

## Performance Optimization

### Bundle Size Optimization

ECharts supports tree-shaking to reduce bundle size. Our components use direct imports:

```typescript
import * as echarts from 'echarts'
```

This works with our build configuration, but for maximum optimization, consider using core imports:

```typescript
import * as echarts from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

echarts.use([LineChart, GridComponent, TooltipComponent, CanvasRenderer])
```

**Current bundle sizes** (after tree-shaking):
- ConvergenceChart: ~350KB
- ECDFChart: ~340KB
- ViolinPlot: ~360KB (includes boxplot)
- FitnessLandscape3D: ~450KB (Three.js + TresJS)

### Large Dataset Performance

**ECharts advantages for benchmark data**:
- WebGL rendering support for 30,000+ data points
- Efficient canvas rendering for typical datasets (100-10,000 points)
- Built-in data sampling and progressive rendering
- Hardware acceleration for animations

**Best practices**:
- Use `sampling: 'lttb'` for large convergence curves (Largest-Triangle-Three-Buckets downsampling)
- Disable animations for datasets >5,000 points
- Use `lazy: true` for 3D landscapes with high resolution

### Lazy Loading

Our theme setup uses `defineAsyncComponent` for code splitting:

```typescript
// docs/.vitepress/theme/index.ts
import { defineAsyncComponent } from 'vue'

app.component('ConvergenceChart', defineAsyncComponent(() =>
  import('./components/ConvergenceChart.vue')
))
```

This ensures chart components are only loaded when actually used on a page.

## Color Theming

All chart components use the **Catppuccin Mocha** theme for consistency:

```typescript
import { catppuccinMochaTheme, catppuccinColors } from '../../themes/catppuccin'

echarts.registerTheme('catppuccin-mocha', catppuccinMochaTheme)
const chart = echarts.init(chartRef.value, 'catppuccin-mocha')
```

**Color palette**:
- Primary algorithm colors: Mauve, Blue, Green, Yellow, Peach, Red, Teal, Pink
- Background: Surface0 (#313244)
- Text: Text (#cdd6f4), Subtext0 (#a6adc8)
- Grid lines: Surface0, Surface1, Surface2

## Accessibility

**Keyboard navigation**: All charts support keyboard interaction through ECharts' built-in accessibility features.

**Screen readers**: Charts include:
- Semantic HTML structure
- ARIA labels on interactive elements
- Text descriptions in surrounding content

**Color blindness**: Catppuccin Mocha palette has been tested for color contrast and distinguishability.

## Browser Compatibility

**Supported browsers**:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

**Progressive enhancement**:
- Static content renders even if JavaScript fails
- Graceful degradation for older browsers
- No critical functionality blocked by chart rendering

## Troubleshooting

### Build Fails with "window is not defined"

**Symptom**: VitePress build crashes during SSR phase.

**Solution**: Ensure `noExternal` configuration is present:
```typescript
vite: {
  ssr: {
    noExternal: ['echarts', 'vue-echarts', 'zrender']
  }
}
```

### Charts Not Rendering

**Symptom**: Empty div where chart should appear.

**Solution**: Check that:
1. Component is registered in `theme/index.ts`
2. Data prop format matches expected interface
3. Browser console for JavaScript errors
4. Chart container has explicit height CSS

### Performance Issues

**Symptom**: Slow rendering or laggy interactions.

**Solution**:
- Reduce `resolution` prop for 3D landscapes
- Enable data sampling for large datasets
- Disable confidence bands if not needed
- Use `shallowRef` for large data objects

## Future Enhancements

**Planned features**:
- [ ] Parallel coordinates chart for high-dimensional optimization
- [ ] Animated convergence playback
- [ ] Export to PNG/SVG functionality
- [ ] Interactive benchmark suite builder
- [ ] Real-time optimization visualization

## Resources

- [ECharts Documentation](https://echarts.apache.org/en/index.html)
- [vue-echarts GitHub](https://github.com/ecomfe/vue-echarts)
- [VitePress SSR Guide](https://vitepress.dev/guide/ssr-compat)
- [TresJS Documentation](https://tresjs.org/)
- [COCO Platform](https://github.com/numbbo/coco)
- [Catppuccin Theme](https://github.com/catppuccin/catppuccin)
