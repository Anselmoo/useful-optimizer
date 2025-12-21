/**
 * Catppuccin Mocha theme for ECharts
 * Used for scientific visualizations in benchmark dashboards
 */

export const catppuccinMochaTheme = {
  color: [
    '#cba6f7', // mauve (primary)
    '#89b4fa', // blue
    '#a6e3a1', // green
    '#f9e2af', // yellow
    '#fab387', // peach
    '#f38ba8', // red
    '#94e2d5', // teal
    '#f5c2e7', // pink
    '#74c7ec', // sapphire
    '#b4befe', // lavender
  ],
  
  backgroundColor: '#1e1e2e', // base
  
  textStyle: {
    color: '#cdd6f4', // text
  },
  
  title: {
    textStyle: {
      color: '#cdd6f4',
      fontWeight: 600,
    },
    subtextStyle: {
      color: '#a6adc8', // subtext0
    },
  },
  
  line: {
    itemStyle: {
      borderWidth: 2,
    },
    lineStyle: {
      width: 2,
    },
    symbolSize: 6,
    symbol: 'circle',
    smooth: false,
  },
  
  radar: {
    itemStyle: {
      borderWidth: 2,
    },
    lineStyle: {
      width: 2,
    },
    symbolSize: 6,
    symbol: 'circle',
    smooth: false,
  },
  
  bar: {
    itemStyle: {
      barBorderWidth: 0,
      barBorderRadius: 4,
    },
  },
  
  pie: {
    itemStyle: {
      borderWidth: 0,
      borderColor: '#1e1e2e',
    },
  },
  
  scatter: {
    itemStyle: {
      borderWidth: 0,
    },
  },
  
  boxplot: {
    itemStyle: {
      borderWidth: 2,
      color: '#313244',
      borderColor: '#cba6f7',
    },
  },
  
  parallel: {
    itemStyle: {
      borderWidth: 0,
    },
  },
  
  sankey: {
    itemStyle: {
      borderWidth: 0,
    },
  },
  
  funnel: {
    itemStyle: {
      borderWidth: 0,
    },
  },
  
  gauge: {
    itemStyle: {
      borderWidth: 0,
    },
  },
  
  candlestick: {
    itemStyle: {
      color: '#a6e3a1',
      color0: '#f38ba8',
      borderColor: '#a6e3a1',
      borderColor0: '#f38ba8',
      borderWidth: 1,
    },
  },
  
  graph: {
    itemStyle: {
      borderWidth: 0,
    },
    lineStyle: {
      width: 1,
      color: '#585b70', // surface2
    },
    symbolSize: 6,
    symbol: 'circle',
    smooth: false,
    label: {
      color: '#cdd6f4',
    },
  },
  
  map: {
    itemStyle: {
      areaColor: '#313244', // surface0
      borderColor: '#585b70', // surface2
      borderWidth: 0.5,
    },
    label: {
      color: '#cdd6f4',
    },
    emphasis: {
      itemStyle: {
        areaColor: '#45475a', // surface1
        borderColor: '#cba6f7', // mauve
        borderWidth: 1,
      },
      label: {
        color: '#cdd6f4',
      },
    },
  },
  
  geo: {
    itemStyle: {
      areaColor: '#313244',
      borderColor: '#585b70',
      borderWidth: 0.5,
    },
    label: {
      color: '#cdd6f4',
    },
    emphasis: {
      itemStyle: {
        areaColor: '#45475a',
        borderColor: '#cba6f7',
        borderWidth: 1,
      },
      label: {
        color: '#cdd6f4',
      },
    },
  },
  
  categoryAxis: {
    axisLine: {
      show: true,
      lineStyle: {
        color: '#585b70', // surface2
      },
    },
    axisTick: {
      show: true,
      lineStyle: {
        color: '#585b70',
      },
    },
    axisLabel: {
      show: true,
      color: '#a6adc8', // subtext0
    },
    splitLine: {
      show: true,
      lineStyle: {
        color: ['#313244'], // surface0
      },
    },
    splitArea: {
      show: false,
      areaStyle: {
        color: ['#1e1e2e', '#181825'],
      },
    },
  },
  
  valueAxis: {
    axisLine: {
      show: true,
      lineStyle: {
        color: '#585b70',
      },
    },
    axisTick: {
      show: true,
      lineStyle: {
        color: '#585b70',
      },
    },
    axisLabel: {
      show: true,
      color: '#a6adc8',
    },
    splitLine: {
      show: true,
      lineStyle: {
        color: ['#313244'],
      },
    },
    splitArea: {
      show: false,
      areaStyle: {
        color: ['#1e1e2e', '#181825'],
      },
    },
  },
  
  logAxis: {
    axisLine: {
      show: true,
      lineStyle: {
        color: '#585b70',
      },
    },
    axisTick: {
      show: true,
      lineStyle: {
        color: '#585b70',
      },
    },
    axisLabel: {
      show: true,
      color: '#a6adc8',
    },
    splitLine: {
      show: true,
      lineStyle: {
        color: ['#313244'],
      },
    },
    splitArea: {
      show: false,
      areaStyle: {
        color: ['#1e1e2e', '#181825'],
      },
    },
  },
  
  timeAxis: {
    axisLine: {
      show: true,
      lineStyle: {
        color: '#585b70',
      },
    },
    axisTick: {
      show: true,
      lineStyle: {
        color: '#585b70',
      },
    },
    axisLabel: {
      show: true,
      color: '#a6adc8',
    },
    splitLine: {
      show: true,
      lineStyle: {
        color: ['#313244'],
      },
    },
    splitArea: {
      show: false,
      areaStyle: {
        color: ['#1e1e2e', '#181825'],
      },
    },
  },
  
  toolbox: {
    iconStyle: {
      borderColor: '#a6adc8',
    },
    emphasis: {
      iconStyle: {
        borderColor: '#cba6f7',
      },
    },
  },
  
  legend: {
    textStyle: {
      color: '#cdd6f4',
    },
    pageIconColor: '#cba6f7',
    pageIconInactiveColor: '#585b70',
    pageTextStyle: {
      color: '#cdd6f4',
    },
  },
  
  tooltip: {
    backgroundColor: '#181825', // mantle
    borderColor: '#45475a', // surface1
    borderWidth: 1,
    textStyle: {
      color: '#cdd6f4',
    },
    axisPointer: {
      lineStyle: {
        color: '#585b70',
        width: 1,
      },
      crossStyle: {
        color: '#585b70',
        width: 1,
      },
    },
  },
  
  timeline: {
    lineStyle: {
      color: '#585b70',
      width: 2,
    },
    itemStyle: {
      color: '#cba6f7',
      borderWidth: 1,
    },
    controlStyle: {
      color: '#cba6f7',
      borderColor: '#cba6f7',
      borderWidth: 1,
    },
    checkpointStyle: {
      color: '#89b4fa',
      borderColor: '#89b4fa',
    },
    label: {
      color: '#a6adc8',
    },
    emphasis: {
      itemStyle: {
        color: '#b4befe',
      },
      controlStyle: {
        color: '#cba6f7',
        borderColor: '#cba6f7',
        borderWidth: 1,
      },
      label: {
        color: '#cdd6f4',
      },
    },
  },
  
  visualMap: {
    color: ['#f38ba8', '#fab387', '#f9e2af', '#a6e3a1'],
    textStyle: {
      color: '#cdd6f4',
    },
  },
  
  dataZoom: {
    backgroundColor: '#1e1e2e',
    dataBackgroundColor: '#313244',
    fillerColor: 'rgba(203, 166, 247, 0.2)',
    handleColor: '#cba6f7',
    handleSize: '100%',
    textStyle: {
      color: '#cdd6f4',
    },
    borderColor: '#45475a',
  },
  
  markPoint: {
    label: {
      color: '#cdd6f4',
    },
    emphasis: {
      label: {
        color: '#cdd6f4',
      },
    },
  },
}

/**
 * Catppuccin color palette for charts
 */
export const catppuccinColors = {
  // Primary colors
  mauve: '#cba6f7',
  blue: '#89b4fa',
  green: '#a6e3a1',
  yellow: '#f9e2af',
  peach: '#fab387',
  red: '#f38ba8',
  teal: '#94e2d5',
  pink: '#f5c2e7',
  sapphire: '#74c7ec',
  lavender: '#b4befe',
  
  // Background colors
  base: '#1e1e2e',
  mantle: '#181825',
  crust: '#11111b',
  
  // Surface colors
  surface0: '#313244',
  surface1: '#45475a',
  surface2: '#585b70',
  
  // Text colors
  text: '#cdd6f4',
  subtext1: '#bac2de',
  subtext0: '#a6adc8',
  
  // Overlay colors
  overlay2: '#9399b2',
  overlay1: '#7f849c',
  overlay0: '#6c7086',
}

/**
 * Get gradient colors for heatmaps and continuous visualizations
 */
export function getGradientColors(type: 'performance' | 'fitness' | 'viridis' = 'performance') {
  switch (type) {
    case 'performance':
      return [catppuccinColors.green, catppuccinColors.yellow, catppuccinColors.red]
    case 'fitness':
      return [catppuccinColors.teal, catppuccinColors.blue, catppuccinColors.mauve, catppuccinColors.red]
    case 'viridis':
      return ['#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725']
    default:
      return [catppuccinColors.green, catppuccinColors.yellow, catppuccinColors.red]
  }
}

/**
 * Algorithm category colors
 */
export const categoryColors = {
  swarm_intelligence: catppuccinColors.blue,
  evolutionary: catppuccinColors.green,
  gradient_based: catppuccinColors.peach,
  classical: catppuccinColors.mauve,
  metaheuristic: catppuccinColors.yellow,
  constrained: catppuccinColors.teal,
  probabilistic: catppuccinColors.pink,
}
