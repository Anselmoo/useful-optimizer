import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Useful Optimizer',
  description: 'A comprehensive collection of 54+ optimization algorithms for numeric problems',
  
  base: '/useful-optimizer/',
  
  // TODO: Remove ignoreDeadLinks once all algorithm and API pages are created
  // Currently ignoring dead links as not all navigation links have corresponding pages yet
  ignoreDeadLinks: true,
  
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/logo.svg' }],
    ['meta', { name: 'theme-color', content: '#cba6f7' }],
    ['meta', { name: 'og:type', content: 'website' }],
    ['meta', { name: 'og:title', content: 'Useful Optimizer' }],
    ['meta', { name: 'og:description', content: 'A comprehensive collection of 54+ optimization algorithms for numeric problems' }],
  ],
  
  themeConfig: {
    logo: '/logo.svg',
    
    nav: [
      { text: 'Guide', link: '/guide/' },
      { text: 'Algorithms', link: '/algorithms/' },
      { text: 'API Reference', link: '/api/' },
      { text: 'Benchmarks', link: '/benchmarks/' },
      {
        text: 'v0.1.2',
        items: [
          { text: 'Changelog', link: '/changelog' },
          { text: 'Contributing', link: '/contributing' }
        ]
      }
    ],
    
    sidebar: {
      '/guide/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Introduction', link: '/guide/' },
            { text: 'Installation', link: '/guide/installation' },
            { text: 'Quick Start', link: '/guide/quickstart' },
            { text: 'Advanced Usage', link: '/guide/advanced' }
          ]
        }
      ],
      '/algorithms/': [
        {
          text: 'Overview',
          items: [
            { text: 'Introduction', link: '/algorithms/' }
          ]
        },
        {
          text: 'Swarm Intelligence',
          collapsed: false,
          items: [
            { text: 'Particle Swarm', link: '/algorithms/swarm-intelligence/particle-swarm' },
            { text: 'Ant Colony', link: '/algorithms/swarm-intelligence/ant-colony' },
            { text: 'Firefly Algorithm', link: '/algorithms/swarm-intelligence/firefly' },
            { text: 'Bat Algorithm', link: '/algorithms/swarm-intelligence/bat' },
            { text: 'Grey Wolf', link: '/algorithms/swarm-intelligence/grey-wolf' },
            { text: 'Whale Optimization', link: '/algorithms/swarm-intelligence/whale' },
            { text: 'Cuckoo Search', link: '/algorithms/swarm-intelligence/cuckoo' }
          ]
        },
        {
          text: 'Evolutionary',
          collapsed: true,
          items: [
            { text: 'Genetic Algorithm', link: '/algorithms/evolutionary/genetic-algorithm' },
            { text: 'Differential Evolution', link: '/algorithms/evolutionary/differential-evolution' },
            { text: 'CMA-ES', link: '/algorithms/evolutionary/cma-es' },
            { text: 'Cultural Algorithm', link: '/algorithms/evolutionary/cultural' }
          ]
        },
        {
          text: 'Gradient-Based',
          collapsed: true,
          items: [
            { text: 'SGD Momentum', link: '/algorithms/gradient-based/sgd-momentum' },
            { text: 'Adam', link: '/algorithms/gradient-based/adam' },
            { text: 'AdamW', link: '/algorithms/gradient-based/adamw' },
            { text: 'RMSprop', link: '/algorithms/gradient-based/rmsprop' },
            { text: 'Adagrad', link: '/algorithms/gradient-based/adagrad' }
          ]
        },
        {
          text: 'Classical',
          collapsed: true,
          items: [
            { text: 'BFGS', link: '/algorithms/classical/bfgs' },
            { text: 'Nelder-Mead', link: '/algorithms/classical/nelder-mead' },
            { text: 'Simulated Annealing', link: '/algorithms/classical/simulated-annealing' },
            { text: 'Hill Climbing', link: '/algorithms/classical/hill-climbing' }
          ]
        },
        {
          text: 'Metaheuristic',
          collapsed: true,
          items: [
            { text: 'Harmony Search', link: '/algorithms/metaheuristic/harmony-search' },
            { text: 'Cross Entropy', link: '/algorithms/metaheuristic/cross-entropy' },
            { text: 'Sine Cosine', link: '/algorithms/metaheuristic/sine-cosine' }
          ]
        }
      ],
      '/api/': [
        {
          text: 'API Reference',
          items: [
            { text: 'Overview', link: '/api/' },
            { text: 'Abstract Optimizer', link: '/api/abstract-optimizer' }
          ]
        },
        {
          text: 'Modules',
          items: [
            { text: 'Swarm Intelligence', link: '/api/swarm-intelligence' },
            { text: 'Evolutionary', link: '/api/evolutionary' },
            { text: 'Gradient-Based', link: '/api/gradient-based' },
            { text: 'Classical', link: '/api/classical' },
            { text: 'Metaheuristic', link: '/api/metaheuristic' },
            { text: 'Constrained', link: '/api/constrained' },
            { text: 'Probabilistic', link: '/api/probabilistic' },
            { text: 'Benchmark Functions', link: '/api/benchmark-functions' }
          ]
        }
      ],
      '/benchmarks/': [
        {
          text: 'Benchmarks',
          items: [
            { text: 'Overview', link: '/benchmarks/' },
            { text: 'Methodology', link: '/benchmarks/methodology' },
            { text: 'Results', link: '/benchmarks/results' },
            { text: 'Benchmark Functions', link: '/benchmarks/functions' }
          ]
        }
      ]
    },
    
    socialLinks: [
      { icon: 'github', link: 'https://github.com/Anselmoo/useful-optimizer' }
    ],
    
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024 Anselm Hahn'
    },
    
    search: {
      provider: 'local'
    },
    
    editLink: {
      pattern: 'https://github.com/Anselmoo/useful-optimizer/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    }
  },
  
  markdown: {
    math: true,
    lineNumbers: true
  },

  vite: {
    ssr: {
      noExternal: ['echarts', 'vue-echarts', 'echarts-gl', 'zrender']
    }
  }
})
