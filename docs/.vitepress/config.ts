import { defineConfig } from 'vitepress'
import { algorithmsSidebar, apiModuleLinks } from './algorithmsSidebar'

const base = '/useful-optimizer/'

export default defineConfig({
  title: 'useful-optimizer',
  description: 'A dedicated set of optimization algorithms for numeric problems',

  base,

  // Markdown under public/ is data documentation, not a page. The developer demo
  // pages are available in `docs:dev` but not published.
  srcExclude: [
    'public/**/*.md',
    ...(process.env.NODE_ENV === 'production' ? ['benchmark-demo.md', 'test-charts.md'] : [])
  ],

  head: [
    // Must include base prefix; otherwise GitHub Pages serves this at /useful-optimizer/favicon.svg
    ['link', { rel: 'icon', type: 'image/svg+xml', href: `${base}favicon.svg` }],
    ['meta', { name: 'theme-color', content: '#cba6f7' }],
    ['meta', { name: 'og:type', content: 'website' }],
    ['meta', { name: 'og:title', content: 'Useful Optimizer' }],
    ['meta', { name: 'og:description', content: 'A comprehensive collection of 118 optimization algorithms for numeric problems' }],
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
      '/algorithms/': algorithmsSidebar,
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
            ...apiModuleLinks,
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
            { text: 'Benchmark Functions', link: '/benchmarks/functions' },
            { text: 'Visualization Architecture', link: '/benchmarks/visualization' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/Anselmoo/useful-optimizer' }
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024-present Anselmoo'
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
      noExternal: ['echarts', 'zrender']
    },
    optimizeDeps: {
      // Pre-bundle everything the client-only chart components import, so the
      // dev server does not re-optimise and reload on the first chart page.
      include: ['echarts', 'zrender', 'three', '@tresjs/core', '@tresjs/cientos', 'd3']
    }
  }
})
