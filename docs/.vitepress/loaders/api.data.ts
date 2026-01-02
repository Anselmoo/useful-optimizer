// Auto-generated VitePress data loader for API documentation
// Loads Griffe-generated JSON API data for use in Vue components
import { defineLoader } from 'vitepress'
import { readFileSync } from 'fs'
import { resolve } from 'path'

export interface Parameter {
  name: string
  annotation: string
  description: string
  default?: string
}

export interface Method {
  name: string
  docstring: string
  parameters: Parameter[]
  returns: string
}

export interface Class {
  name: string
  docstring: string
  bases: string[]
  parameters: Parameter[]
  methods: Method[]
  attributes: Array<{ name: string; annotation: string; description: string }>
}

export interface Module {
  name: string
  docstring: string
  classes: Class[]
  functions: Method[]
}

export interface APIData {
  modules: Record<string, Module>
  categories: string[]
}

function transformGriffeOutput(data: any): Module {
  const classes: Class[] = []
  const functions: Method[] = []

  // Process each member in the module
  for (const [name, member] of Object.entries(data.members || {})) {
    const memberData = member as any
    
    if (memberData.kind === 'class') {
      const classData: Class = {
        name: name,
        docstring: memberData.docstring?.value || '',
        bases: memberData.bases || [],
        parameters: [],
        methods: [],
        attributes: []
      }

      // Extract __init__ parameters
      const initMethod = memberData.members?.['__init__']
      if (initMethod && initMethod.parameters) {
        classData.parameters = Object.entries(initMethod.parameters)
          .filter(([pname]) => pname !== 'self')
          .map(([pname, param]: [string, any]) => ({
            name: pname,
            annotation: param.annotation || 'Any',
            description: param.description || '',
            default: param.default
          }))
      }

      // Extract methods
      for (const [methodName, method] of Object.entries(memberData.members || {})) {
        const methodData = method as any
        if (methodData.kind === 'function' && !methodName.startsWith('_')) {
          classData.methods.push({
            name: methodName,
            docstring: methodData.docstring?.value || '',
            parameters: Object.entries(methodData.parameters || {})
              .filter(([pname]) => pname !== 'self')
              .map(([pname, param]: [string, any]) => ({
                name: pname,
                annotation: param.annotation || 'Any',
                description: param.description || '',
                default: param.default
              })),
            returns: methodData.returns || 'None'
          })
        }
      }

      // Extract attributes
      for (const [attrName, attr] of Object.entries(memberData.members || {})) {
        const attrData = attr as any
        if (attrData.kind === 'attribute') {
          classData.attributes.push({
            name: attrName,
            annotation: attrData.annotation || 'Any',
            description: attrData.docstring?.value || ''
          })
        }
      }

      classes.push(classData)
    } else if (memberData.kind === 'function') {
      functions.push({
        name: name,
        docstring: memberData.docstring?.value || '',
        parameters: Object.entries(memberData.parameters || {}).map(([pname, param]: [string, any]) => ({
          name: pname,
          annotation: param.annotation || 'Any',
          description: param.description || '',
          default: param.default
        })),
        returns: memberData.returns || 'None'
      })
    }
  }

  return {
    name: data.name || 'unknown',
    docstring: data.docstring?.value || '',
    classes,
    functions
  }
}

export default defineLoader({
  async load(): Promise<APIData> {
    const categories = [
      'swarm_intelligence',
      'evolutionary',
      'gradient_based',
      'classical',
      'metaheuristic',
      'physics_inspired',
      'probabilistic',
      'social_inspired',
      'constrained',
      'multi_objective'
    ]

    const modules: Record<string, Module> = {}

    for (const category of categories) {
      try {
        const jsonPath = resolve(__dirname, `../../api/${category}.json`)
        const rawData = readFileSync(jsonPath, 'utf-8')
        const data = JSON.parse(rawData)
        
        // The root key in Griffe output is the package name
        const packageData = data.opt?.[category] || data[`opt.${category}`] || {}
        modules[category] = transformGriffeOutput(packageData)
      } catch (error) {
        console.warn(`Failed to load API data for ${category}:`, error)
        modules[category] = {
          name: category,
          docstring: '',
          classes: [],
          functions: []
        }
      }
    }

    return {
      modules,
      categories
    }
  }
})
