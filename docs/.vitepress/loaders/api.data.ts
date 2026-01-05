// VitePress data loader for API documentation
// Loads Griffe-generated JSON API data for use in Vue components
import { defineLoader } from 'vitepress'
import { readFileSync } from 'fs'
import { resolve } from 'path'
import type { GriffeOutput, GriffeMember, Parameter as GriffeParameter, Annotation } from '../types/griffe'

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
  classIndex: Record<string, { category: string; module: string; class: Class }>
}

/**
 * Convert Griffe annotation to string representation
 */
function annotationToString(annotation: Annotation): string {
  if (!annotation || annotation === null) return 'Any'
  
  // Handle ExprName type
  if (annotation.cls === 'ExprName' && 'name' in annotation && annotation.name) {
    return annotation.name
  }
  
  // Handle ExprSubscript type (e.g., Callable[[ndarray], float])
  if (annotation.cls === 'ExprSubscript' && 'left' in annotation && 'slice' in annotation) {
    const left = annotationToString(annotation.left)
    const slice = annotationToString(annotation.slice)
    return `${left}[${slice}]`
  }
  
  // Handle ExprTuple type
  if (annotation.cls === 'ExprTuple' && 'elements' in annotation && annotation.elements) {
    const elements = annotation.elements.map(e => annotationToString(e)).join(', ')
    return elements
  }
  
  // Handle ExprList type
  if (annotation.cls === 'ExprList' && 'elements' in annotation && annotation.elements) {
    const elements = annotation.elements.map(e => annotationToString(e)).join(', ')
    return `[${elements}]`
  }
  
  return 'Any'
}

/**
 * Transform Griffe parameters to our Parameter interface
 */
function transformParameters(parameters: GriffeParameter[]): Parameter[] {
  return parameters
    .filter(p => p.name !== 'self')
    .map(p => ({
      name: p.name,
      annotation: annotationToString(p.annotation),
      description: '', // Griffe doesn't include param descriptions in parameters
      default: p.default || undefined
    }))
}

/**
 * Process a class member from Griffe output
 */
function processClass(classMember: GriffeMember, className: string): Class {
  const classData: Class = {
    name: className,
    docstring: classMember.docstring?.value || '',
    bases: (classMember.bases || []).map(b => b.name),
    parameters: [],
    methods: [],
    attributes: []
  }

  // Extract __init__ parameters
  if (classMember.members && '__init__' in classMember.members) {
    const initMethod = classMember.members['__init__']
    if (initMethod.parameters) {
      classData.parameters = transformParameters(initMethod.parameters)
    }
  }

  // Extract methods
  if (classMember.members) {
    for (const [methodName, method] of Object.entries(classMember.members)) {
      if (method.kind === 'function' && !methodName.startsWith('_')) {
        classData.methods.push({
          name: methodName,
          docstring: method.docstring?.value || '',
          parameters: method.parameters ? transformParameters(method.parameters) : [],
          returns: annotationToString(method.returns || null)
        })
      }
    }
  }

  // Extract attributes
  if (classMember.members) {
    for (const [attrName, attr] of Object.entries(classMember.members)) {
      if (attr.kind === 'attribute') {
        classData.attributes.push({
          name: attrName,
          annotation: annotationToString(attr.annotation || null),
          description: attr.docstring?.value || ''
        })
      }
    }
  }

  return classData
}

/**
 * Transform a category module from Griffe output
 */
function transformCategoryModule(categoryModule: GriffeMember): Module {
  const classes: Class[] = []
  const functions: Method[] = []

  // Iterate through submodules (e.g., ant_colony, particle_swarm)
  if (categoryModule.members) {
    for (const [submoduleName, submodule] of Object.entries(categoryModule.members)) {
      // Skip non-module members
      if (submodule.kind !== 'module') continue

      // Process classes in the submodule
      if (submodule.members) {
        for (const [memberName, member] of Object.entries(submodule.members)) {
          if (member.kind === 'class') {
            classes.push(processClass(member, memberName))
          } else if (member.kind === 'function') {
            functions.push({
              name: memberName,
              docstring: member.docstring?.value || '',
              parameters: member.parameters ? transformParameters(member.parameters) : [],
              returns: annotationToString(member.returns || null)
            })
          }
        }
      }
    }
  }

  return {
    name: categoryModule.name,
    docstring: categoryModule.docstring?.value || '',
    classes,
    functions
  }
}

export default defineLoader({
  watch: ['../../api/*.json'],
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
    const classIndex: Record<string, { category: string; module: string; class: Class }> = {}

    // We only need to load one JSON file since they're all identical
    // They all contain the full API data
    try {
      const jsonPath = resolve(__dirname, '../../api/full_api.json')
      const rawData = readFileSync(jsonPath, 'utf-8')
      const data: GriffeOutput = JSON.parse(rawData)

      // Navigate to opt.members to find category modules
      const optMembers = data.opt.members

      for (const category of categories) {
        if (optMembers && category in optMembers) {
          const categoryModule = optMembers[category]
          if (categoryModule.kind === 'module') {
            modules[category] = transformCategoryModule(categoryModule)
            
            // Build class index for quick lookup
            for (const cls of modules[category].classes) {
              const key = cls.name.toLowerCase()
              classIndex[key] = {
                category,
                module: category,
                class: cls
              }
            }
          }
        } else {
          console.warn(`Category ${category} not found in API data`)
          modules[category] = {
            name: category,
            docstring: '',
            classes: [],
            functions: []
          }
        }
      }
    } catch (error) {
      console.error('Failed to load API data:', error)
      // Return empty data structure on error
      for (const category of categories) {
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
      categories,
      classIndex
    }
  }
})
