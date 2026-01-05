import fs from 'node:fs'
import { basename, resolve } from 'node:path'
import { defineLoader } from 'vitepress'
import type { DocstringSection, GriffeClass, GriffeMember, Parameter } from '../types/griffe'

export interface APIAttribute {
  name: string
  annotation: string
  docstring: { parsed: DocstringSection[] }
}

export interface APIMethod extends GriffeMember {
  parameters: Parameter[]
  returns?: string
}

export interface APIClassDoc extends GriffeClass {
  parameters: Parameter[]
  methods: APIMethod[]
  attributes: APIAttribute[]
  example?: string
}

export interface APIData {
  categories: Record<string, APIClassDoc[]>
  totalClasses: number
}

const CATEGORY_FILES = [
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

const CATEGORY_SET = new Set(CATEGORY_FILES)

const normalizeDocstring = (docstring: any): { parsed: DocstringSection[] } => {
  const value = typeof docstring === 'string' ? docstring : docstring?.value
  return {
    parsed: value ? [{ kind: 'text', value }] : []
  }
}

const renderType = (annotation: any): string => {
  if (!annotation) return ''
  if (typeof annotation === 'string' || typeof annotation === 'number' || typeof annotation === 'boolean') {
    return String(annotation)
  }

  if (Array.isArray(annotation)) {
    return annotation.map((item) => renderType(item)).filter(Boolean).join(', ')
  }

  switch (annotation.cls) {
    case 'ExprName':
      return annotation.name || ''
    case 'ExprAttribute':
      return (annotation.values || []).map((value: any) => renderType(value)).filter(Boolean).join('.')
    case 'ExprSubscript':
      return `${renderType(annotation.left)}[${renderType(annotation.slice)}]`
    case 'ExprTuple':
    case 'ExprList':
      return (annotation.elements || []).map((el: any) => renderType(el)).filter(Boolean).join(', ')
    case 'ExprKeyword':
      return renderType(annotation.value || annotation.name)
    case 'ExprBinOp':
      return `${renderType(annotation.left)} ${annotation.operator ?? ''} ${renderType(annotation.right)}`
    default:
      return annotation.name || annotation.cls || ''
  }
}

const buildParameters = (params: any[] = []): Parameter[] =>
  params
    .filter((param) => param?.name && param.name !== 'self')
    .map((param) => ({
      name: param.name,
      annotation: renderType(param.annotation) || 'Any',
      description: param.description || '',
      default: renderType(param.default)
    }))

const buildSignature = (parameters: Parameter[], returns?: string): string => {
  const paramText = parameters.map((param) => `${param.name}: ${param.annotation}${param.default ? ` = ${param.default}` : ''}`).join(', ')
  return `${paramText ? `(${paramText})` : '()'}${returns ? ` -> ${returns}` : ''}`
}

const transformFunction = (name: string, fn: any): APIMethod => {
  const parameters = buildParameters(Array.isArray(fn.parameters) ? fn.parameters : [])
  const returns = renderType(fn.returns)
  return {
    kind: 'function',
    name,
    signature: buildSignature(parameters, returns),
    docstring: normalizeDocstring(fn.docstring),
    parameters,
    returns
  }
}

const transformClass = (cls: any): APIClassDoc => {
  const members = cls.members || {}
  const initMethod = members.__init__ || {}
  const parameters = buildParameters(Array.isArray(initMethod.parameters) ? initMethod.parameters : [])

  const methods: APIMethod[] = Object.entries(members)
    .filter(([methodName, method]) => (method as any)?.kind === 'function' && methodName !== '__init__')
    .map(([methodName, method]) => transformFunction(methodName, method))

  const attributes: APIAttribute[] = Object.entries(members)
    .filter(([, member]) => (member as any)?.kind === 'attribute')
    .map(([attrName, attr]) => ({
      name: attrName,
      annotation: renderType((attr as any).annotation) || 'Any',
      docstring: normalizeDocstring((attr as any).docstring)
    }))

  return {
    name: cls.name,
    docstring: normalizeDocstring(cls.docstring),
    bases: (cls.bases || []).map((base: any) => renderType(base)).filter(Boolean),
    parameters,
    methods,
    attributes
  }
}

const collectClasses = (member: any): APIClassDoc[] => {
  if (!member) return []
  if (member.kind === 'class') {
    return [transformClass(member)]
  }
  if (member.kind === 'module') {
    return Object.values(member.members || {}).flatMap((child) => collectClasses(child))
  }
  return []
}

const transformGriffeToAPI = (data: any): APIClassDoc[] => {
  if (!data) return []
  const members = data.members || {}
  return Object.values(members).flatMap((member) => collectClasses(member))
}

const loadAPIData = async (watchedFiles?: string[]): Promise<APIData> => {
  const filePaths =
    watchedFiles && watchedFiles.length > 0
      ? watchedFiles.filter(Boolean)
      : CATEGORY_FILES.map((category) => resolve(__dirname, `../../api/${category}.json`))

  const categories: Record<string, APIClassDoc[]> = {}

  for (const file of filePaths) {
    const category = basename(file, '.json')
    if (!CATEGORY_SET.has(category)) continue

    try {
      const rawData = await fs.promises.readFile(file, 'utf-8')
      const data = JSON.parse(rawData)
      const packageData =
        data?.opt?.members?.[category] || data?.opt?.[category] || data[`opt.${category}`] || data?.opt || data
      categories[category] = transformGriffeToAPI(packageData)
    } catch (error) {
      console.warn(`Failed to load API data for ${category}:`, error)
      categories[category] = []
    }
  }

  const totalClasses = Object.values(categories).reduce((sum, list) => sum + list.length, 0)
  return {
    categories,
    totalClasses
  }
}

export default defineLoader({
  watch: ['../../api/*.json'],
  load: loadAPIData
})

declare const data: Awaited<ReturnType<typeof loadAPIData>>
export { data }
export type { APIClassDoc, Parameter }
