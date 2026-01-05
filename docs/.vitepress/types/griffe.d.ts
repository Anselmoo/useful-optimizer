export interface TextDocstringSection {
  kind: 'text' | 'returns' | 'examples'
  value: string
}

export interface ParametersDocstringSection {
  kind: 'parameters'
  value: Parameter[]
}

export type DocstringSection = TextDocstringSection | ParametersDocstringSection

export interface Parameter {
  name: string
  annotation?: string
  description?: string
  default?: string
}

export interface GriffeMember {
  kind: 'function' | 'attribute'
  name: string
  signature?: string
  docstring: { parsed: DocstringSection[] }
}

export interface GriffeClass {
  name: string
  docstring: { parsed: DocstringSection[] }
  members?: GriffeMember[]
  bases: string[]
}
