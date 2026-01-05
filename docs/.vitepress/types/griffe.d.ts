export interface DocstringSection {
  kind: 'text' | 'parameters' | 'returns' | 'examples'
  value: string | Parameter[]
}

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
  members: GriffeMember[]
  bases: string[]
}
