<script setup lang="ts">
import { computed } from 'vue'

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

export interface ClassDoc {
  name: string
  docstring: string
  bases: string[]
  parameters: Parameter[]
  methods: Method[]
  attributes: Array<{ name: string; annotation: string; description: string }>
  example?: string
}

const props = defineProps<{
  classDoc: ClassDoc
}>()

const hasParameters = computed(() => props.classDoc.parameters && props.classDoc.parameters.length > 0)
const hasMethods = computed(() => props.classDoc.methods && props.classDoc.methods.length > 0)
const hasAttributes = computed(() => props.classDoc.attributes && props.classDoc.attributes.length > 0)
const hasExample = computed(() => props.classDoc.example && props.classDoc.example.length > 0)

// Extract short description (first paragraph)
const shortDescription = computed(() => {
  const docstring = props.classDoc.docstring || ''
  const firstParagraph = docstring.split('\n\n')[0]
  return firstParagraph.trim()
})

// Extract long description (remaining paragraphs)
const longDescription = computed(() => {
  const docstring = props.classDoc.docstring || ''
  const paragraphs = docstring.split('\n\n')
  if (paragraphs.length > 1) {
    return paragraphs.slice(1).join('\n\n').trim()
  }
  return ''
})
</script>

<template>
  <div class="api-doc">
    <div class="api-header">
      <h2 class="api-title">{{ classDoc.name }}</h2>
      <div v-if="classDoc.bases && classDoc.bases.length > 0" class="api-bases">
        <span class="base-label">Extends:</span>
        <code class="base-class" v-for="base in classDoc.bases" :key="base">{{ base }}</code>
      </div>
    </div>

    <div v-if="shortDescription" class="api-description">
      <p>{{ shortDescription }}</p>
    </div>

    <div v-if="longDescription" class="api-details">
      <p>{{ longDescription }}</p>
    </div>

    <div v-if="hasParameters" class="api-section">
      <h3>Parameters</h3>
      <div class="parameter-table">
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
              <th>Default</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="param in classDoc.parameters" :key="param.name">
              <td><code>{{ param.name }}</code></td>
              <td><code class="type-annotation">{{ param.annotation }}</code></td>
              <td>
                <code v-if="param.default" class="default-value">{{ param.default }}</code>
                <span v-else class="required-marker">Required</span>
              </td>
              <td>{{ param.description }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div v-if="hasAttributes" class="api-section">
      <h3>Attributes</h3>
      <div class="attributes-list">
        <div v-for="attr in classDoc.attributes" :key="attr.name" class="attribute-item">
          <div class="attribute-signature">
            <code class="attribute-name">{{ attr.name }}</code>
            <span class="attribute-type">: <code>{{ attr.annotation }}</code></span>
          </div>
          <p class="attribute-description">{{ attr.description }}</p>
        </div>
      </div>
    </div>

    <div v-if="hasMethods" class="api-section">
      <h3>Methods</h3>
      <div class="methods-list">
        <div v-for="method in classDoc.methods" :key="method.name" class="method-item">
          <div class="method-signature">
            <code class="method-name">{{ method.name }}</code>
            <span class="method-params">(
              <span v-for="(param, idx) in method.parameters" :key="param.name">
                <span class="param-name">{{ param.name }}</span>: <span class="param-type">{{ param.annotation }}</span>
                <span v-if="param.default"> = {{ param.default }}</span>
                <span v-if="idx < method.parameters.length - 1">, </span>
              </span>
            )</span>
            <span class="method-returns"> → <code>{{ method.returns }}</code></span>
          </div>
          <p class="method-description">{{ method.docstring }}</p>
        </div>
      </div>
    </div>

    <div v-if="hasExample" class="api-section">
      <h3>Example</h3>
      <div class="example-code">
        <pre><code class="language-python">{{ classDoc.example }}</code></pre>
      </div>
    </div>
  </div>
</template>

<style scoped>
.api-doc {
  max-width: 100%;
  padding: 1rem 0;
}

.api-header {
  margin-bottom: 1.5rem;
}

.api-title {
  margin: 0 0 0.5rem 0;
  font-size: 2rem;
  font-weight: 700;
  border-bottom: none;
}

.api-bases {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
}

.base-label {
  font-weight: 500;
}

.base-class {
  background: var(--vp-c-bg-soft);
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  font-size: 0.85rem;
}

.api-description {
  font-size: 1.1rem;
  margin-bottom: 1.5rem;
  color: var(--vp-c-text-1);
}

.api-details {
  margin-bottom: 1.5rem;
  color: var(--vp-c-text-2);
}

.api-section {
  margin: 2rem 0;
}

.api-section h3 {
  font-size: 1.5rem;
  margin-bottom: 1rem;
  border-bottom: 1px solid var(--vp-c-divider);
  padding-bottom: 0.5rem;
}

.parameter-table table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 1rem;
}

.parameter-table th,
.parameter-table td {
  text-align: left;
  padding: 0.75rem;
  border: 1px solid var(--vp-c-divider);
}

.parameter-table th {
  background: var(--vp-c-bg-soft);
  font-weight: 600;
}

.parameter-table code {
  background: var(--vp-c-bg-soft);
  padding: 0.2rem 0.4rem;
  border-radius: 3px;
  font-size: 0.9em;
}

.type-annotation {
  color: var(--vp-c-brand);
}

.default-value {
  color: var(--vp-c-green);
}

.required-marker {
  color: var(--vp-c-red);
  font-style: italic;
  font-size: 0.9em;
}

.attributes-list,
.methods-list {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.attribute-item,
.method-item {
  padding: 1rem;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
  border-left: 3px solid var(--vp-c-brand);
}

.attribute-signature,
.method-signature {
  font-family: var(--vp-font-family-mono);
  font-size: 1rem;
  margin-bottom: 0.5rem;
  color: var(--vp-c-text-1);
}

.attribute-name,
.method-name {
  font-weight: 600;
  color: var(--vp-c-brand);
}

.attribute-type,
.param-type {
  color: var(--vp-c-text-2);
}

.method-params {
  color: var(--vp-c-text-1);
}

.param-name {
  font-style: italic;
}

.method-returns {
  color: var(--vp-c-text-2);
}

.attribute-description,
.method-description {
  margin: 0;
  color: var(--vp-c-text-2);
  line-height: 1.6;
}

.example-code {
  margin-top: 1rem;
}

.example-code pre {
  background: var(--vp-c-bg-soft);
  padding: 1rem;
  border-radius: 6px;
  overflow-x: auto;
}

.example-code code {
  font-family: var(--vp-font-family-mono);
  font-size: 0.9rem;
  line-height: 1.6;
}
</style>
