<script setup lang="ts">
import { computed } from 'vue'
import { useData } from 'vitepress'
import { data as apiData } from '../../loaders/api.data'
import type { APIClassDoc } from '../../loaders/api.data'
import type { DocstringSection } from '../../types/griffe'

const props = defineProps<{
  classDoc?: APIClassDoc
  category?: string
  optimizer?: string
}>()

const { page } = useData()

const normalizeCategory = (value?: string) => (value ? value.replace(/-/g, '_') : '')

const resolvedClass = computed<APIClassDoc | undefined>(() => {
  if (props.classDoc) return props.classDoc

  const params = page.value?.params || {}
  const category = normalizeCategory(props.category || params.category || params.module)
  const optimizerName = props.optimizer || params.optimizer || params.name

  if (!category || !optimizerName) return undefined
  const candidates = apiData.categories?.[category] || []
  return candidates.find((entry) => entry.name === optimizerName)
})

const docstringText = computed(() => {
  const sections = resolvedClass.value?.docstring?.parsed || []
  return sections
    .map((section) => (typeof section.value === 'string' ? section.value : ''))
    .filter(Boolean)
    .join('\n\n')
})

const firstTextSection = (sections: DocstringSection[] = []) => {
  const section = sections.find((entry) => entry.kind !== 'parameters')
  return section && typeof section.value === 'string' ? section.value : ''
}

const hasParameters = computed(() => (resolvedClass.value?.parameters?.length || 0) > 0)
const hasMethods = computed(() => (resolvedClass.value?.methods?.length || 0) > 0)
const hasAttributes = computed(() => (resolvedClass.value?.attributes?.length || 0) > 0)
const hasExample = computed(() => resolvedClass.value?.example && resolvedClass.value.example.length > 0)

const shortDescription = computed(() => docstringText.value.split('\n\n')[0]?.trim() || '')

const longDescription = computed(() => {
  const paragraphs = docstringText.value.split('\n\n')
  if (paragraphs.length > 1) {
    return paragraphs.slice(1).join('\n\n').trim()
  }
  return ''
})
</script>

<template>
  <div v-if="resolvedClass" class="api-doc">
    <div class="api-header">
      <h2 class="api-title">{{ resolvedClass.name }}</h2>
      <div v-if="resolvedClass.bases && resolvedClass.bases.length > 0" class="api-bases">
        <span class="base-label">Extends:</span>
        <code class="base-class" v-for="base in resolvedClass.bases" :key="base">{{ base }}</code>
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
            <tr v-for="param in resolvedClass.parameters" :key="param.name">
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
        <div v-for="attr in resolvedClass.attributes" :key="attr.name" class="attribute-item">
          <div class="attribute-signature">
            <code class="attribute-name">{{ attr.name }}</code>
            <span class="attribute-type">: <code>{{ attr.annotation }}</code></span>
          </div>
          <p class="attribute-description">
            {{ firstTextSection(attr.docstring?.parsed) }}
          </p>
        </div>
      </div>
    </div>

    <div v-if="hasMethods" class="api-section">
      <h3>Methods</h3>
      <div class="methods-list">
        <div v-for="method in resolvedClass.methods" :key="method.name" class="method-item">
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
          <p class="method-description">
            {{ firstTextSection(method.docstring?.parsed) }}
          </p>
        </div>
      </div>
    </div>

    <div v-if="hasExample" class="api-section">
      <h3>Example</h3>
      <div class="example-code">
        <pre><code class="language-python">{{ resolvedClass.example }}</code></pre>
      </div>
    </div>
  </div>
  <div v-else class="api-doc">
    <p>API documentation for this optimizer is not available.</p>
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
