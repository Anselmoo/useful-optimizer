/**
 * Loader for the published benchmark data (schema version 2).
 *
 *   index.json                    protocol, optimizers, available cells
 *   cells/<function>-<dim>.json   every optimizer on one function and dimension
 *   ranks.json                    average ranks over all blocks
 *   trajectory-<function>-2.json  best-so-far paths of 2D runs
 *
 * A page needs `index.json` plus one cell file, so the browser never downloads the
 * whole data set. Results are cached for the lifetime of the page; failed requests
 * are not cached so a later attempt can succeed.
 */
import { withBase } from 'vitepress'
import type { Cell, Index, Ranks, Trajectory } from '../types/benchmark'

const cache = new Map<string, Promise<unknown>>()

const fetchJson = async <T>(url: string): Promise<T> => {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.status} ${response.statusText}`)
  }
  // The dev server answers a missing file with index.html and status 200.
  if ((response.headers.get('content-type') ?? '').includes('text/html')) {
    throw new Error(`Failed to load ${url}: got HTML instead of JSON`)
  }
  return (await response.json()) as T
}

const getJson = <T>(path: string): Promise<T> => {
  const url = withBase(`/benchmarks/${path}`)
  let pending = cache.get(url) as Promise<T> | undefined
  if (!pending) {
    pending = fetchJson<T>(url)
    pending.catch(() => cache.delete(url))
    cache.set(url, pending)
  }
  return pending
}

export const loadIndex = () => getJson<Index>('index.json')

export const loadCell = (functionName: string, dim: number) =>
  getJson<Cell>(`cells/${functionName}-${dim}.json`)

export const loadRanks = () => getJson<Ranks>('ranks.json')

export const loadTrajectory = (functionName: string) =>
  getJson<Trajectory>(`trajectory-${functionName}-2.json`)
