// Loads the exported observatory JSON. The files ship with the dashboard build under assets/observatory/
// (a byte-identical mirror of docs/research/results/sleep-observatory/), so this works with the API offline.

import type { ObservatoryIndex, Run, Trajectory } from './types';

export const SCHEMA = 'plastic.sleep-observatory/1';

export class ObservatoryDataError extends Error {}

function base(): string {
  const root = typeof document !== 'undefined' ? document.baseURI : 'http://localhost/';
  return new URL('assets/observatory/', root).toString();
}

const cache = new Map<string, Promise<unknown>>();

async function getJson<T>(path: string, check: (value: unknown) => value is T): Promise<T> {
  const url = base() + path;
  if (!cache.has(url)) {
    cache.set(
      url,
      (async () => {
        const res = await fetch(url);
        if (!res.ok) throw new ObservatoryDataError(`${path}: HTTP ${res.status}`);
        return res.json();
      })().catch((err) => {
        cache.delete(url); // a failed fetch can be retried
        throw err;
      })
    );
  }
  const value = await cache.get(url);
  if (!check(value)) throw new ObservatoryDataError(`${path}: unexpected format`);
  return value;
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);

export function isIndex(v: unknown): v is ObservatoryIndex {
  return isObject(v) && v.schema === SCHEMA && Array.isArray(v.runs) && Array.isArray(v.sessions);
}

export function isRun(v: unknown): v is Run {
  return isObject(v) && typeof v.id === 'string' && Array.isArray(v.arms) && isObject(v.checkpoint) && isObject(v.protocol);
}

export function isTrajectory(v: unknown): v is Trajectory {
  return isObject(v) && v.schema === SCHEMA && Array.isArray(v.sessions) && isObject(v.per_layer);
}

const safeId = (id: string) => /^[A-Za-z0-9._-]+$/.test(id);

export function loadIndex(): Promise<ObservatoryIndex> {
  return getJson('index.json', isIndex);
}

export function loadRun(id: string): Promise<Run> {
  if (!safeId(id)) return Promise.reject(new ObservatoryDataError(`invalid run id ${id}`));
  return getJson(`runs/${id}.json`, isRun);
}

export function loadTrajectory(id: string): Promise<Trajectory> {
  if (!safeId(id)) return Promise.reject(new ObservatoryDataError(`invalid run id ${id}`));
  return getJson(`sessions/${id}.json`, isTrajectory);
}

/** Test hook: forget cached responses. */
export function clearObservatoryCache(): void {
  cache.clear();
}
