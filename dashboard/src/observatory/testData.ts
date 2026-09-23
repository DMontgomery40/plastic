// Test helper: serve the committed observatory export (dashboard/public/assets/observatory) through a stubbed fetch.
// Bundled with Vite's raw glob, so tests need no Node file APIs.
const FILES = import.meta.glob('../../public/assets/observatory/**/*.json', { eager: true, query: '?raw', import: 'default' }) as Record<string, string>;
const PREFIX = '../../public/assets/observatory/';

export function exportFiles(): string[] {
  return Object.keys(FILES).map((k) => k.slice(PREFIX.length));
}

export function readExport<T = unknown>(rel: string): T {
  const text = FILES[PREFIX + rel];
  if (text === undefined) throw new Error(`no exported file ${rel}`);
  return JSON.parse(text) as T;
}

export async function exportFetch(url: string | URL | Request): Promise<Response> {
  const path = String(url).split('assets/observatory/')[1] ?? '';
  const text = FILES[PREFIX + path];
  if (!path || path.includes('..') || text === undefined) return new Response('not found', { status: 404 });
  return new Response(text, { status: 200, headers: { 'Content-Type': 'application/json' } });
}
