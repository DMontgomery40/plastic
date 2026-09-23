import type { TabKey } from './store';

export const PUBLIC_TAB_KEYS = ['chat', 'session', 'sessions'] as const satisfies readonly TabKey[];

export function isPublicDemo(): boolean {
  return typeof document !== 'undefined' && document.body?.dataset.publicDemo === 'true';
}

export function visibleTabKeys(): readonly TabKey[] {
  return isPublicDemo() ? PUBLIC_TAB_KEYS : ['sessions', 'session', 'chat', 'physics', 'train', 'redteam', 'architecture'];
}

export function visibleTab(tab: TabKey): TabKey {
  return visibleTabKeys().includes(tab) ? tab : 'chat';
}

export function publicErrorMessage(error: string): string {
  if (/HTTP 409/.test(error) && /context limit/i.test(error)) return 'This shared session is full. Reset it in Sessions to continue.';
  if (/HTTP 503/.test(error)) return 'Another visitor is using the demo. Try again shortly.';
  if (/HTTP 409/.test(error)) return 'This session could not process the request. Try again or reset it in Sessions.';
  return 'Request failed. Try again.';
}
