import { afterEach, describe, expect, it, vi } from 'vitest';
import { useStore } from './store';

const initial = useStore.getState();
afterEach(() => { vi.unstubAllGlobals(); useStore.setState(initial, true); });

const model = (id: string) => ({ model_id: id, backend: 'plastic', domain: 'text', status: 'completed', params: 1, calibrated: false, has_canary: false, created_at_unix: 0, updated_at_unix: 0 });
const session = (id: string, m: string) => ({ session_id: id, model_id: m, domain: 'text', pos: 0, n_transactions: 0, commits: 0, rollbacks: 0 });

describe('bootstrap', () => {
  it('opens the first catalog model’s session, not the session that sorts first', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url.endsWith('/api/health')) return new Response(JSON.stringify({ ok: true, public: true, capabilities: {} }));
      if (url.endsWith('/api/models')) return new Response(JSON.stringify([model('qwen_pinned'), model('lm_wikitext_l4')]));
      if (url.endsWith('/api/sessions')) return new Response(JSON.stringify([session('demo_core', 'lm_wikitext_l4'), session('demo_text', 'qwen_pinned')]));
      return new Response('{}', { status: 404 });
    }));
    await useStore.getState().bootstrap();
    expect(useStore.getState().currentSessionId).toBe('demo_text');
  });
});

describe('long sessions', () => {
  it('loads the newest chunks, not the first page of the log', async () => {
    const rec = (i: number) => ({ index: i });
    const requested: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      requested.push(url);
      if (url.includes('/transactions')) {
        const q = new URL(url, 'http://x').searchParams;
        const offset = Number(q.get('offset')), limit = Number(q.get('limit'));
        const all = Array.from({ length: 650 }, (_, i) => rec(i));
        return new Response(JSON.stringify({ total: 650, items: all.slice(offset, offset + limit) }));
      }
      if (url.endsWith('/state')) return new Response(JSON.stringify({ pos: 0 }));
      return new Response(JSON.stringify({ meta: { session_id: 's' }, summary: {}, transactions: [], trace: [] }));
    }));
    useStore.setState({ currentSessionId: 's' });
    await useStore.getState().reloadCurrent();
    const tx = useStore.getState().transactions as unknown as { index: number }[];
    expect(tx).toHaveLength(400);
    expect(tx[0].index).toBe(250);
    expect(tx[tx.length - 1].index).toBe(649);
  });
});
