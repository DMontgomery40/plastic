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
