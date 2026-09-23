import { useCallback, useEffect, useState } from 'react';
import { Empty } from '../components/panels/Empty';
import { Button } from '../playground/ui';
import { loadIndex } from './data';
import { RunsView } from './RunsView';
import type { ObservatoryIndex } from './types';

export const VIEWS = ['runs'] as const;
export type View = (typeof VIEWS)[number];
const VIEW_LABEL: Record<View, string> = { runs: 'Runs' };

export interface Route {
  view: View;
  run: string | null;
  arm: string | null;
}

/** #sleep/<view>/<run>/<arm>: every screen state has a shareable link. */
export function parseRoute(hash: string): Route {
  const [head, view, run, arm] = hash.replace(/^#/, '').split('/');
  return {
    view: head === 'sleep' && (VIEWS as readonly string[]).includes(view) ? (view as View) : 'runs',
    run: run ? decodeURIComponent(run) : null,
    arm: arm ? decodeURIComponent(arm) : null,
  };
}

export function formatRoute(r: Route): string {
  return ['#sleep', r.view, r.run, r.run ? r.arm : null].filter(Boolean).map((p, i) => (i === 0 ? p : encodeURIComponent(p as string))).join('/');
}

export function ObservatoryScreen() {
  const [index, setIndex] = useState<ObservatoryIndex | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [route, setRoute] = useState<Route>(() => parseRoute(window.location.hash));

  const load = useCallback(() => {
    setError(null);
    loadIndex().then(setIndex, (e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);
  useEffect(load, [load]);

  useEffect(() => {
    const onHash = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const go = useCallback((patch: Partial<Route>) => {
    setRoute((prev) => {
      const next = { ...prev, ...patch };
      window.history.replaceState(null, '', formatRoute(next));
      return next;
    });
  }, []);

  if (error) {
    return (
      <Empty title="Sleep runs could not be loaded." detail={error}>
        <Button onClick={load}>Retry</Button>
      </Empty>
    );
  }
  if (!index) return <p className="py-8 text-sm text-ink-secondary">Loading Sleep runs…</p>;

  const runId = route.run && index.runs.some((r) => r.id === route.run) ? route.run : null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-ink-primary">Sleep</h1>
          <p className="mt-1 max-w-3xl text-base text-ink-secondary">
            Can what a model learns during a conversation become a lasting change to the model? Each run teaches facts in a session, consolidates the accepted learning into a candidate child model, then tests it from a fresh start.
          </p>
        </div>
        {VIEWS.length > 1 ? (
        <nav aria-label="Sleep views" className="flex rounded-lg border border-edge-strong bg-surface-raised p-1">
          {VIEWS.map((v) => (
            <button
              key={v}
              type="button"
              aria-current={route.view === v ? 'page' : undefined}
              onClick={() => go({ view: v })}
              className={`rounded px-3 py-1.5 text-sm font-semibold focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                route.view === v ? 'bg-accent-soft text-accent' : 'text-ink-secondary hover:text-ink-primary'
              }`}
            >
              {VIEW_LABEL[v]}
            </button>
          ))}
        </nav>
        ) : null}
      </div>
      <RunsView index={index} runId={runId} arm={route.arm} onSelect={(run, arm) => go({ run, arm })} />
    </div>
  );
}
