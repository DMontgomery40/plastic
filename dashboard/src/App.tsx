import { useEffect } from 'react';
import { Header } from './components/layout/Header';
import { TabNav } from './components/layout/TabNav';
import { ErrorBanner } from './components/panels';
import { SessionsTab } from './components/tabs/SessionsTab';
import { SessionTab } from './components/tabs/SessionTab';
import { ChatTab } from './components/tabs/ChatTab';
import { PhysicsTab } from './components/tabs/PhysicsTab';
import { TrainTab } from './components/tabs/TrainTab';
import { RedTeamTab } from './components/tabs/RedTeamTab';
import { ArchitectureTab } from './components/tabs/ArchitectureTab';
import { TAB_KEYS, stopJobPolling, syncJobPolling, useStore } from './store';

const TAB_VIEWS = {
  sessions: SessionsTab,
  session: SessionTab,
  chat: ChatTab,
  physics: PhysicsTab,
  train: TrainTab,
  redteam: RedTeamTab,
  architecture: ArchitectureTab,
} as const;

/** True while the user is typing, so digit shortcuts must not steal the key. */
function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable;
}

export default function App() {
  const activeTab = useStore((s) => s.activeTab);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const error = useStore((s) => s.error);
  const clearError = useStore((s) => s.clearError);
  const bootstrap = useStore((s) => s.bootstrap);
  const jobs = useStore((s) => s.jobs);

  useEffect(() => {
    void bootstrap();
    return stopJobPolling;
  }, [bootstrap]);

  // Poll running training jobs every 2 s; the loop stops itself when none runs.
  useEffect(() => {
    syncJobPolling(jobs);
  }, [jobs]);

  // Keys 1 to 7 select a tab. Nothing else is bound, so nothing is claimed that
  // is not implemented.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || isTyping(event.target)) return;
      const n = Number(event.key);
      if (!Number.isInteger(n) || n < 1 || n > TAB_KEYS.length) return;
      event.preventDefault();
      setActiveTab(TAB_KEYS[n - 1]);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [setActiveTab]);

  const View = TAB_VIEWS[activeTab];

  return (
    <div className="min-h-screen bg-surface">
      <Header />
      <TabNav />
      <main className="mx-auto max-w-[1700px] px-5 py-5">
        {error ? (
          <div className="mb-4">
            <ErrorBanner message={error} onDismiss={clearError} />
          </div>
        ) : null}
        <div key={activeTab} className="tab-enter">
          <View />
        </div>
      </main>
    </div>
  );
}
