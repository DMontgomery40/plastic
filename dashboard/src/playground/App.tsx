import { useEffect } from 'react';
import { Empty } from '../components/panels/Empty';
import { ChatScreen } from './chat/ChatScreen';
import { Header } from './Header';
import { SessionsScreen } from './sessions/SessionsScreen';
import { SignalsScreen } from './signals/SignalsScreen';
import { TABS, useStore } from './store';
import { Button, ErrorBanner } from './ui';

function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName) || target.isContentEditable;
}

export default function App() {
  const tab = useStore((s) => s.tab);
  const setTab = useStore((s) => s.setTab);
  const error = useStore((s) => s.error);
  const clearError = useStore((s) => s.clearError);
  const bootstrap = useStore((s) => s.bootstrap);
  const health = useStore((s) => s.health);
  const healthChecked = useStore((s) => s.healthChecked);
  const health_public = health?.public ?? false;

  useEffect(() => {
    void bootstrap();
  }, [bootstrap]);

  // keys 1–3 switch screens when not typing
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || isTyping(e.target)) return;
      const n = Number(e.key);
      if (n >= 1 && n <= TABS.length) {
        e.preventDefault();
        setTab(TABS[n - 1]);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setTab]);

  const offline = healthChecked && health === null;
  const View = tab === 'chat' ? ChatScreen : tab === 'signals' ? SignalsScreen : SessionsScreen;

  return (
    <div className="min-h-screen bg-surface">
      <Header />
      <main className="mx-auto max-w-[1500px] px-4 py-4">
        {error ? (
          <div className="mb-4">
            <ErrorBanner message={error} onDismiss={clearError} />
          </div>
        ) : null}
        {offline ? (
          <Empty
            title="Connection unavailable."
            detail={health_public ? 'Try again in a moment.' : 'Start the service and reconnect.'}
            command={health_public ? undefined : 'uv run plastic serve --artifacts-root artifacts --port 13579 --device cpu'}
          >
            <Button onClick={() => void bootstrap()}>Retry</Button>
          </Empty>
        ) : (
          <div key={tab} className="tab-enter">
            <View />
          </div>
        )}
      </main>
    </div>
  );
}
