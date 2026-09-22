import { TAB_KEYS, TAB_LABELS, useStore, type TabKey } from '../../store';

/** Text labels, no icons and no emoji. The digit is the keyboard shortcut. */
export function TabNav() {
  const activeTab = useStore((s) => s.activeTab);
  const setActiveTab = useStore((s) => s.setActiveTab);

  return (
    <nav aria-label="Sections" className="border-b border-edge bg-surface">
      <ul className="mx-auto flex max-w-[1700px] flex-wrap gap-1 px-5">
        {TAB_KEYS.map((key: TabKey, i) => {
          const active = key === activeTab;
          return (
            <li key={key}>
              <button
                type="button"
                onClick={() => setActiveTab(key)}
                aria-current={active ? 'page' : undefined}
                className={`-mb-px flex items-center gap-2 border-b-2 px-3 py-2.5 text-sm font-semibold transition-colors ${
                  active
                    ? 'border-accent text-accent'
                    : 'border-transparent text-ink-secondary hover:border-edge-strong hover:text-ink-primary'
                }`}
              >
                <span className="font-mono text-micro text-ink-muted">{i + 1}</span>
                {TAB_LABELS[key]}
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
