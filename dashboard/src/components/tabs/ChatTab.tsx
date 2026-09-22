import { useState } from 'react';
import { useStore } from '../../store';
import { fmt, fmtInt, fmtRelative } from '../../utils/formatting';
import { Button, DecisionBadge, Empty, Field, KeyValue, NumberInput, Panel } from '../panels';

export function ChatTab() {
  const sessions = useStore((s) => s.sessions);
  const currentSessionId = useStore((s) => s.currentSessionId);
  const setCurrentSession = useStore((s) => s.setCurrentSession);
  const sessionDetail = useStore((s) => s.sessionDetail);
  const chatResult = useStore((s) => s.chatResult);
  const sendChat = useStore((s) => s.sendChat);
  const busy = useStore((s) => s.loading.chat);

  const [prompt, setPrompt] = useState('');
  const [maxNewTokens, setMaxNewTokens] = useState(128);
  const [temperature, setTemperature] = useState(0.9);
  const [topK, setTopK] = useState(50);
  const [seed, setSeed] = useState(0);

  const textSessions = sessions.filter((s) => s.domain === 'text');
  const domain = sessionDetail?.meta.domain;

  if (textSessions.length === 0) {
    return (
      <Empty
        title="No text session."
        detail="Chat runs the prompt through the harness one chunk at a time, so it needs a session on a text model."
        command={'uv run plastic train text --data artifacts/data/wikitext --steps 3000 --device mps\nuv run plastic session new --model <model_id>'}
      />
    );
  }

  // The API returns 400 for a chat request against a physics session, so offer
  // the switch instead of sending a request that cannot succeed.
  if (domain === 'physics') {
    return (
      <Empty
        title={`Session ${currentSessionId} is a physics session.`}
        detail="Chat is only defined for text sessions. Pick a text session in the header, or use one of these."
      >
        <div className="flex flex-wrap gap-2">
          {textSessions.map((s) => (
            <Button key={s.session_id} size="sm" onClick={() => setCurrentSession(s.session_id)}>
              {s.session_id}
            </Button>
          ))}
        </div>
      </Empty>
    );
  }

  const turns = (sessionDetail?.trace ?? []).filter((t) => t.kind === 'chat').slice().reverse();

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
      <div className="space-y-4">
        <Panel title="Prompt" subtitle="Prompt tokens are learned through transactions; generated tokens are read-only by default.">
          <label htmlFor="chat-prompt" className="mb-1 block text-label font-semibold uppercase tracking-wide text-ink-muted">
            Prompt text
          </label>
          <textarea
            id="chat-prompt"
            className="h-28 w-full rounded border border-edge bg-surface-overlay px-3 py-2 text-base text-ink-primary placeholder:text-ink-muted focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent"
            value={prompt}
            placeholder="Type a prompt. Every chunk boundary inside it produces one transaction."
            onChange={(e) => setPrompt(e.target.value)}
          />
          <div className="mt-3 flex items-center gap-3">
            <Button variant="primary" disabled={busy || prompt.trim().length === 0} onClick={() => void sendChat({ prompt, max_new_tokens: maxNewTokens, temperature, top_k: topK, seed })}>
              {busy ? 'Running…' : 'Send'}
            </Button>
            <span className="text-xs text-ink-muted">Session {currentSessionId}</span>
          </div>
        </Panel>

        {chatResult ? (
          <>
            <Panel
              title="Completion"
              subtitle={`${fmtInt(chatResult.n_tokens_in)} tokens in, ${fmtInt(chatResult.n_tokens_out)} out`}
            >
              <pre className="max-h-72 overflow-auto whitespace-pre-wrap rounded border border-edge bg-surface-inset px-3 py-2.5 font-mono text-sm text-ink-primary">
                {chatResult.completion || '(the model produced no tokens)'}
              </pre>
            </Panel>

            <Panel title="Transactions of this turn" subtitle="Every chunk boundary crossed while the turn was processed.">
              {chatResult.transactions.length === 0 ? (
                <Empty
                  title="No chunk boundary was crossed."
                  detail={`The turn was shorter than one chunk, so nothing was committed or refused. ${fmtInt(chatResult.summary.pending)} tokens are pending.`}
                />
              ) : (
                <ul className="space-y-2">
                  {chatResult.transactions.map((tx) => (
                    <li key={tx.index} className="rounded border border-edge bg-surface-overlay px-3 py-2">
                      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
                        <DecisionBadge kind={tx.decision.kind} size="sm" />
                        {tx.requested.kind !== tx.decision.kind ? (
                          <span className="font-mono text-micro text-status-scale">requested {tx.requested.kind}</span>
                        ) : null}
                        <span className="font-mono text-xs text-ink-secondary">chunk {tx.index}</span>
                        <span className="font-mono text-xs text-ink-secondary">
                          pos {tx.pos_start}–{tx.pos_end}
                        </span>
                        <span className="font-mono text-xs text-ink-secondary">loss {fmt(tx.signals.chunk_loss, 4)}</span>
                        <span className="font-mono text-xs text-ink-secondary">β {fmt(tx.signals.beta_mean, 4)}</span>
                        <span className="font-mono text-xs text-status-scale">‖Δ‖ proposed {fmt(tx.signals.delta_norm, 4)}</span>
                        <span className="font-mono text-xs text-status-commit">‖Δ‖ accepted {fmt(tx.accepted?.delta_norm, 4)}</span>
                      </div>
                      {tx.decision.reasons.length > 0 ? (
                        <ul className="mt-1.5 flex flex-wrap gap-1.5">
                          {tx.decision.reasons.map((reason, i) => (
                            <li key={`${reason}-${i}`} className="rounded border border-edge bg-surface-inset px-1.5 py-0.5 font-mono text-micro text-ink-secondary">
                              {reason}
                            </li>
                          ))}
                        </ul>
                      ) : null}
                    </li>
                  ))}
                </ul>
              )}
            </Panel>
          </>
        ) : (
          <Empty
            title="No turn in this view yet."
            detail="Send a prompt above, or replay one from the command line."
            command={`uv run plastic chat ${currentSessionId ?? '<session_id>'} "a first prompt"`}
          />
        )}
      </div>

      <div className="space-y-4">
        <Panel title="Sampling">
          <div className="space-y-3">
            <Field label="Max new tokens" htmlFor="chat-max">
              <NumberInput id="chat-max" value={maxNewTokens} min={1} max={2048} step={8} onChange={setMaxNewTokens} />
            </Field>
            <Field label="Temperature" htmlFor="chat-temp">
              <NumberInput id="chat-temp" value={temperature} min={0.01} max={2} step={0.05} onChange={setTemperature} />
            </Field>
            <Field label="Top-k" htmlFor="chat-topk" hint="0 disables the cutoff.">
              <NumberInput id="chat-topk" value={topK} min={0} max={4096} step={1} onChange={setTopK} />
            </Field>
            <Field label="Seed" htmlFor="chat-seed">
              <NumberInput id="chat-seed" value={seed} min={0} step={1} onChange={setSeed} />
            </Field>
          </div>
        </Panel>

        {chatResult ? (
          <Panel title="Runner after the turn">
            <KeyValue
              rows={[
                { label: 'Position', value: fmtInt(chatResult.summary.pos) },
                { label: 'Pending tokens', value: fmtInt(chatResult.summary.pending) },
                { label: 'Transactions', value: fmtInt(chatResult.summary.n_transactions) },
                { label: 'Budget used', value: fmt(chatResult.summary.budget_used, 4) },
                { label: 'Read-only', value: chatResult.summary.read_only ? 'yes' : 'no', note: chatResult.summary.read_only_reason ?? undefined },
              ]}
            />
          </Panel>
        ) : null}

        <Panel title="Earlier turns" subtitle="From the session trace, newest first.">
          {turns.length === 0 ? (
            <p className="text-sm text-ink-secondary">This session has no recorded chat turns.</p>
          ) : (
            <ul className="space-y-2">
              {turns.map((t, i) => (
                <li key={`${t.t_unix}-${i}`} className="rounded border border-edge bg-surface-overlay px-2.5 py-2">
                  <p className="font-mono text-xs text-ink-primary">{t.prompt ?? '(no prompt recorded)'}</p>
                  <p className="mt-1 text-micro text-ink-muted">
                    {fmtRelative(t.t_unix)} · pos {fmtInt(t.pos_end)} · {fmtInt(t.n_transactions)} tx
                  </p>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </div>
  );
}
