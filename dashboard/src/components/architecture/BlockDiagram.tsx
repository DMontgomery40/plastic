import type { ModelConfig } from '../../api/types';

const INK = '#e9eff5';
const INK2 = '#b8c4d0';
const MUTED = '#94a3b4';
const EDGE = '#3d4a59';
const SURFACE = '#151b23';
const OVERLAY = '#1c242e';
const SSM = '#58a6ff';
const MEM = '#3fd17a';
const MLP = '#c792ea';

interface BoxProps {
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  lines?: string[];
  accent?: string;
}

function Box({ x, y, w, h, title, lines = [], accent = EDGE }: BoxProps) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx="7" fill={OVERLAY} stroke={accent} strokeWidth="1.5" />
      <text x={x + 12} y={y + 20} fill={accent === EDGE ? INK : accent} fontSize="13" fontWeight="600">
        {title}
      </text>
      {lines.map((line, i) => (
        <text key={i} x={x + 12} y={y + 38 + i * 15} fill={INK2} fontSize="11.5" fontFamily="JetBrains Mono, monospace">
          {line}
        </text>
      ))}
    </g>
  );
}

function Arrow({ x, y1, y2, label }: { x: number; y1: number; y2: number; label?: string }) {
  return (
    <g>
      <line x1={x} y1={y1} x2={x} y2={y2 - 7} stroke={EDGE} strokeWidth="1.5" />
      <polygon points={`${x - 4.5},${y2 - 7} ${x + 4.5},${y2 - 7} ${x},${y2}`} fill={EDGE} />
      {label ? (
        <text x={x + 10} y={(y1 + y2) / 2 + 4} fill={MUTED} fontSize="11">
          {label}
        </text>
      ) : null}
    </g>
  );
}

/** One block of the stack, drawn from the live config of a trained model. */
export function BlockDiagram({ cfg }: { cfg: ModelConfig }) {
  const headDim = Math.floor(cfg.d_model / cfg.n_heads);
  const conv = cfg.conv_kernel > 1 ? `Conv_K, K = ${cfg.conv_kernel}` : 'no conv (K = 1)';
  const inputLabel =
    cfg.domain === 'text'
      ? `token ids, vocab ${cfg.vocab_size}${cfg.tie_embeddings ? ', tied head' : ''}`
      : `[obs ${cfg.obs_dim}, act ${cfg.act_dim}, reset 1] -> Linear(${cfg.obs_dim + cfg.act_dim + 1} to ${cfg.d_model})`;

  return (
    <div className="overflow-x-auto">
      <svg width="700" height="600" viewBox="0 0 700 600" role="img" aria-label="One plastic block">
        <rect x="0" y="0" width="700" height="600" fill={SURFACE} />

        <Box x={170} y={10} w={360} h={44} title="Input" lines={[inputLabel]} />
        <Arrow x={350} y1={54} y2={74} />

        <rect x="140" y="74" width="420" height="238" rx="10" fill="none" stroke={EDGE} strokeDasharray="4 4" />
        <text x="150" y="92" fill={MUTED} fontSize="11.5">
          block, repeated {cfg.n_layers} times
        </text>

        <Box
          x={170}
          y={102}
          w={360}
          h={62}
          title="SSM branch, selective diagonal recurrence"
          lines={[`u = SiLU(${conv}(RMSNorm(x)))`, `a = sigma(lam)^(${cfg.ssm_c} r),  h = a h + z,  D = ${cfg.d_model}`]}
          accent={SSM}
        />
        <Arrow x={350} y1={164} y2={186} label="residual" />

        <Box
          x={170}
          y={186}
          w={360}
          h={76}
          title={`Memory branch, fast weights (${cfg.rule}, ${cfg.memory})`}
          lines={[
            `S per head: ${cfg.n_heads} x ${headDim} x ${headDim}, input ${cfg.memory_input}`,
            'e = v - k (a S),  S = a S + b k^T e',
            'm = q S   (read after the token writes)',
          ]}
          accent={MEM}
        />
        <Arrow x={350} y1={262} y2={280} label="residual" />

        <Box x={170} y={280} w={360} h={26} title={`MLP, GELU, ${cfg.mlp_mult}x width`} accent={MLP} />

        <Arrow x={350} y1={312} y2={334} />
        <Box
          x={170}
          y={334}
          w={360}
          h={44}
          title="Output head"
          lines={[cfg.domain === 'text' ? 'next-token cross-entropy' : `Linear(${cfg.d_model} to ${cfg.obs_dim}), obs delta, MSE`]}
        />

        <line x1="20" y1="400" x2="680" y2="400" stroke={EDGE} strokeWidth="1" />
        <text x="20" y="424" fill={INK} fontSize="13" fontWeight="600">
          Session state, carried between chunks
        </text>
        <Box
          x={20}
          y={436}
          w={212}
          h={62}
          title="Memory S"
          lines={[`${cfg.n_layers} x ${cfg.n_heads} x ${headDim} x ${headDim}`, 'beta writes, alpha decays']}
          accent={MEM}
        />
        <Box x={244} y={436} w={212} h={62} title="Activation h" lines={[`${cfg.n_layers} x ${cfg.d_model}`, 'gated recurrence']} accent={SSM} />
        <Box
          x={468}
          y={436}
          w={212}
          h={62}
          title="Conv buffers"
          lines={[`${cfg.n_layers} x 2 x ${Math.max(0, cfg.conv_kernel - 1)} x ${cfg.d_model}`, 'chunked = recurrent']}
        />

        <text x="20" y="530" fill={INK} fontSize="13" fontWeight="600">
          Transaction granularity
        </text>
        <text x="20" y="550" fill={INK2} fontSize="11.5" fontFamily="JetBrains Mono, monospace">
          chunk L = {cfg.chunk} tokens, scan chunk = {cfg.scan_chunk}
        </text>
        <text x="20" y="568" fill={MUTED} fontSize="11.5">
          The scan chunk only sets the working set; results are identical for any value.
        </text>
        <text x="20" y="586" fill={MUTED} fontSize="11.5">
          Three state copies live per session: committed, working, and pending (under L).
        </text>
      </svg>
    </div>
  );
}

/** Signals to policy to one of five decisions. */
export function HarnessDiagram() {
  const decisions = [
    { label: 'commit', color: '#3fd17a', note: 'committed := working' },
    { label: 'scale', color: '#f0b429', note: 'reprocess with beta scaled' },
    { label: 'project', color: '#58a6ff', note: 'remove the canary-aligned part of the delta' },
    { label: 'rollback', color: '#ff6b6b', note: 'reprocess frozen: read, do not learn' },
    { label: 'readonly', color: '#94a3b4', note: 'observation: no write proposed (generation, spent budget, or latched alarm)' },
  ];

  return (
    <div className="overflow-x-auto">
      <svg width="980" height="330" viewBox="0 0 980 330" role="img" aria-label="The harness pipeline">
        <rect x="0" y="0" width="980" height="330" fill={SURFACE} />

        <Box
          x={16}
          y={20}
          w={246}
          h={180}
          title="Signals, all from the model"
          lines={[
            'chunk loss (NLL or MSE)',
            'surprise mean and max, ||e||',
            'beta mean, alpha mean',
            '||Delta||_F, per layer and total',
            'Fisher update and drift',
            'canary coherence and poison',
            'canary alignment cos(Delta, g_C)',
            'robust z (median, 1.4826 MAD)',
            'CUSUM on log ||Delta||',
          ]}
        />

        <line x1="262" y1="110" x2="303" y2="110" stroke={EDGE} strokeWidth="1.5" />
        <polygon points="303,105.5 303,114.5 310,110" fill={EDGE} />

        <Box
          x={312}
          y={20}
          w={246}
          h={180}
          title="Policy, checked in order"
          lines={[
            '1. budget, per chunk and session',
            '2. projection against g_C',
            '3. rollback: canary, z, CUSUM',
            '4. scale: intermediate z',
            '5. otherwise commit',
          ]}
        />

        <line x1="558" y1="110" x2="599" y2="110" stroke={EDGE} strokeWidth="1.5" />
        <polygon points="599,105.5 599,114.5 606,110" fill={EDGE} />

        {decisions.map((d, i) => (
          <g key={d.label}>
            <rect x="608" y={20 + i * 36} width="356" height="30" rx="6" fill={OVERLAY} stroke={d.color} strokeWidth="1.5" />
            <text x="620" y={40 + i * 36} fill={d.color} fontSize="12.5" fontWeight="600">
              {d.label}
            </text>
            <text x="690" y={40 + i * 36} fill={INK2} fontSize="11.5">
              {d.note}
            </text>
          </g>
        ))}

        <text x="16" y="238" fill={INK} fontSize="13" fontWeight="600">
          Nothing here reads the text
        </text>
        <text x="16" y="258" fill={INK2} fontSize="11.5">
          There is no pattern match, no keyword list, and no regular expression in the decision path.
        </text>
        <text x="16" y="276" fill={INK2} fontSize="11.5">
          Every input to the policy is a number the model itself produced on this chunk.
        </text>
        <text x="16" y="300" fill={MUTED} fontSize="11.5">
          Text generated inside a rolled-back chunk is not retracted: learning is refused, inference continued.
        </text>
      </svg>
    </div>
  );
}
