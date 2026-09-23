"""Free-CPU public demo adapter; the research model and local API stay unchanged."""
from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
import re

from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from deploy.huggingface.pretrained import MODEL_ID

MAX_BODY = 8192
DEMO_SESSION = 'demo_text'

# Hosted sleep is bounded: cheap methods, the fast-weight target, a few steps, a few probes. The GPU is
# shared with chat, so one run at a time. Everything else uses the server's defaults.
SLEEP_LIMITS = {'methods': ('anchor', 'replay'), 'target': 'w0', 'steps': 10, 'seq_len': 256, 'batch_size': 1,
                'replay_rows': 8, 'heldout_rows': 2, 'probes': 6, 'probe_chars': 300}
SLEEP_KEYS = {'method', 'target', 'steps', 'seq_len', 'batch_size', 'replay_rows', 'heldout_rows', 'anchor_lambda', 'sessions', 'probes'}


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _valid_body(action, body):
    if not isinstance(body, dict):
        return False
    if action in ('reset', 'resume'):
        return not body
    if not (body.get('seed') is None or _integer(body['seed'], 0, 2**32 - 1)):
        return False
    if action == 'chat':
        return (
            set(body) <= {'prompt', 'max_new_tokens', 'temperature', 'top_k', 'seed'}
            and isinstance(body.get('prompt'), str) and len(body['prompt']) <= 1024
            and _integer(body.get('max_new_tokens', 128), 0, 128)
            and _number(body.get('temperature', 0.9), 0.01, 5)
            and _integer(body.get('top_k', 50), 0, 8192)
        )
    return False


def _valid_sleep_body(body, public_sessions):
    L = SLEEP_LIMITS
    if not isinstance(body, dict) or not set(body) <= SLEEP_KEYS:
        return False
    if body.get('method', 'anchor') not in L['methods'] or body.get('target', 'w0') != L['target']:
        return False
    if not (_integer(body.get('steps', 1), 1, L['steps']) and _integer(body.get('seq_len', 256), 32, L['seq_len'])
            and _integer(body.get('batch_size', 1), 1, L['batch_size']) and _integer(body.get('replay_rows', 8), 0, L['replay_rows'])
            and _integer(body.get('heldout_rows', 2), 0, L['heldout_rows']) and _number(body.get('anchor_lambda', 0.5), 0, 1)):
        return False
    sessions = body.get('sessions')
    if sessions is not None and not (isinstance(sessions, list) and sessions and set(sessions) <= set(public_sessions)):
        return False
    probes = body.get('probes')
    if probes is not None:
        if not (isinstance(probes, list) and 1 <= len(probes) <= L['probes']):
            return False
        for p in probes:
            if not (isinstance(p, dict) and set(p) <= {'question', 'answer', 'paraphrase'}
                    and isinstance(p.get('question'), str) and 0 < len(p['question']) <= L['probe_chars']
                    and isinstance(p.get('answer'), str) and 0 < len(p['answer']) <= 100
                    and (p.get('paraphrase') is None or (isinstance(p['paraphrase'], str) and len(p['paraphrase']) <= L['probe_chars']))):
                return False
    return True


# what the shared demo lets a visitor do; the UI renders exactly this set. Sleep is on only when the public
# model has fast weights to consolidate (a TTT record), which is decided from the store at request time.
PUBLIC_CAPABILITIES = {'create_session': False, 'fork': False, 'reset': True, 'delete': False, 'resume': True, 'calibrate': False, 'sleep': False}


class PublicDemoGate:
    """Expose the fixed text playground without changing the local research API.

    The public catalog is the pinned model plus every model sleep derived from it, and one demo session per
    model (``demo_text`` for the root, ``demo_<child>`` for a child, created when the child appears)."""
    def __init__(self, app, store=None, device='cpu', model_id=MODEL_ID):
        self.app = app
        self.store = store
        self.device = device
        self.model_id = model_id
        self.busy = asyncio.Lock()

    # ------------------------------------------------------------------ the public catalog
    def public_models(self):
        if self.store is None:
            return [self.model_id]
        by_parent = {}
        for rec in self.store.list_models():
            by_parent.setdefault(rec.get('parent_model_id'), []).append(rec['model_id'])
        out, queue = [], [self.model_id]
        while queue:
            mid = queue.pop(0)
            if mid in out:
                continue
            out.append(mid)
            queue.extend(sorted(by_parent.get(mid, [])))
        return out

    def public_sessions(self, models=None):
        models = models or self.public_models()
        if self.store is None:
            return [DEMO_SESSION]
        wanted = {DEMO_SESSION} | {f'demo_{m}' for m in models[1:]}
        return [rec['session_id'] for rec in self.store.list_sessions() if rec['session_id'] in wanted and rec.get('model_id') in models]

    def sleep_enabled(self):
        if self.store is None:
            return False
        try:
            return self.store.load_model_record(self.model_id).get('backend') == 'ttt'
        except (FileNotFoundError, KeyError):
            return False

    def capabilities(self):
        return {**PUBLIC_CAPABILITIES, 'sleep': self.sleep_enabled()}

    def ensure_child_sessions(self):
        """A demo session for every accepted sleep child, so a visitor can ask the child what the parent was taught."""
        if self.store is None:
            return
        from deploy.huggingface.pretrained import NATIVE_HARNESS
        for mid in self.public_models()[1:]:
            sid = f'demo_{mid}'
            if not self.store.session_exists(sid):
                self.store.create_session(sid, model_id=mid, domain='text', harness_cfg=NATIVE_HARNESS)

    def sleep_running(self):
        jobs = getattr(getattr(self.app, 'state', None), 'sleep_jobs', None)
        return jobs is not None and any(j.get('status') == 'running' for j in jobs.list())

    # ------------------------------------------------------------------ the gate
    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or not scope['path'].startswith('/api'):
            return await self.app(scope, receive, send)
        path, method = scope['path'], scope['method']

        async def reject(status, message):
            await JSONResponse({'detail': message}, status_code=status)(scope, receive, send)

        models = self.public_models()
        sleep_on = self.sleep_enabled()
        if sleep_on and path in ('/api/models', '/api/sessions') or path.startswith('/api/sleep'):
            self.ensure_child_sessions()
        sessions = self.public_sessions(models)
        reads = ('/api/health', '/api/models', '/api/sessions')
        session_read = re.fullmatch(r'/api/sessions/([^/]+)(?:/(state|transactions))?', path)
        session_read = session_read and session_read.group(1) in sessions
        model_read = re.fullmatch(r'/api/models/([^/]+)', path)
        model_read = model_read and model_read.group(1) in models
        sleep_read = sleep_on and re.fullmatch(r'/api/sleep(?:/[^/]+)?', path)
        if method in ('GET', 'HEAD'):
            if path not in reads and not session_read and not model_read and not sleep_read:
                return await reject(404, 'Not available in this demo.')
            if self.store is not None and path in reads:
                from plastic.api.service import model_summary, sanitize
                by_id = {rec['model_id']: rec for rec in self.store.list_models()}
                recs = [by_id[m] for m in models if m in by_id]  # lineage order: the root first, then its children
                sess = [rec for rec in self.store.list_sessions() if rec['session_id'] in sessions]
                if path == '/api/models':
                    result = [model_summary(self.store, rec) for rec in recs]
                elif path == '/api/sessions':
                    result = sess
                else:
                    result = {'ok': True, 'artifacts_root': self.store.root, 'device': self.device,
                              'n_models': len(recs), 'n_sessions': len(sess),
                              'capabilities': self.capabilities(), 'public': True}
                return await JSONResponse(sanitize(result))(scope, receive, send)
            return await self.app(scope, receive, send)
        action_match = re.fullmatch(r'/api/sessions/([^/]+)/(chat|reset|resume)', path)
        sleep_match = sleep_on and re.fullmatch(rf'/api/models/({re.escape(self.model_id)})/sleep', path)
        if method != 'POST' or not (action_match and action_match.group(1) in sessions or sleep_match):
            return await reject(403, 'This action is unavailable in the shared demo.')
        sid, action = action_match.groups() if action_match else (None, 'sleep')
        raw = bytearray()
        while True:
            msg = await receive()
            if msg['type'] == 'http.disconnect':
                return
            raw.extend(msg.get('body', b''))
            if len(raw) > MAX_BODY:
                return await reject(413, 'The public demo accepts request bodies up to 8 KiB.')
            if not msg.get('more_body', False):
                break
        try:
            body = json.loads(raw or b'{}')
            valid = _valid_sleep_body(body, sessions) if action == 'sleep' else _valid_body(action, body)
        except (ValueError, TypeError, OverflowError):
            valid = False
        if not valid:
            if action == 'sleep':
                return await reject(422, 'Hosted sleep accepts anchor or replay on W0, up to 10 steps, 256 tokens, batch 1, 8 replay rows, 2 held-out rows and 6 probes, over the public sessions.')
            return await reject(422, 'Use a prompt up to 1024 characters, 0–128 output tokens, and valid numeric settings.')
        if action == 'sleep' and self.sleep_running():
            return await reject(409, 'A sleep run is already in progress. Wait for its card to finish.')
        if self.busy.locked():
            return await reject(503, 'Another visitor is using the shared demo. Retry shortly.')
        async with self.busy:
            if self.store is not None and action == 'chat':
                meta = self.store.load_session_meta(sid)
                if int(meta.get('pos', 0)) >= 4096:
                    return await reject(409, 'This shared demo session reached its context limit. Reset it from Sessions before continuing.')
            sent = False

            async def replay():
                nonlocal sent
                if not sent:
                    sent = True
                    return {'type': 'http.request', 'body': bytes(raw), 'more_body': False}
                return await receive()

            return await self.app(scope, replay, send)


NOTICE = '''<aside class="bg-surface-overlay text-ink-primary border-b border-edge text-sm px-5 py-3">
<strong>Huihui Qwen3.5-0.8B · abliterated · observational mode</strong>
<span class="ml-2">No automatic rollback.</span><br>
Sessions are <strong>public and shared</strong>. Do not enter private information.
<a class="ml-2 text-accent" href="https://github.com/DMontgomery40/plastic" target="_blank" rel="noreferrer">Project documentation</a>
</aside>'''


def pick_device() -> str:
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def create_demo(artifacts_root: str, dashboard_dist: str, device: str = 'cpu'):
    from plastic.api.app import create_app
    # the shared demo lets visitors chat and reset only; the UI renders exactly these capabilities
    app = create_app(artifacts_root, device=device, public=True, capabilities=PUBLIC_CAPABILITIES)
    dist = Path(dashboard_dist)
    page, count = re.subn(r'(<body\b[^>]*>)', lambda match: match[0][:-1] + ' data-public-demo="true">' + NOTICE,
                         (dist / 'index.html').read_text(), count=1, flags=re.IGNORECASE)
    if count != 1:
        raise ValueError('Dashboard HTML has no body element for the public-session notice')
    app.mount('/assets', StaticFiles(directory=dist / 'assets'), name='assets')

    @app.get('/', response_class=HTMLResponse)
    def index():
        return page

    return PublicDemoGate(app, app.state.store, device)


def main():
    import torch
    import uvicorn
    from deploy.huggingface.prepare import prepare_store
    from deploy.huggingface.pretrained import prepare_pretrained_sessions
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 2)))
    device = pick_device()
    print(f'[demo] device {device}, torch {torch.__version__}, threads {torch.get_num_threads()}', flush=True)
    root = os.environ.get('ARTIFACTS_ROOT', '/tmp/plastic-demo')
    store = prepare_store(Path('.'), Path(root), seed_sessions=False)
    prepare_pretrained_sessions(store, Path(os.environ.get('QWEN_CHECKPOINT', '/opt/qwen')))
    app = create_demo(root, os.environ.get('DASHBOARD_DIST', 'dashboard/dist'), device)
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', '7860')), access_log=False)


if __name__ == '__main__':
    main()
