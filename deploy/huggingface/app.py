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

MODELS = (MODEL_ID, 'lm_wikitext_l4', 'phys_mps_3k')
SESSIONS = ('demo_text', 'demo_physics')
MAX_BODY = 8192


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
    return (
        set(body) <= {'steps', 'mu', 'seed', 'nonlinear'}
        and _integer(body.get('steps', 256), 1, 256)
        and _number(body.get('mu', 0.12), 0, 1)
        and _integer(body.get('seed', 0), 0, 2**32 - 1)
        and type(body.get('nonlinear', False)) is bool
    )


class PublicDemoGate:
    """Allow only bounded operations on two intentionally public, fixed sessions."""
    def __init__(self, app, store=None):
        self.app = app
        self.store = store
        self.busy = asyncio.Lock()

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or not scope['path'].startswith('/api'):
            return await self.app(scope, receive, send)
        path, method = scope['path'], scope['method']

        async def reject(status, message):
            await JSONResponse({'detail': message}, status_code=status)(scope, receive, send)

        reads = ('/api/health', '/api/models', '/api/sessions', '/api/train/jobs', '/api/data', '/api/redteam')
        session_read = re.fullmatch(r'/api/sessions/(demo_text|demo_physics)(?:/(state|transactions))?', path)
        model_read = re.fullmatch(r'/api/(models|train)/([^/]+)(?:/log)?', path)
        model_read = model_read and model_read.group(2) in MODELS
        if method in ('GET', 'HEAD'):
            if path not in reads and not session_read and not model_read:
                return await reject(404, 'This public demo exposes only its published models and shared sessions.')
            return await self.app(scope, receive, send)
        action_match = re.fullmatch(r'/api/sessions/(demo_text|demo_physics)/(chat|physics|reset|resume)', path)
        if method != 'POST' or not action_match:
            return await reject(403, 'This free-CPU demo uses two shared sessions. Training, calibration, red-team jobs, consolidation, and session creation/deletion/forking are available in the downloadable local project.')
        sid, action = action_match.groups()
        if (action == 'chat' and sid != 'demo_text') or (action == 'physics' and sid != 'demo_physics'):
            return await reject(422, 'Choose the matching text or physics session.')
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
            valid = _valid_body(action, body)
        except (ValueError, TypeError, OverflowError):
            valid = False
        if not valid:
            return await reject(422, 'Demo limits: prompt up to 1024 characters, 0–128 generated tokens, 1–256 physics steps, friction 0–1, and finite numeric settings. Use the local project for larger runs.')
        if self.busy.locked():
            return await reject(503, 'Another visitor is running the shared CPU demo. Retry shortly.')
        async with self.busy:
            if self.store is not None and action in ('chat', 'physics'):
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
<strong>Huihui Qwen3.5-0.8B · abliterated research model · free CPU</strong><br>
Open <strong>Chat</strong> with <strong>demo_text</strong> to explore prompts and inspect context changes.
This is Huihui's refusal-ablated derivative of Qwen, not the original aligned checkpoint.
Text uses native context updates with an <strong>observational guard: no rollback protection</strong>.
The turn-boundary retention policy is experimental and is not enabled here.
The optional <strong>demo_physics</strong> session keeps the Plastic research model.<br>
These two sessions are <strong>public and shared</strong>; prompts and outputs are visible to other visitors. Do not enter private information.
Runs are limited to 128 generated tokens or 256 physics steps. Reset a session to start fresh; restart clears all demo activity.
Training, calibration, red-team jobs, and creating/forking/deleting sessions are local-only features.<br>
<a class="text-accent" href="https://huggingface.co/dmontgomery40/plastic" target="_blank" rel="noreferrer">Project, code &amp; checkpoints</a>
 · <a class="text-accent" href="https://github.com/DMontgomery40/plastic" target="_blank" rel="noreferrer">GitHub</a>
</aside>'''


def create_demo(artifacts_root: str, dashboard_dist: str):
    from plastic.api.app import create_app
    app = create_app(artifacts_root, device='cpu')
    dist = Path(dashboard_dist)
    page, count = re.subn(r'(<body\b[^>]*>)', lambda match: match[0] + NOTICE,
                         (dist / 'index.html').read_text(), count=1, flags=re.IGNORECASE)
    if count != 1:
        raise ValueError('Dashboard HTML has no body element for the public-session notice')
    app.mount('/assets', StaticFiles(directory=dist / 'assets'), name='assets')

    @app.get('/', response_class=HTMLResponse)
    def index():
        return page

    return PublicDemoGate(app, app.state.store)


def main():
    import torch
    import uvicorn
    from deploy.huggingface.prepare import prepare_store
    from deploy.huggingface.pretrained import prepare_pretrained_sessions
    torch.set_num_threads(2)
    root = os.environ.get('ARTIFACTS_ROOT', '/tmp/plastic-demo')
    store = prepare_store(Path('.'), Path(root), seed_sessions=False)
    prepare_pretrained_sessions(store, Path(os.environ.get('QWEN_CHECKPOINT', '/opt/qwen')))
    app = create_demo(root, os.environ.get('DASHBOARD_DIST', 'dashboard/dist'))
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', '7860')), access_log=False)


if __name__ == '__main__':
    main()
