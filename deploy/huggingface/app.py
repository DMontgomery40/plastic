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

MODELS = (MODEL_ID,)
SESSIONS = ('demo_text',)
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
    return False


# what the shared demo lets a visitor do; the UI renders exactly this set
PUBLIC_CAPABILITIES = {'create_session': False, 'fork': False, 'reset': True, 'delete': False, 'resume': True, 'calibrate': False, 'sleep': False}


class PublicDemoGate:
    """Expose the fixed text playground without changing the local research API."""
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

        reads = ('/api/health', '/api/models', '/api/sessions')
        session_read = re.fullmatch(r'/api/sessions/([^/]+)(?:/(state|transactions))?', path)
        session_read = session_read and session_read.group(1) in SESSIONS
        model_read = re.fullmatch(r'/api/models/([^/]+)', path)
        model_read = model_read and model_read.group(1) in MODELS
        if method in ('GET', 'HEAD'):
            if path not in reads and not session_read and not model_read:
                return await reject(404, 'Not available in this demo.')
            if self.store is not None and path in ('/api/health', '/api/models', '/api/sessions'):
                from plastic.api.service import model_summary, sanitize
                models = [rec for rec in self.store.list_models() if rec['model_id'] in MODELS]
                sessions = [rec for rec in self.store.list_sessions() if rec['session_id'] in SESSIONS]
                if path == '/api/models':
                    result = [model_summary(self.store, rec) for rec in models]
                elif path == '/api/sessions':
                    result = sessions
                else:
                    result = {'ok': True, 'artifacts_root': self.store.root, 'device': 'cpu',
                              'n_models': len(models), 'n_sessions': len(sessions),
                              'capabilities': dict(PUBLIC_CAPABILITIES), 'public': True}
                return await JSONResponse(sanitize(result))(scope, receive, send)
            return await self.app(scope, receive, send)
        action_match = re.fullmatch(r'/api/sessions/(demo_text)/(chat|reset|resume)', path)
        if method != 'POST' or not action_match:
            return await reject(403, 'This action is unavailable in the shared demo.')
        sid, action = action_match.groups()
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
            return await reject(422, 'Use a prompt up to 1024 characters, 0–128 output tokens, and valid numeric settings.')
        if self.busy.locked():
            return await reject(503, 'Another visitor is running the shared CPU demo. Retry shortly.')
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


def create_demo(artifacts_root: str, dashboard_dist: str):
    from plastic.api.app import create_app
    # the shared demo lets visitors chat and reset only; the UI renders exactly these capabilities
    app = create_app(artifacts_root, device='cpu', public=True, capabilities=PUBLIC_CAPABILITIES)
    dist = Path(dashboard_dist)
    page, count = re.subn(r'(<body\b[^>]*>)', lambda match: match[0][:-1] + ' data-public-demo="true">' + NOTICE,
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
