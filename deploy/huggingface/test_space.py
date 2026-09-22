"""Public hosting boundaries: exercise route, payload, and concurrency families."""
import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from deploy.huggingface.app import PublicDemoGate


def fake_app():
    app = FastAPI()

    @app.api_route('/{path:path}', methods=['GET', 'POST', 'DELETE'])
    async def echo(path: str, request: Request):
        return {'path': path, 'body': await request.json() if request.method == 'POST' else None}

    return PublicDemoGate(app)


@pytest.mark.parametrize('path', ['/api/train', '/api/sleep', '/api/redteam', '/api/models/lm_wikitext_l4/calibrate', '/api/sessions', '/api/sessions/demo_text/fork'])
def test_expensive_or_unbounded_mutations_are_closed(path):
    with TestClient(fake_app()) as client:
        assert client.post(path, json={}).status_code == 403


@pytest.mark.parametrize('path', ['/api/health', '/api/models', '/api/models/lm_wikitext_l4', '/api/sessions', '/api/sessions/demo_text/state', '/api/train/jobs', '/api/data'])
def test_existing_dashboard_reads_work(path):
    with TestClient(fake_app()) as client:
        assert client.get(path).status_code == 200


@pytest.mark.parametrize('path,body', [
    ('/api/sessions/demo_text/chat', {'prompt': 'Hello', 'max_new_tokens': 128}),
    ('/api/sessions/demo_physics/physics', {'steps': 256, 'mu': 0.12, 'seed': 0}),
    ('/api/sessions/demo_text/reset', {}),
    ('/api/sessions/demo_physics/resume', {}),
])
def test_bounded_actions_pass_without_silently_changing_parameters(path, body):
    with TestClient(fake_app()) as client:
        response = client.post(path, json=body)
        assert response.status_code == 200
        assert response.json()['body'] == body


@pytest.mark.parametrize('body', [[], None, {'prompt': 'x' * 1025}, {'prompt':'x', 'max_new_tokens':129}, {'prompt':'x', 'max_new_tokens':True}, {'prompt':'x', 'temperature':float('inf')}, {'prompt':'x', 'seed':2**80}])
def test_bad_chat_body_family(body):
    with TestClient(fake_app()) as client:
        assert client.post('/api/sessions/demo_text/chat', content=json.dumps(body)).status_code == 422


@pytest.mark.parametrize('body', [{'steps':257}, {'steps':0}, {'mu':float('nan')}, {'mu':100}, {'seed':-1}, {'nonlinear':'yes'}])
def test_bad_physics_body_family(body):
    with TestClient(fake_app()) as client:
        assert client.post('/api/sessions/demo_physics/physics', content=json.dumps(body)).status_code == 422


def test_unknown_sessions_methods_and_large_bodies_fail_closed():
    with TestClient(fake_app()) as client:
        assert client.get('/api/sessions/unknown').status_code == 404
        assert client.delete('/api/sessions/demo_text').status_code == 403
        assert client.post('/api/sessions/demo_text/chat', content='x' * 8193).status_code == 413
        assert client.post('/api/sessions/demo_text/chat', content='{').status_code == 422
        assert client.get('/api/models/unknown').status_code == 404
        assert client.get('/assets/main.js').status_code == 200


def test_busy_response_and_failure_recovery():
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        inner = FastAPI()
        calls = 0

        @inner.post('/api/sessions/demo_text/reset')
        async def action():
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                await release.wait()
                raise RuntimeError('simulated failure')
            return {'ok': True}

        gate = PublicDemoGate(inner)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gate, raise_app_exceptions=False), base_url='http://test') as client:
            first = asyncio.create_task(client.post('/api/sessions/demo_text/reset', json={}))
            await started.wait()
            assert (await client.post('/api/sessions/demo_text/reset', json={})).status_code == 503
            release.set()
            assert (await first).status_code == 500
            assert (await client.post('/api/sessions/demo_text/reset', json={})).status_code == 200
    asyncio.run(scenario())


def test_prepare_checks_integrity_and_preserves_existing_state(tmp_path):
    from pathlib import Path
    import hashlib
    import torch
    from plastic.config import ModelConfig
    from plastic.model.lm import build_model
    from plastic.store import ArtifactStore
    from deploy.huggingface.prepare import prepare_store

    source, target = tmp_path / 'source', tmp_path / 'target'
    original = ArtifactStore(str(tmp_path / 'original'))
    for sub, mid, domain in [('text', 'lm_wikitext_l4', 'text'), ('physics', 'phys_mps_3k', 'physics')]:
        cfg = ModelConfig(domain=domain, d_model=8, n_heads=1, n_layers=1, chunk=4, vocab_size=32)
        original.save_checkpoint(mid, cfg, build_model(cfg), step=1)
        import shutil
        shutil.copytree(original.model_dir(mid), source / sub)
        folder = source / sub
        (folder / 'eval.json').write_text('{}')
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
        (folder / 'manifest.json').write_text(json.dumps({'files':hashes}))
    store = prepare_store(source, target)
    assert len(store.list_models()) == 2
    store.register_model('lm_wikitext_l4', {'custom': 'preserve me'})
    prepare_store(source, target)
    assert store.load_model_record('lm_wikitext_l4')['custom'] == 'preserve me'
    (Path(store.model_dir('phys_mps_3k')) / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='overwrite'):
        prepare_store(source, target)
    (source / 'text' / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        prepare_store(source, tmp_path / 'fresh')


@pytest.mark.parametrize('body_tag', ['<body>', '<body class="bg-surface text-ink-primary">'])
def test_public_notice_survives_real_dashboard_body_attributes(tmp_path, body_tag):
    from deploy.huggingface.app import create_demo
    (tmp_path / 'dist' / 'assets').mkdir(parents=True)
    (tmp_path / 'dist' / 'index.html').write_text(f'<html>{body_tag}<div id="root"></div></body></html>')
    with TestClient(create_demo(str(tmp_path / 'store'), str(tmp_path / 'dist'))) as client:
        page = client.get('/')
        assert 'public and shared' in page.text
        assert 'Do not enter private information' in page.text
        assert page.text.index('public and shared') < page.text.index('id="root"')
