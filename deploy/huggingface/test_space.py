"""Public hosting boundaries: exercise route, payload, and concurrency families."""
import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from deploy.huggingface.app import PublicDemoGate
from deploy.huggingface.pretrained import MODEL_ID


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


@pytest.mark.parametrize('path', ['/api/health', '/api/models', f'/api/models/{MODEL_ID}', '/api/sessions', '/api/sessions/demo_text/state'])
def test_existing_dashboard_reads_work(path):
    with TestClient(fake_app()) as client:
        assert client.get(path).status_code == 200


@pytest.mark.parametrize('path,body', [
    ('/api/sessions/demo_text/chat', {'prompt': 'Hello', 'max_new_tokens': 128}),
    ('/api/sessions/demo_text/reset', {}),
    ('/api/sessions/demo_text/resume', {}),
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


@pytest.mark.parametrize('path', ['/api/sessions/demo_physics/physics', '/api/sessions/demo_physics/reset', '/api/sessions/demo_text/physics'])
def test_public_physics_actions_are_closed(path):
    with TestClient(fake_app()) as client:
        assert client.post(path, json={}).status_code == 403


@pytest.mark.parametrize('path', ['/api/sessions/demo_physics', '/api/sessions/demo_physics/state', '/api/models/phys_mps_3k', '/api/models/lm_wikitext_l4', f'/api/train/{MODEL_ID}', f'/api/train/{MODEL_ID}/log'])
def test_internal_research_reads_are_not_public(path):
    with TestClient(fake_app()) as client:
        assert client.get(path).status_code == 404


def test_unknown_sessions_methods_and_large_bodies_fail_closed():
    with TestClient(fake_app()) as client:
        assert client.get('/api/sessions/unknown').status_code == 404
        assert client.delete('/api/sessions/demo_text').status_code == 403
        assert client.post('/api/sessions/demo_text/chat', content='x' * 8193).status_code == 413
        assert client.post('/api/sessions/demo_text/chat', content='{').status_code == 422
        assert client.get('/api/models/unknown').status_code == 404
        assert client.get('/api/models/qwen3_5_0_8b').status_code == 404
        assert client.get(f'/api/models/{MODEL_ID}_unknown/log').status_code == 404
        assert client.get(f'/api/models/{MODEL_ID}/private').status_code == 404
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
    for sub, mid, domain in [('text', 'lm_wikitext_l4', 'text')]:
        cfg = ModelConfig(domain=domain, d_model=8, n_heads=1, n_layers=1, chunk=4, vocab_size=32)
        original.save_checkpoint(mid, cfg, build_model(cfg), step=1)
        import shutil
        shutil.copytree(original.model_dir(mid), source / sub)
        folder = source / sub
        (folder / 'eval.json').write_text('{}')
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
        (folder / 'manifest.json').write_text(json.dumps({'files':hashes}))
    store = prepare_store(source, target)
    assert len(store.list_models()) == 1
    store.register_model('lm_wikitext_l4', {'custom': 'preserve me'})
    prepare_store(source, target)
    assert store.load_model_record('lm_wikitext_l4')['custom'] == 'preserve me'
    (Path(store.model_dir('lm_wikitext_l4')) / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='overwrite'):
        prepare_store(source, target)
    (source / 'text' / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        prepare_store(source, tmp_path / 'fresh')


@pytest.mark.parametrize('body_tag', ['<body>', '<body class="bg-surface text-ink-primary">', '<BODY class="test">'])
def test_public_notice_survives_real_dashboard_body_attributes(tmp_path, body_tag):
    from deploy.huggingface.app import create_demo
    (tmp_path / 'dist' / 'assets').mkdir(parents=True)
    (tmp_path / 'dist' / 'index.html').write_text(f'<html>{body_tag}<div id="root"></div></body></html>')
    with TestClient(create_demo(str(tmp_path / 'store'), str(tmp_path / 'dist'))) as client:
        page = client.get('/')
        assert 'data-public-demo="true"' in page.text
        assert 'public and shared' in page.text
        assert 'Do not enter private information' in page.text
        assert 'Qwen3.5-0.8B' in page.text
        assert 'abliterated · observational mode' in page.text
        assert 'No automatic rollback.' in page.text
        assert 'Project documentation' in page.text
        assert page.text.index('public and shared') < page.text.index('id="root"')


def test_pretrained_demo_registers_native_sessions_and_preserves_existing_state(tmp_path, monkeypatch):
    from plastic.backends import qwen
    from plastic.config import ModelConfig
    from plastic.model.lm import build_model
    from plastic.store import ArtifactStore
    from deploy.huggingface.pretrained import CHECKPOINT_DIGEST, MODEL_ID, prepare_pretrained_sessions
    monkeypatch.setattr(qwen, '_checkpoint_digest', lambda _: CHECKPOINT_DIGEST)
    store = ArtifactStore(str(tmp_path / 'store'))
    cfg = ModelConfig(domain='physics', d_model=8, n_heads=1, n_layers=1, chunk=4)
    store.save_checkpoint('phys_mps_3k', cfg, build_model(cfg), step=1)
    store.register_model('phys_mps_3k', {'domain': 'physics'})
    prepare_pretrained_sessions(store, tmp_path / 'checkpoint')
    meta = store.load_session_meta('demo_text')
    assert meta['model_id'] == MODEL_ID
    assert meta['harness']['log_only'] is True
    assert meta['harness']['freeze_on_alarm'] is False
    assert meta['harness']['learn_from_generation'] is True
    assert store.load_model_record(MODEL_ID)['backend'] == 'qwen'
    prepare_pretrained_sessions(store, tmp_path / 'checkpoint')
    assert store.load_session_meta('demo_text') == meta
    assert len(store.list_sessions()) == 1


def test_public_catalog_filters_without_removing_local_artifacts(tmp_path):
    from plastic.api.app import create_app
    from plastic.harness.config import HarnessConfig
    from plastic.store import ArtifactStore
    store = ArtifactStore(str(tmp_path))
    for mid in [MODEL_ID, 'private_model', 'phys_mps_3k']:
        store.register_model(mid, {'backend': 'qwen', 'domain': 'text', 'params': 10})
    for sid, mid in [('demo_text', MODEL_ID), ('demo_physics', 'phys_mps_3k'), ('private_session', 'private_model')]:
        store.create_session(sid, model_id=mid, domain='text', harness_cfg=HarnessConfig())
    app = create_app(str(tmp_path), device='cpu')
    with TestClient(PublicDemoGate(app, store)) as client:
        assert [m['model_id'] for m in client.get('/api/models').json()] == [MODEL_ID]
        assert [s['session_id'] for s in client.get('/api/sessions').json()] == ['demo_text']
        health = client.get('/api/health').json()
        assert health['n_models'] == 1 and health['n_sessions'] == 1
        assert health['public'] is True and health['capabilities']['create_session'] is False and health['capabilities']['reset'] is True
        for path in ['/api/train/jobs', '/api/data', '/api/redteam']:
            assert client.get(path).status_code == 404
    with TestClient(app) as client:
        assert len(client.get('/api/models').json()) == 3
        assert len(client.get('/api/sessions').json()) == 3


@pytest.mark.parametrize('mismatch', ['checkpoint', 'model', 'controls'])
def test_pretrained_demo_refuses_incompatible_artifacts_or_sessions(tmp_path, monkeypatch, mismatch):
    from plastic.backends import qwen
    from plastic.harness.config import HarnessConfig
    from plastic.store import ArtifactStore
    from deploy.huggingface.pretrained import CHECKPOINT_DIGEST, MODEL_ID, prepare_pretrained_sessions
    monkeypatch.setattr(qwen, '_checkpoint_digest', lambda _: 'changed' if mismatch == 'checkpoint' else CHECKPOINT_DIGEST)
    store = ArtifactStore(str(tmp_path / 'store'))
    if mismatch != 'checkpoint':
        mid = 'old_model' if mismatch == 'model' else MODEL_ID
        store.register_model(mid, {'backend': 'qwen', 'checkpoint_digest': CHECKPOINT_DIGEST})
        store.create_session('demo_text', model_id=mid, domain='text', harness_cfg=HarnessConfig())
    before = store.load_session_meta('demo_text') if store.session_exists('demo_text') else None
    with pytest.raises(ValueError):
        prepare_pretrained_sessions(store, tmp_path / 'checkpoint')
    if before is not None:
        assert store.load_session_meta('demo_text') == before
    else:
        assert not store.list_sessions()
