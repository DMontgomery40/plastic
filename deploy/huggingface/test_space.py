"""Public hosting boundaries: exercise route, payload, and concurrency families."""
import asyncio
import json
import time

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


@pytest.mark.parametrize('path', ['/api/sessions/demo_physics', '/api/sessions/demo_physics/state', '/api/models/phys_mps_3k', f'/api/train/{MODEL_ID}', f'/api/train/{MODEL_ID}/log'])
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
    assert store.load_model_record('lm_wikitext_l4')['chat_tuned'] is False
    store.register_model('lm_wikitext_l4', {'custom': 'preserve me', 'chat_tuned': True})
    prepare_store(source, target)
    assert store.load_model_record('lm_wikitext_l4')['custom'] == 'preserve me'
    assert store.load_model_record('lm_wikitext_l4')['chat_tuned'] is False
    (Path(store.model_dir('lm_wikitext_l4')) / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='overwrite'):
        prepare_store(source, target)
    (source / 'text' / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        prepare_store(source, tmp_path / 'fresh')


@pytest.mark.parametrize('body_tag', ['<body>', '<body class="bg-surface text-ink-primary">', '<BODY class="test">'])
def test_public_notice_is_model_neutral_and_survives_dashboard_body_attributes(tmp_path, body_tag):
    from deploy.huggingface.app import create_demo
    (tmp_path / 'dist' / 'assets').mkdir(parents=True)
    (tmp_path / 'dist' / 'index.html').write_text(f'<html>{body_tag}<div id="root"></div></body></html>')
    with TestClient(create_demo(str(tmp_path / 'store'), str(tmp_path / 'dist'))) as client:
        page = client.get('/')
        assert 'data-public-demo="true"' in page.text
        assert 'public and shared' in page.text
        assert 'Qwen3.5-0.8B' not in page.text
        assert 'observational mode' not in page.text
        assert 'No automatic rollback.' not in page.text
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


def _ttt_public_store(tmp_path):
    """A store whose public model is a TTT record with one sleep child already present."""
    from plastic.harness.config import HarnessConfig
    from plastic.store import ArtifactStore
    store = ArtifactStore(str(tmp_path / 'store'))
    ckpt = tmp_path / 'ckpt'
    ckpt.mkdir(exist_ok=True)
    store.register_model('ttt_pub', {'backend': 'ttt', 'domain': 'text', 'status': 'completed', 'params': 10, 'checkpoint_dir': str(ckpt)})
    store.register_model('sleep_a', {'backend': 'ttt', 'domain': 'text', 'status': 'completed', 'params': 10, 'checkpoint_dir': str(ckpt),
                                     'parent_model_id': 'ttt_pub', 'type': 'sleep'})
    store.register_model('unrelated', {'backend': 'ttt', 'domain': 'text', 'status': 'completed', 'params': 10, 'checkpoint_dir': str(ckpt)})
    store.create_session('demo_text', model_id='ttt_pub', domain='text', harness_cfg=HarnessConfig())
    store.create_session('private', model_id='ttt_pub', domain='text', harness_cfg=HarnessConfig())
    return store


def test_public_sleep_is_closed_while_the_public_model_has_no_fast_weights():
    with TestClient(fake_app()) as client:
        assert client.get('/api/sleep').status_code == 404
        assert client.post(f'/api/models/{MODEL_ID}/sleep', json={}).status_code == 403


def test_public_catalog_follows_sleep_lineage_and_opens_bounded_sleep_for_a_ttt_model(tmp_path, monkeypatch):
    import sys
    from plastic.api import sleep_jobs
    from plastic.api.app import create_app
    store = _ttt_public_store(tmp_path)
    app = create_app(str(tmp_path / 'store'), device='cpu')

    def fake_argv(model_id, run_dir, artifacts_root, device, options, sessions, probes_path):
        script = ("import json,sys,os; d=sys.argv[1]; open(os.path.join(d,'log.txt'),'a').write('[sleep] stand-in\\n');"
                  f"json.dump({{'run_id': os.path.basename(d), 'parent_model_id': {model_id!r}, 'status': 'accepted', 'model_id': 'sleep_b',"
                  " 'gate': {'passed': True, 'measured': True, 'checks': []}}, open(os.path.join(d,'sleep_report.json'),'w'));"
                  "import shutil")
        return [sys.executable, '-c', script, run_dir]

    monkeypatch.setattr(sleep_jobs, 'build_sleep_argv', fake_argv)
    with TestClient(PublicDemoGate(app, store, 'cpu', model_id='ttt_pub')) as client:
        health = client.get('/api/health').json()
        assert health['capabilities']['sleep'] is True and health['n_models'] == 2
        # the catalog: the public model and its descendant, never the unrelated record or the private session
        assert [m['model_id'] for m in client.get('/api/models').json()] == ['ttt_pub', 'sleep_a']
        assert client.get('/api/models/unrelated').status_code == 404
        assert client.get('/api/models/sleep_a').status_code == 200
        # the child got a demo session on first sight; the private session stays private
        assert [s['session_id'] for s in client.get('/api/sessions').json()] == ['demo_sleep_a', 'demo_text'] or \
            sorted(s['session_id'] for s in client.get('/api/sessions').json()) == ['demo_sleep_a', 'demo_text']
        assert client.get('/api/sessions/private').status_code == 404
        assert store.load_session_meta('demo_sleep_a')['harness']['log_only'] is True  # native observational controls
        assert client.get('/api/sleep').status_code == 200
        # bounded body family
        for bad in ({'method': 'distill'}, {'target': 'all'}, {'steps': 11}, {'seq_len': 512}, {'batch_size': 2}, {'lr': 1e-3},
                    {'sessions': ['private']}, {'sessions': []}, {'probes': [{'question': 'q', 'answer': 'x' * 101}]},
                    {'probes': [{'question': 'q', 'answer': 'a'}] * 7}):
            assert client.post('/api/models/ttt_pub/sleep', json=bad).status_code == 422, bad
        assert client.post('/api/models/sleep_a/sleep', json={}).status_code == 403  # only the root model sleeps publicly
        r = client.post('/api/models/ttt_pub/sleep', json={'method': 'anchor', 'sessions': ['demo_text'],
                                                          'probes': [{'question': 'q', 'answer': 'a', 'paraphrase': 'q2'}]})
        assert r.status_code == 200, r.text
        run_id = r.json()['run_id']
        for _ in range(100):
            s = client.get(f'/api/sleep/{run_id}').json()
            if s['status'] != 'running':
                break
            time.sleep(0.05)
        assert s['status'] == 'accepted' and s['report']['model_id'] == 'sleep_b'
        # the new child needs a record for the catalog to pick it up (the real run registers it; the stand-in did not)
        store.register_model('sleep_b', {'backend': 'ttt', 'domain': 'text', 'status': 'completed', 'params': 10,
                                         'checkpoint_dir': str(tmp_path / 'ckpt'), 'parent_model_id': 'ttt_pub', 'type': 'sleep'})
        assert [m['model_id'] for m in client.get('/api/models').json()] == ['ttt_pub', 'sleep_a', 'sleep_b']
        assert 'demo_sleep_b' in [x['session_id'] for x in client.get('/api/sessions').json()]
        assert store.session_exists('demo_sleep_b') and store.load_session_meta('demo_sleep_b')['model_id'] == 'sleep_b'


def test_public_sleep_forwards_materialized_public_defaults_not_the_visitor_body(tmp_path, monkeypatch):
    """ASTRA-162: the gate must hand the server the bounded body it validated, never the visitor's body plus the
    server's richer local defaults. The options SleepJobs.start actually receives are what is asserted."""
    from plastic.api import sleep_jobs
    from plastic.api.app import create_app
    store = _ttt_public_store(tmp_path)
    app = create_app(str(tmp_path / 'store'), device='cpu')
    seen = []

    def fake_start(self, model_id, options, *, sessions, probes):
        seen.append({'model_id': model_id, 'options': options, 'sessions': sessions, 'probes': probes})
        return {'run_id': f'run{len(seen)}', 'model_id': model_id, 'status': 'running', 'exit_code': None, 'pid': 1, 'started_at_unix': 0,
                'options': options, 'sessions': sessions, 'n_probes': len(probes or []), 'report': None, 'log_tail': [], 'stderr_tail': []}

    monkeypatch.setattr(sleep_jobs.SleepJobs, 'start', fake_start)
    monkeypatch.setattr(sleep_jobs.SleepJobs, 'list', lambda self: [])
    with TestClient(PublicDemoGate(app, store, 'cpu', model_id='ttt_pub')) as client:
        assert client.post('/api/models/ttt_pub/sleep', json={}).status_code == 200
        o = seen[-1]['options']
        assert (o['method'], o['target'], o['steps'], o['seq_len'], o['batch_size'], o['replay_rows'], o['heldout_rows']) == ('anchor', 'w0', 5, 256, 1, 8, 2)
        assert o['lr'] == 1e-4 and o['tolerance_nll'] == 0.05  # the request schema's defaults, not a richer visitor body
        assert seen[-1]['sessions'] == ['demo_text'] and seen[-1]['probes'] is None  # the root's sessions only, never a child's
        assert client.post('/api/models/ttt_pub/sleep', json={'sessions': ['demo_sleep_a']}).status_code == 422
        assert client.post('/api/models/ttt_pub/sleep', json={'method': 'replay', 'steps': 10, 'sessions': ['demo_text'],
                                                              'probes': [{'question': 'q', 'answer': 'a'}]}).status_code == 200
        o = seen[-1]['options']
        assert (o['method'], o['steps'], seen[-1]['sessions'], seen[-1]['probes']) == ('replay', 10, ['demo_text'], [{'question': 'q', 'answer': 'a', 'paraphrase': None}])
        # boundary and over-limit values
        assert client.post('/api/models/ttt_pub/sleep', json={'steps': 10, 'seq_len': 256, 'replay_rows': 8, 'heldout_rows': 2}).status_code == 200
        for bad in ({'steps': 11}, {'seq_len': 257}, {'replay_rows': 9}, {'heldout_rows': 3}, {'anchor_lambda': 1.5}, {'lr': 1e-4}, {'tolerance_nll': 1.0}):
            assert client.post('/api/models/ttt_pub/sleep', json=bad).status_code == 422, bad
        assert len(seen) == 3


def test_public_sleep_refuses_a_second_concurrent_run(tmp_path, monkeypatch):
    import sys
    from plastic.api import sleep_jobs
    from plastic.api.app import create_app
    store = _ttt_public_store(tmp_path)
    app = create_app(str(tmp_path / 'store'), device='cpu')
    monkeypatch.setattr(sleep_jobs, 'build_sleep_argv', lambda *a, **k: [sys.executable, '-c', 'import time; time.sleep(2)'])
    with TestClient(PublicDemoGate(app, store, 'cpu', model_id='ttt_pub')) as client:
        assert client.post('/api/models/ttt_pub/sleep', json={'method': 'anchor'}).status_code == 200
        assert client.post('/api/models/ttt_pub/sleep', json={'method': 'anchor'}).status_code == 409


def test_public_model_specs_register_the_right_backend_and_refuse_an_unpinned_release(tmp_path, monkeypatch):
    from dataclasses import replace
    from plastic.store import ArtifactStore
    from deploy.huggingface import pretrained
    monkeypatch.setenv('PUBLIC_MODEL', 'ttt')
    with pytest.raises(ValueError, match='not pinned'):
        pretrained.active_spec()
    monkeypatch.setenv('PUBLIC_MODEL', 'qwen')
    assert pretrained.active_spec().kind == 'qwen'
    pinned = replace(pretrained.TTT_CHAT, revision='abc', checkpoint_digest='d1')
    monkeypatch.setattr(pretrained.PublicModelSpec, 'digest_of', lambda self, folder: 'd1')
    store = ArtifactStore(str(tmp_path / 'store'))
    (tmp_path / 'ckpt' / 'text-chat').mkdir(parents=True)
    pretrained.prepare_pretrained_sessions(store, tmp_path / 'ckpt', spec=pinned)
    rec = store.load_model_record('ttt_mlp_760m_chat_v1')
    assert (rec['backend'], rec['chunk'], rec['chat_tuned'], rec['checkpoint_digest']) == ('ttt', 16, True, 'd1')
    assert rec['checkpoint_dir'].endswith('text-chat')
    assert store.load_session_meta('demo_text')['model_id'] == 'ttt_mlp_760m_chat_v1'
    # with the TTT record public, the gate turns sleep on by itself
    from plastic.api.app import create_app
    app = create_app(str(tmp_path / 'store'), device='cpu')
    with TestClient(PublicDemoGate(app, store, 'cpu', model_id='ttt_mlp_760m_chat_v1')) as client:
        assert client.get('/api/health').json()['capabilities']['sleep'] is True
    monkeypatch.setattr(pretrained.PublicModelSpec, 'digest_of', lambda self, folder: 'other')
    with pytest.raises(ValueError, match='does not match'):
        pretrained.prepare_pretrained_sessions(ArtifactStore(str(tmp_path / 'b')), tmp_path / 'ckpt', spec=pinned)


def test_hosted_research_model_is_listed_with_one_guarded_session_and_stays_out_of_sleep_lineage(tmp_path):
    """The published PlasticCore text model is in the image; the catalog lists it beside the pinned model, with one
    guarded demo session that does not latch read-only. Physics and private records stay hidden; the research model is
    not a sleep child and gets no demo_<model> session."""
    from deploy.huggingface.app import ensure_research_sessions
    from plastic.api.app import create_app
    from plastic.harness.config import HarnessConfig
    from plastic.store import ArtifactStore
    store = ArtifactStore(str(tmp_path))
    store.register_model(MODEL_ID, {'backend': 'qwen', 'domain': 'text', 'params': 10})
    # stand-in records (as in the catalog test above): the gate reads records and sessions, not weights
    store.register_model('lm_wikitext_l4', {'backend': 'qwen', 'domain': 'text', 'params': 6845984})
    store.register_model('phys_mps_3k', {'backend': 'qwen', 'domain': 'physics', 'params': 10})
    store.create_session('demo_text', model_id=MODEL_ID, domain='text', harness_cfg=HarnessConfig())
    store.create_session('demo_physics', model_id='phys_mps_3k', domain='text', harness_cfg=HarnessConfig())
    ensure_research_sessions(store)
    ensure_research_sessions(store)  # idempotent
    harness = store.load_session_meta('demo_core')['harness']
    assert harness['log_only'] is False and harness['freeze_on_alarm'] is False and harness['learn_from_generation'] is True
    app = create_app(str(tmp_path), device='cpu')
    gate = PublicDemoGate(app, store)
    with TestClient(gate) as client:
        assert [m['model_id'] for m in client.get('/api/models').json()] == [MODEL_ID, 'lm_wikitext_l4']
        assert sorted(s['session_id'] for s in client.get('/api/sessions').json()) == ['demo_core', 'demo_text']
        assert client.get('/api/models/lm_wikitext_l4').status_code == 200
        assert client.get('/api/sessions/demo_physics').status_code == 404
        assert client.post('/api/sessions/demo_core/chat', json={'prompt': 'x' * 2000}).status_code == 422  # same bounds
        assert client.post('/api/models/lm_wikitext_l4/sleep', json={}).status_code == 403
    assert gate.lineage_models() == [MODEL_ID]
    assert not store.session_exists('demo_lm_wikitext_l4')


def test_research_model_absent_from_the_store_is_never_listed(tmp_path):
    from deploy.huggingface.app import ensure_research_sessions
    from plastic.api.app import create_app
    from plastic.harness.config import HarnessConfig
    from plastic.store import ArtifactStore
    store = ArtifactStore(str(tmp_path))
    store.register_model(MODEL_ID, {'backend': 'qwen', 'domain': 'text', 'params': 10})
    store.create_session('demo_text', model_id=MODEL_ID, domain='text', harness_cfg=HarnessConfig())
    ensure_research_sessions(store)
    assert not store.session_exists('demo_core')
    with TestClient(PublicDemoGate(create_app(str(tmp_path), device='cpu'), store)) as client:
        assert [m['model_id'] for m in client.get('/api/models').json()] == [MODEL_ID]


@pytest.mark.parametrize('creation_order', [
    ('demo_text', 'demo_sleep_a', 'demo_core'),
    ('demo_core', 'demo_text', 'demo_sleep_a'),
    ('demo_sleep_a', 'demo_core', 'demo_text'),
])
def test_public_session_order_keeps_the_pinned_model_first(tmp_path, creation_order):
    from plastic.api.app import create_app
    from plastic.harness.config import HarnessConfig
    from plastic.store import ArtifactStore
    store = ArtifactStore(str(tmp_path))
    pairs = {'demo_text': MODEL_ID, 'demo_sleep_a': 'sleep_a', 'demo_core': 'lm_wikitext_l4'}
    for mid in pairs.values():
        store.register_model(mid, {'backend': 'qwen', 'domain': 'text', 'params': 10,
                                   'parent_model_id': MODEL_ID if mid == 'sleep_a' else None})
    for created_at, sid in enumerate(creation_order):
        store.create_session(sid, model_id=pairs[sid], domain='text', harness_cfg=HarnessConfig())
        meta = store.load_session_meta(sid)
        meta['created_at_unix'] = created_at
        store._upsert_session_index(sid, store._session_summary(meta))
    gate = PublicDemoGate(create_app(str(tmp_path), device='cpu'), store)
    with TestClient(gate) as client:
        assert gate.public_sessions() == ['demo_text', 'demo_sleep_a', 'demo_core']
        assert [s['session_id'] for s in client.get('/api/sessions').json()] == gate.public_sessions()


@pytest.mark.parametrize('mismatch', [None, 'model', 'domain', 'controls', 'signature'])
def test_research_session_reuse_preserves_state_and_refuses_incompatibility(tmp_path, mismatch):
    from deploy.huggingface.app import ensure_research_sessions, research_harness
    from plastic.harness.config import HarnessConfig
    from plastic.store import ArtifactStore, atomic_write_json
    store = ArtifactStore(str(tmp_path))
    for mid in ('lm_wikitext_l4', 'other'):
        store.register_model(mid, {'backend': 'qwen', 'domain': 'text', 'checkpoint_digest': mid})
    store.create_session('demo_core', model_id='other' if mismatch == 'model' else 'lm_wikitext_l4',
                         domain='physics' if mismatch == 'domain' else 'text',
                         harness_cfg=HarnessConfig() if mismatch == 'controls' else research_harness(),
                         runner_state={'committed_pos': 17})
    if mismatch == 'signature':
        meta = store.load_session_meta('demo_core')
        meta['model_signature'] = 'old checkpoint'
        atomic_write_json(store.session_meta_path('demo_core'), meta)
    before = store.load_session_meta('demo_core')
    if mismatch is None:
        ensure_research_sessions(store)
    else:
        with pytest.raises(ValueError, match='different|another'):
            ensure_research_sessions(store)
    assert store.load_session_meta('demo_core') == before
    assert store.load_runner_state('demo_core') == {'committed_pos': 17}
