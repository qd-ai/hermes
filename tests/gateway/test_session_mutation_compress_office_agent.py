"""``/compress`` from Desktop and Ink names the session it compacts to Office Agent.

``session.mutate(compress)`` summarizes with a compressor of its own, outside any agent's turn and outside
``AIAgent._compress_context``, so nothing else binds the session its summary request belongs to.
"""
from types import SimpleNamespace

import pytest

import hermes_state_runtime as rt
from agent.context_compressor import ContextCompressor
from agent.office_agent_headers import office_agent_headers
from gateway import run
from gateway.config import GatewayConfig
from gateway.session import SessionStore
from gateway.session_authority import SessionAuthority
from gateway.session_controls import AuthorityConnection
from gateway.session_local import create_local_session


@pytest.mark.asyncio
async def test_canonical_compress_names_the_session_it_compacts(tmp_path, monkeypatch):
    named = []

    def compress(self, messages, **kw):
        named.append(office_agent_headers('https://model.officeagent.us/deepseek/v4-flash/v1').get('X-OA-Session-Id'))
        return [{'role': 'user', 'content': 'summary', '_compressed_summary': True}]

    monkeypatch.setattr(run, '_load_gateway_config', dict)
    monkeypatch.setattr(ContextCompressor, '__init__', lambda self, *a, **k: None)
    monkeypatch.setattr(ContextCompressor, 'compress', compress)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    epoch = rt.begin_runtime_epoch(db, instance_id='owner')
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None,
                             _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    authority = SessionAuthority(runner, profile_id=str(tmp_path / 'home'), instance_id='owner', db=db, epoch=epoch)
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    ref = create_local_session(authority, owner.actor, dict(request_id='oa', source='cli', cwd=str(tmp_path),
                                                            model='frozen', toolsets=[]))

    async def mutate(operation):
        row = db.get_session(ref.session_id)
        reply = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': operation, 'expected_revision': row['runtime_revision'],
            'expected_generation': row['runtime_generation'], 'operation': operation, 'payload': {}}})
        assert 'result' in reply, reply
        return reply['result']

    try:
        # After a reset the conversation the chat route points at is another session than the route's own id.
        target = (await mutate('reset'))['target_session_id']
        assert target != ref.session_id
        for i in range(4):
            db.append_message(target, 'user', f'question {i}')
            db.append_message(target, 'assistant', f'answer {i}')

        await mutate('compress')

        assert named == [target]
    finally:
        await owner.close()
