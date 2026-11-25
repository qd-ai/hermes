"""An image attached to a chat is described before its turn starts, and that request names the chat's session.

Nothing has bound a session yet at that point, so without its own binding the request to Office Agent would say
which agent and installation sent it but not which session.
"""
import asyncio
import base64
from types import SimpleNamespace

from agent import auxiliary_client as aux
from gateway.run_inbound import GatewayInboundMixin

BASE = "https://model.officeagent.us/deepseek/v4-flash/v1"
PIXEL = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


class Runner(GatewayInboundMixin):
    session_store = SimpleNamespace(peek_session_id={"agent:main:telegram:1": "sess-tg"}.get)

    def _decide_image_input_mode(self, **_kw):
        return "text"

    def _resolve_session_agent_runtime(self, **_kw):
        return "v4-flash", {"provider": "custom", "base_url": BASE, "api_key": "test-key"}


def _describe(monkeypatch, tmp_path, session_key):
    sent = []

    async def create(**kwargs):
        sent.append(kwargs.get("extra_headers") or {})
        message = SimpleNamespace(content="a single pixel")
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], model="v4-flash", usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), base_url=BASE)
    monkeypatch.setattr(aux, "resolve_vision_provider_client", lambda *_a, **_k: ("custom", client, "v4-flash"))
    image = tmp_path / "a.png"
    image.write_bytes(PIXEL)
    asyncio.run(Runner()._enrich_inbound_images(SimpleNamespace(), session_key, "look", [str(image)]))
    return sent


def test_an_attached_image_is_described_under_the_chats_own_session(monkeypatch, tmp_path):
    sent = _describe(monkeypatch, tmp_path, "agent:main:telegram:1")

    assert sent and {h.get("X-OA-Session-Id") for h in sent} == {"sess-tg"}


def test_a_chat_with_no_session_yet_still_gets_its_image_described(monkeypatch, tmp_path):
    sent = _describe(monkeypatch, tmp_path, "agent:main:telegram:new")

    assert sent and all("X-OA-Agent" in h and "X-OA-Session-Id" not in h for h in sent)
