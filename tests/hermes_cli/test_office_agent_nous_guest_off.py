"""Office Agent ships the Nous free tier off: a fresh install sets nothing up with the Nous portal.

Contracts, for a first install with no provider key and no ``nous`` section in config.yaml:
- the desktop's welcome-trial claim creates no guest, and the free model is not active
- ``manage_connections`` is not offered, and the hosted-connector path creates no guest
Driven through the fake portal (``anon_portal``) with Hermes' real client code; the marker keeps the
shipped default that the root conftest otherwise swaps for upstream's.
"""

import pytest

from hermes_cli import anon_auth, welcome_trial
from tests.hermes_cli.anon_portal import as_local_desktop_backend, install_portal

pytestmark = pytest.mark.shipped_nous_guest


@pytest.fixture
def portal(monkeypatch, tmp_path):
    return install_portal(monkeypatch, tmp_path)


def test_a_first_desktop_install_claims_no_welcome_trial(portal, monkeypatch):
    as_local_desktop_backend(monkeypatch)

    assert welcome_trial.claim_desktop_free_tier() is None
    assert not anon_auth.free_inference_active()
    assert portal.calls == []


def test_hosted_connectors_create_no_guest(portal):
    from tools.connectors.gateway.config import connectors_available, ensure_guest_identity

    assert not connectors_available()
    assert ensure_guest_identity() is None
    assert portal.calls == []
