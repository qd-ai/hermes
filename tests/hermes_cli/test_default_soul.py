"""The shipped persona, and the places that must seed it identically.

``DEFAULT_SOUL_MD`` is the runtime's copy (``_ensure_default_soul_md`` seeds it
on first run; the installers no longer write their own). The Docker image still
ships a literal copy, so drift there would give containers a different persona
from the one the code believes in.
"""

from pathlib import Path

from hermes_cli.default_soul import DEFAULT_SOUL_MD, is_legacy_template_soul

REPO = Path(__file__).resolve().parents[2]

# The persona this fork replaced. Kept verbatim because one test below pins how
# an install still carrying it is treated.
_UPSTREAM_SOUL = (
    "You are Hermes Agent, an intelligent AI assistant created by Nous Research. "
    "You are helpful, knowledgeable, and direct. You assist users with a wide "
    "range of tasks including answering questions, writing and editing code, "
    "analyzing information, creative work, and executing actions via your tools. "
    "You communicate clearly, admit uncertainty when appropriate, and prioritize "
    "being genuinely useful over being verbose unless otherwise directed below. "
    "Be targeted and efficient in your exploration and investigations."
)


def test_persona_names_the_product_not_the_backend():
    assert DEFAULT_SOUL_MD.startswith("# Office Agent Helper")
    assert "https://officeagent.us/" in DEFAULT_SOUL_MD
    assert "Hermes" not in DEFAULT_SOUL_MD
    assert "Nous Research" not in DEFAULT_SOUL_MD


def test_persona_keeps_the_general_assistant_role():
    # It is the only agent the product ships: a persona scoped to product
    # support alone would leave the app unable to do the user's own work.
    assert "writing and editing code" in DEFAULT_SOUL_MD


def test_docker_image_seeds_the_same_bytes():
    soul = (REPO / "docker" / "SOUL.md").read_text(encoding="utf-8")
    assert soul == DEFAULT_SOUL_MD


def test_an_install_still_carrying_the_old_upstream_persona_upgrades():
    # That text was auto-seeded, never written by the user, so it carries no
    # user intent: the in-place upgrade replaces it with the Office Agent
    # persona. The shipped persona itself is never mistaken for a scaffold.
    assert is_legacy_template_soul(_UPSTREAM_SOUL)
    assert not is_legacy_template_soul(DEFAULT_SOUL_MD)
