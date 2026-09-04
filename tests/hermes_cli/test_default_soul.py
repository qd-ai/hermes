"""The shipped persona, and the four places that must seed it identically.

``DEFAULT_SOUL_MD`` is the runtime's copy, but three installers write their own
literal copy of the same text before Python ever runs. The "MUST match" notes on
those copies were prose until this file: drift there ships fresh installs a
different persona from the one the code believes in, and nothing catches it.
"""

from pathlib import Path

from hermes_cli.default_soul import DEFAULT_SOUL_MD, is_legacy_template_soul

REPO = Path(__file__).resolve().parents[2]

# The persona this fork replaced. Kept verbatim because one test below pins the
# promise that installs already carrying it are left alone.
_UPSTREAM_SOUL = (
    "You are Hermes Agent, an intelligent AI assistant created by Nous Research. "
    "You are helpful, knowledgeable, and direct. You assist users with a wide "
    "range of tasks including answering questions, writing and editing code, "
    "analyzing information, creative work, and executing actions via your tools. "
    "You communicate clearly, admit uncertainty when appropriate, and prioritize "
    "being genuinely useful over being verbose unless otherwise directed below. "
    "Be targeted and efficient in your exploration and investigations."
)


def _install_sh_soul() -> str:
    text = (REPO / "scripts" / "install.sh").read_text(encoding="utf-8")
    body = text.split('cat > "$HERMES_HOME/SOUL.md" << \'SOUL_EOF\'\n', 1)[1]
    # The heredoc terminator eats no newline of its own: what lands on disk is
    # the body plus the newline that ends its last line.
    return body.split("\nSOUL_EOF\n", 1)[0] + "\n"


def _install_ps1_soul() -> str:
    # read_text() folds this file's CRLF to LF, which is exactly what the
    # script's own `-replace "`r`n", "`n"` does before writing.
    text = (REPO / "scripts" / "install.ps1").read_text(encoding="utf-8")
    body = text.split('$soulContent = @"\n', 1)[1].split('\n"@\n', 1)[0]
    return body + "\n"  # install.ps1 appends exactly one newline


def test_persona_names_the_product_not_the_backend():
    assert DEFAULT_SOUL_MD.startswith("# Office Agent Helper")
    assert "https://officeagent.us/" in DEFAULT_SOUL_MD
    assert "Hermes" not in DEFAULT_SOUL_MD
    assert "Nous Research" not in DEFAULT_SOUL_MD


def test_persona_keeps_the_general_assistant_role():
    # It is the only agent the product ships: a persona scoped to product
    # support alone would leave the app unable to do the user's own work.
    assert "writing and editing code" in DEFAULT_SOUL_MD


def test_install_sh_seeds_the_same_bytes():
    assert _install_sh_soul() == DEFAULT_SOUL_MD


def test_install_ps1_seeds_the_same_bytes():
    assert _install_ps1_soul() == DEFAULT_SOUL_MD


def test_docker_image_seeds_the_same_bytes():
    soul = (REPO / "docker" / "SOUL.md").read_text(encoding="utf-8")
    assert soul == DEFAULT_SOUL_MD


def test_an_install_that_already_has_the_upstream_persona_is_left_alone():
    # Only fresh installs change. The upstream persona is real seeded text, not
    # the empty comment scaffold, so the in-place upgrade must not claim it.
    assert not is_legacy_template_soul(_UPSTREAM_SOUL)
    assert not is_legacy_template_soul(DEFAULT_SOUL_MD)
