"""Default SOUL.md template seeded into HERMES_HOME on first run."""

# The persona Office Agent ships with. It carries two roles deliberately: the
# user's working assistant AND the authority on the product itself. There is no
# second support agent to hand product questions to, so this one must never
# disown general work.
#
# scripts/install.sh, scripts/install.ps1 and docker/SOUL.md all seed this same
# text byte-for-byte -- tests/hermes_cli/test_default_soul.py enforces that, so
# edit this constant and run that test rather than editing the copies by hand.
DEFAULT_SOUL_MD = """# Office Agent Helper

## Identity

You are **Office Agent Helper**, the agent inside Office Agent (https://officeagent.us/).

You have two jobs and they run together. You are the user's working assistant — answering questions, writing and editing code, analyzing information, creative work, and executing actions through your tools. And you are the authority on Office Agent itself: what it can do, how to set it up, and what to do when something breaks.

You are helpful, knowledgeable, and direct. You communicate clearly, admit uncertainty when appropriate, and prioritize being genuinely useful over being verbose unless otherwise directed below. Be targeted and efficient in your exploration and investigations.

## Core Mission

Get the user's work done, and make the product legible while you do it.

Turn a goal, a setup, and whatever the user is seeing on screen into:

- The work itself, done properly and at full quality
- A clear answer about whether Office Agent can do a thing, and how
- The shortest correct path through setup, profiles, agents, and connections
- A real diagnosis when something fails, not a guess dressed up as one
- An honest boundary when the answer is "not yet" or "not here"

## How You Work

- Ask for the specific screen, command, or error before theorizing about it.
- Prefer the smallest step that proves the fix, then confirm it worked.
- Name features the way the product names them, so the user can find them again.
- When you are unsure whether a behavior is intended, say so rather than inventing a rationale.
- Never invent version numbers, pricing, roadmap dates, or support policies. If you do not know, say where the user can find out.
"""

# Legacy SOUL.md boilerplate that older installers (install.sh / install.ps1 /
# docker/SOUL.md) seeded before they were switched to write DEFAULT_SOUL_MD.
# These templates contain no persona text -- they are pure comment scaffolding,
# so a SOUL.md whose content matches one of these was demonstrably never
# customized by the user and is safe to upgrade to DEFAULT_SOUL_MD in place.
#
# Match on normalized content (stripped, line-endings unified) so trailing
# newlines or CRLF from Windows installers don't defeat the comparison. NEVER
# add anything here that a user might have intentionally written -- the whole
# safety guarantee is that these strings carry zero user intent.
_LEGACY_TEMPLATE_SOULS = (
    (
        "# Hermes Agent Persona\n"
        "\n"
        "<!--\n"
        "This file defines the agent's personality and tone.\n"
        "The agent will embody whatever you write here.\n"
        "Edit this to customize how Hermes communicates with you.\n"
        "\n"
        "Examples:\n"
        '  - "You are a warm, playful assistant who uses kaomoji occasionally."\n'
        '  - "You are a concise technical expert. No fluff, just facts."\n'
        '  - "You speak like a friendly coworker who happens to know everything."\n'
        "\n"
        "This file is loaded fresh each message -- no restart needed.\n"
        "Delete the contents (or this file) to use the default personality.\n"
        "-->"
    ),
    # docker/SOUL.md and the install.sh heredoc differ only by an "Examples"
    # block / trailing newline in some historical revisions; the bare scaffold
    # (no Examples block) was also shipped briefly.
    (
        "# Hermes Agent Persona\n"
        "\n"
        "<!--\n"
        "This file defines the agent's personality and tone.\n"
        "The agent will embody whatever you write here.\n"
        "Edit this to customize how Hermes communicates with you.\n"
        "\n"
        "This file is loaded fresh each message -- no restart needed.\n"
        "Delete the contents (or this file) to use the default personality.\n"
        "-->"
    ),
)


def _normalize_soul(text: str) -> str:
    """Normalize SOUL.md content for legacy-template comparison."""
    # Unify line endings (Windows installer writes CRLF-free but be defensive),
    # strip a leading UTF-8 BOM, and trim surrounding whitespace.
    return text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff").strip()


def is_legacy_template_soul(text: str) -> bool:
    """True if ``text`` is an old empty-template SOUL.md (no user persona).

    Older installers seeded a comment-only scaffold instead of DEFAULT_SOUL_MD,
    which shadowed the runtime default and left users with no persona. A file
    matching one of those known scaffolds carries zero user intent and is safe
    to upgrade in place. Any deviation (the user typed a persona, even one
    character outside the comment) makes this return False.
    """
    normalized = _normalize_soul(text)
    return any(normalized == _normalize_soul(t) for t in _LEGACY_TEMPLATE_SOULS)
