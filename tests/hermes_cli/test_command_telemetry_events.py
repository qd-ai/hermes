"""Office Agent command telemetry: the two event bodies and the size fitter.

Contracts (OA ingest, ``POST /api/v1/telemetry/commands``):
- ``command.started`` / ``command.finished`` carry exactly the contract's fields, in its shapes
- ``execution.exit_code`` is always present (``null`` when the process gave none); other absent
  optional values are left out, never sent as ``null``
- no body is ever over the cap (4,096 bytes, under OA's 5,120 limit), and no event is dropped for
  size: the preview keeps its start, the command text its start and end around an exact
  ``…[<n> bytes omitted]…`` marker, the cwd its end
"""

import json
import re

import pytest

from hermes_cli.observability.command_telemetry_events import (
    CommandOutcome,
    CommandRecord,
    build_finished,
    build_started,
)
from hermes_cli.observability.command_telemetry_fit import MAX_BODY_BYTES, encode, fit_event

OA_LIMIT = 5120
MARKER = re.compile(r"\n…\[(\d+) bytes omitted\]…\n")

SAMPLE = CommandRecord(
    instance_id="hermes-host-001", session_id="session_123", command_id="cmd_789", tool="terminal",
    text="npm install express", tool_call_id="call_456", cwd="/workspace/project",
)


def _decoded(body: bytes) -> dict:
    assert len(body) <= MAX_BODY_BYTES <= OA_LIMIT
    return json.loads(body.decode("utf-8"))


def test_started_event_matches_the_contract_example():
    event = build_started(SAMPLE, event_id="evt_001", timestamp="2026-11-10T20:00:00Z")

    assert event == {
        "schema_version": "1.0",
        "event_id": "evt_001",
        "event_type": "command.started",
        "timestamp": "2026-11-10T20:00:00Z",
        "source": {"agent": "hermes", "instance_id": "hermes-host-001", "session_id": "session_123",
                   "tool_call_id": "call_456"},
        "command": {"id": "cmd_789", "tool": "terminal", "text": "npm install express", "cwd": "/workspace/project"},
    }


def test_finished_event_matches_the_contract_example():
    outcome = CommandOutcome("completed", 0, 1250, "added 57 packages...", 8540)
    event = build_finished(SAMPLE, outcome, event_id="evt_002", timestamp="2026-11-10T20:00:02Z")

    assert event == {
        "schema_version": "1.0",
        "event_id": "evt_002",
        "event_type": "command.finished",
        "timestamp": "2026-11-10T20:00:02Z",
        "source": {"agent": "hermes", "instance_id": "hermes-host-001", "session_id": "session_123",
                   "tool_call_id": "call_456"},
        "command": {"id": "cmd_789"},
        "execution": {"status": "completed", "exit_code": 0, "duration_ms": 1250,
                      "output": {"preview": "added 57 packages...", "total_bytes": 8540, "truncated": True}},
    }


def test_no_exit_code_is_sent_as_null_and_absent_optionals_are_left_out():
    bare = CommandRecord(instance_id="i", session_id="s", command_id="c", tool="terminal", text="sleep 99")
    finished = _decoded(fit_event(build_finished(bare, CommandOutcome("timed_out"), event_id="e", timestamp="t")))
    started = _decoded(fit_event(build_started(bare, event_id="e0", timestamp="t")))

    assert finished["execution"] == {"status": "timed_out", "exit_code": None}
    assert "tool_call_id" not in finished["source"] and "cwd" not in started["command"]
    assert None not in started["source"].values() and None not in started["command"].values()


def test_over_long_optional_source_fields_are_left_out():
    record = CommandRecord(instance_id="i", session_id="s", command_id="c", tool="t" * 300, text="ls",
                           tool_call_id="x" * 257)
    event = build_started(record, event_id="e", timestamp="t")

    assert "tool_call_id" not in event["source"] and "tool" not in event["command"]
    assert event["source"]["agent"] == "hermes"


def test_an_event_under_the_limit_is_sent_unchanged():
    event = build_started(SAMPLE, event_id="evt_001", timestamp="2026-11-10T20:00:00Z")

    assert fit_event(event) == encode(event)


def test_a_50kb_output_preview_is_cut_to_fit_and_keeps_the_full_size():
    output = "".join(f"line {n}: added package-{n}\n" for n in range(2500))
    assert len(output.encode()) > 50_000
    outcome = CommandOutcome("completed", 0, 12, output, len(output.encode()))

    body = _decoded(fit_event(build_finished(SAMPLE, outcome, event_id="e", timestamp="t")))

    block = body["execution"]["output"]
    assert block["truncated"] is True
    assert block["total_bytes"] == len(output.encode())
    assert block["preview"] and output.startswith(block["preview"])


def test_a_50kb_multibyte_command_keeps_its_start_and_end_with_an_exact_marker():
    text = "cat <<'EOF' > run.sh\n" + "部署😀ünïcødé-" * 4000 + "\nEOF\nchmod +x run.sh && ./run.sh | sh"
    assert len(text.encode()) > 50_000
    record = CommandRecord(instance_id="i", session_id="s", command_id="c", tool="terminal", text=text)

    body = fit_event(build_started(record, event_id="e", timestamp="t"))
    sent = _decoded(body)["command"]["text"]  # decode("utf-8") above proves valid UTF-8

    match = MARKER.search(sent)
    assert match, sent[:80]
    head, tail = sent[:match.start()], sent[match.end():]
    assert head and tail and text.startswith(head) and text.endswith(tail)
    assert tail.endswith("./run.sh | sh")
    assert int(match.group(1)) == len(text.encode()) - len(head.encode()) - len(tail.encode())
    assert len(head) > len(tail)  # ~3/4 of the room goes to the start


@pytest.mark.parametrize("unit", ['"q"', "\\", "\n", "\x00\x01\x1f", "\t\r\b\f", " </script>"])
def test_text_full_of_json_escapes_still_fits(unit):
    text = unit * 20_000
    record = CommandRecord(instance_id="i", session_id="s", command_id="c", tool="terminal", text=text,
                           cwd="/w")
    outcome = CommandOutcome("failed", 1, 3, text, len(text.encode()))

    started = _decoded(fit_event(build_started(record, event_id="e", timestamp="t")))
    finished = _decoded(fit_event(build_finished(record, outcome, event_id="e", timestamp="t")))

    assert MARKER.search(started["command"]["text"])
    assert text.startswith(finished["execution"]["output"]["preview"])


def test_a_huge_cwd_keeps_its_most_specific_end():
    cwd = "/" + "/".join(f"dir{n}" for n in range(2000)) + "/project/src"
    record = CommandRecord(instance_id="i", session_id="s", command_id="c", tool="terminal", text="make", cwd=cwd)

    sent = _decoded(fit_event(build_started(record, event_id="e", timestamp="t")))["command"]

    assert sent["text"] == "make"
    assert sent["cwd"].endswith("/project/src") and cwd.endswith(sent["cwd"])


def test_ids_status_and_type_are_never_shortened():
    record = CommandRecord(instance_id="i" * 256, session_id="s" * 256, command_id="c" * 256, tool="terminal",
                           text="x" * 9000, tool_call_id="t" * 256, cwd="/" * 9000)
    outcome = CommandOutcome("s" * 64, None, 1, "o" * 9000, 9000)

    for event in (build_started(record, event_id="e" * 128, timestamp="t"),
                  build_finished(record, outcome, event_id="e" * 128, timestamp="t")):
        sent = _decoded(fit_event(event))
        assert sent["event_id"] == "e" * 128 and sent["event_type"] == event["event_type"]
        assert sent["source"]["instance_id"] == "i" * 256 and sent["source"]["session_id"] == "s" * 256
        assert sent["command"]["id"] == "c" * 256
    assert sent["execution"]["status"] == "s" * 64
