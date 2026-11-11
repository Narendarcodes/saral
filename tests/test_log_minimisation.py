"""P0d: handoff and voice-turn logs keep identifiers, hashes and timings, never the applicant's words.

Evidence traces and the audit log are unchanged contracts, so they are pinned here too.
"""
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from decision_engine.evidence.evidence_record import stable_trace
from loan_advisory.agent.contracts import AgentRequest, Session
from loan_advisory.agent.runtime import build_offline_agent
from loan_advisory.agent.tools import HandoffQueue
from loan_advisory.agent.voice_log import TTSPrivacyFilter, mark_tts_success, record_tts_error_event, turn_row
from loan_advisory.infrastructure.evidence_store import FrappeEvidenceSink

SENTINEL = "987654321"


def test_handoff_row_keeps_only_hash_length_and_ids(tmp_path):
    secret = f"My recorded salary is wrong, it is {SENTINEL}"
    queue = HandoffQueue(str(tmp_path / "handoffs.jsonl"))
    ref = queue.create("data_dispute", ["APP-TEST-005"], secret, "req-1")

    raw = (tmp_path / "handoffs.jsonl").read_text(encoding="utf-8")
    assert secret not in raw and SENTINEL not in raw
    row = json.loads(raw)
    assert set(row) == {"handoff_ref", "category", "refs", "request_id", "message_sha256", "message_chars",
                        "created_at"}
    assert row["handoff_ref"] == ref and row["request_id"] == "req-1"
    assert row["message_sha256"] == hashlib.sha256(secret.encode()).hexdigest()
    assert row["message_chars"] == len(secret)


def test_boundary_turn_writes_no_message_text_to_handoff_or_audit(tmp_path):
    secret = f"My recorded salary is wrong, it is {SENTINEL}"
    ag, _, _ = build_offline_agent(llm=object(), profile={"legal_name": "Ref", "entity_type": "nbfc"})
    ag.audit_path = tmp_path / "turns.jsonl"
    ag.handoff.path = tmp_path / "handoffs.jsonl"

    r = ag.run(AgentRequest(request_id="r1", conversation_id="c1", message=secret,
                            application_ref="APP-TEST-005"), Session(applicant_ref="CUS-TEST-005"))

    assert r.action == "handoff" and r.handoff_ref
    handoff_raw = (tmp_path / "handoffs.jsonl").read_text(encoding="utf-8")
    assert secret not in handoff_raw and SENTINEL not in handoff_raw
    row = json.loads(handoff_raw)
    assert row["request_id"] == "r1" and row["handoff_ref"] == r.handoff_ref
    assert row["message_sha256"] == hashlib.sha256(secret.encode()).hexdigest()

    audit_raw = (tmp_path / "turns.jsonl").read_text(encoding="utf-8")
    assert secret not in audit_raw and SENTINEL not in audit_raw
    audit = json.loads(audit_raw)
    assert audit["message_chars"] == len(secret) and "message" not in audit


def test_voice_turn_row_drops_the_transcript_and_keeps_metrics():
    marks = {"user_stopped": 10.0, "stt_done": 10.4, "graph_done": 11.0}
    row = turn_row({"transcript": f"my salary is {SENTINEL}", "locale": "en", "stt_engine": "whisper",
                    "user_audio_s": 1.25, "intent": "explain", "filler": True}, marks, 123.0)

    assert "transcript" not in row and SENTINEL not in json.dumps(row)
    assert row["ts"] == 123.0 and row["locale"] == "en" and row["stt_engine"] == "whisper"
    assert row["user_audio_s"] == 1.25 and row["filler"] is True and row["intent"] == "explain"
    assert row["stt_done_ms"] == 400.0 and row["graph_done_ms"] == 1000.0


def test_voice_turn_records_tts_stages_and_first_audible_proxy_without_private_text():
    secret = f"caller transcript and digits {SENTINEL}"
    info = {"transcript": secret, "locale": "te", "tts_provider": "gnani"}
    record_tts_error_event(info, SimpleNamespace(error=RuntimeError(secret), recoverable=True))
    mark_tts_success(info)
    marks = {"user_stopped": 10.0, "tts_request_start": 10.2, "tts_first_audio": 10.6,
             "graph_done": 10.8, "agent_speaking": 10.9}
    row = turn_row(info, marks, 123.0)

    raw = json.dumps(row, ensure_ascii=False)
    assert secret not in raw and SENTINEL not in raw and "transcript" not in row
    assert row["tts_provider"] == "gnani" and row["tts_status"] == "ok" and row["tts_recovered"] is True
    assert row["tts_error_count"] == 1 and row["tts_retry_count"] == 1 and row["tts_error_kind"] == "provider_error"
    assert row["tts_request_start_ms"] == 200.0 and row["tts_first_audio_ms"] == 600.0
    assert row["first_audible_ms"] == row["agent_speaking_ms"] == 900.0
    assert row["first_audible_stage"] == "livekit_agent_speaking"


def test_tts_sdk_warning_logs_drop_error_body_but_keep_safe_provider_status():
    secret = f"provider echoed caller words {SENTINEL}"
    sanitizer = TTSPrivacyFilter()
    gnani = logging.LogRecord("livekit.plugins.gnani.log", logging.ERROR, "", 0,
                              "Gnani TTS SSE error: %s - %s", (503, secret), None)
    retry = logging.LogRecord("livekit.agents", logging.WARNING, "", 0,
                              "failed to synthesize speech: %s, retrying in %ss", (secret, 2), None)

    sanitizer.filter(gnani)
    sanitizer.filter(retry)
    assert secret not in gnani.getMessage() and "503" in gnani.getMessage()
    assert secret not in retry.getMessage()


def test_voice_worker_never_adds_the_transcript_to_the_turn_info():
    src = (Path(__file__).resolve().parents[1] / "apps" / "voice" / "agent_worker.py").read_text(encoding="utf-8")
    assert "transcript=j.get" not in src


def test_evidence_trace_content_is_written_unchanged(tmp_path):
    class Client:
        base_url = "http://frappe"

        def call_method(self, method, payload):
            return {"record_id": payload["record"]["record_id"], "wrote_native_status": True}

    sink = FrappeEvidenceSink(client=Client(), trace_dir=str(tmp_path / "traces"))
    record = {
        "trace_ref": {"sha256": "abc123"}, "record_id": "rec-1", "application_ref": "APP-1", "decision": "ERROR",
        "lifecycle_state": "REVIEW",
        "policy": {"id": "p", "version": "1", "policy_sha256": "a", "jdm_sha256": "b"},
        "inputs": {"input_sha256": "c"}, "engine": {"name": "ZEN", "version": "1"},
        "evaluated_at": "2026-01-01T00:00:00", "is_simulation": True,
    }
    trace = {"steps": [{"event": "answer", "note": SENTINEL}]}

    sink.persist(record, "Open", trace, expected_application="APP-1")

    raw = (tmp_path / "traces" / "abc123.json").read_text(encoding="utf-8")
    assert raw == json.dumps(stable_trace(trace), sort_keys=True, ensure_ascii=False)
    assert SENTINEL in raw  # trace content is not minimised
