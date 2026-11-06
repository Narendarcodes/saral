"""Voice worker wiring for phone (SIP) callers: caller-ID lookup, the `verify_caller` tool, the llm_node gate that keeps
the advisory server unreachable until the caller is verified, DTMF event routing, and the masked read-back that goes
straight to TTS. LiveKit is imported but never connected; Gnani and Frappe are never called."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
VOICE = ROOT / "apps" / "voice"
KEY = "test-only-hmac-key-0123456789abcdef"
PHONE, APP4, BANK4 = "+910000123456", "4729", "8051"
ENTRY = {"phone": PHONE, "applicant_id": "CUS-TEST-001", "application_id": "APP-TEST-001",
         "application_last4": APP4, "bank_account_last4": BANK4}


@pytest.fixture
def cv(monkeypatch):
    monkeypatch.syspath_prepend(str(VOICE))
    import caller_verification
    return caller_verification


@pytest.fixture
def worker(monkeypatch):
    pytest.importorskip("livekit.agents")
    pytest.importorskip("livekit.plugins.gnani")
    monkeypatch.setenv("GNANI_API_KEY", "test-dummy-gnani-key")
    monkeypatch.syspath_prepend(str(VOICE))
    sys.modules.pop("agent_worker", None)
    import agent_worker
    return agent_worker


@pytest.fixture
def directory_env(tmp_path, cv):
    import json
    path = tmp_path / "callers.json"
    path.write_text(json.dumps(cv.build_seed(KEY.encode(), [ENTRY])))
    return {cv.KEY_ENV: KEY, cv.DIRECTORY_ENV: str(path)}


class FakeTTS:
    def __init__(self):
        self.synthesized = []

    def synthesize(self, text):
        self.synthesized.append(text)
        tts = self

        class _S:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def __aiter__(self):
                async def gen():
                    yield SimpleNamespace(frame=f"frame:{len(tts.synthesized)}")
                return gen()
        return _S()


class FakeSession:
    """Stands in for AgentSession.say: records text, chat-context flag, and drains any direct audio."""

    def __init__(self, on_say=None):
        self.tts, self.said, self.on_say = FakeTTS(), [], on_say

    def say(self, text, *, audio=None, allow_interruptions=None, add_to_chat_ctx=True):
        async def done():
            frames = [f async for f in audio] if audio is not None else []
            self.said.append({"text": text, "add_to_chat_ctx": add_to_chat_ctx, "frames": frames})
            if self.on_say:
                self.on_say(text)
        return done()


def make_agent(worker, cv, record=None, reason=None, on_say=None):
    fake = FakeSession(on_say)

    class TestAgent(worker.AdvisorAgent):
        @property
        def session(self):
            return fake

    calls = []

    async def exchange():
        calls.append(1)
        return {"assertion": "x"}
    gate = cv.VerifiedGate(None, exchange)
    phone = worker.PhoneCall(record, reason)
    agent = TestAgent(gate, "en", room=None, turn=worker.Turn(), phone=phone)
    if record is not None:
        phone.verifier = gate.verifier = cv.KeypadVerifier(record, KEY.encode(), speak=agent._say,
                                                           speak_private=agent._say_private, nudge_s=0.05, give_up_s=0.2)
    return agent, fake, calls


async def drain(gen):
    return [x async for x in gen]


def user_ctx(text):
    from livekit.agents import llm
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content=text)
    return ctx


# ----------------------------------------------------------------------------------------------- caller ID
def test_phone_lookup_known_unknown_and_fail_closed(worker, cv, directory_env):
    rec, key, reason = worker.phone_lookup(PHONE, directory_env)
    assert rec.application_id == "APP-TEST-001" and key == KEY.encode() and reason is None
    assert worker.phone_lookup("+910000999999", directory_env) == (None, None, cv.UNKNOWN_CALLER)
    assert worker.phone_lookup(None, directory_env)[2] == cv.UNKNOWN_CALLER
    assert worker.phone_lookup(PHONE, {cv.DIRECTORY_ENV: directory_env[cv.DIRECTORY_ENV]})[2] == cv.UNAVAILABLE
    assert worker.phone_lookup(PHONE, {cv.KEY_ENV: KEY, cv.DIRECTORY_ENV: "/nonexistent.json"})[2] == cv.UNAVAILABLE


def test_unknown_caller_declines_account_info_and_offers_a_person(worker, cv):
    agent, fake, calls = make_agent(worker, cv, reason=cv.UNKNOWN_CALLER)
    out = asyncio.run(drain(agent.llm_node(user_ctx("what is my loan status"), [], None)))
    assert out == [cv.PROMPT["unknown_caller"]] and "team" in out[0] and calls == [] and agent._http is None
    assert asyncio.run(agent.verify_caller()) == cv.UNKNOWN_CALLER
    assert fake.said[-1]["text"] == cv.PROMPT["unknown_caller"]


# ----------------------------------------------------------------------------------------------- gate
def test_unverified_caller_never_reaches_the_advisory_server(worker, cv, directory_env):
    rec, _, _ = worker.phone_lookup(PHONE, directory_env)
    agent, fake, calls = make_agent(worker, cv, rec)

    async def go():
        out = await drain(agent.llm_node(user_ctx("why was my loan rejected"), [], None))
        agent._verify_task.cancel()
        return out
    out = asyncio.run(go())
    assert out == [cv.PROMPT["needs_verification"]] and calls == [] and agent._http is None


def test_locked_caller_gets_the_handover_line(worker, cv, directory_env):
    rec, _, _ = worker.phone_lookup(PHONE, directory_env)
    agent, fake, calls = make_agent(worker, cv, rec)
    agent.phone.verifier.locked = True
    out = asyncio.run(drain(agent.llm_node(user_ctx("hello"), [], None)))
    assert out == [cv.PROMPT["locked"]] and calls == []


def test_browser_session_is_not_gated(worker, cv):
    agent, _, _ = make_agent(worker, cv, reason=cv.UNKNOWN_CALLER)
    agent.phone = None
    assert agent._gate_reply() is None and asyncio.run(agent.verify_caller()) == cv.NOT_REQUIRED


# ----------------------------------------------------------------------------------------------- tool + DTMF
def test_verify_caller_is_a_function_tool(worker):
    from livekit.agents.llm import find_function_tools
    names = {getattr(t, "info", None) and t.info.name for t in find_function_tools(worker.AdvisorAgent)}
    assert "verify_caller" in names


def test_full_keypad_flow_through_the_agent_keeps_digits_out_of_chat_and_transcript(worker, cv, directory_env):
    rec, _, _ = worker.phone_lookup(PHONE, directory_env)
    script = ["1", APP4, "1"]
    holder = {}

    def on_say(text):
        if script and (text in (cv.PROMPT["menu"], cv.PROMPT["enter"]) or text.startswith("You entered")):
            handler = holder["h"]
            for k in script.pop(0):
                handler(SimpleNamespace(digit=k, code=None, participant=SimpleNamespace(identity="sip_caller")))

    agent, fake, calls = make_agent(worker, cv, rec, on_say=on_say)
    holder["h"] = cv.dtmf_handler("sip_caller", agent.phone.verifier)
    assert asyncio.run(agent.verify_caller()) == cv.VERIFIED and agent.auth.verified
    assert all(s["add_to_chat_ctx"] is False for s in fake.said)
    shown = " ".join(s["text"] for s in fake.said)
    assert "You entered ****." in shown and APP4 not in shown and "4, 7, 2, 9" not in shown
    # the real digits went to TTS only, as the fixed template
    assert fake.tts.synthesized == ["You entered 4, 7, 2, 9. Press 1 to confirm or 2 to retype."]
    assert [s for s in fake.said if s["frames"]][0]["text"] == "You entered ****. Press 1 to confirm or 2 to retype."
    assert agent._gate_reply() is None  # verified: turns may now reach the server
    assert asyncio.run(agent.auth.fresh()) == {"assertion": "x"} and calls == [1]


def test_dtmf_from_another_participant_or_outside_an_attempt_is_ignored(cv):
    pressed = []
    v = SimpleNamespace(press=pressed.append)
    h = cv.dtmf_handler("sip_caller", v)
    h(SimpleNamespace(digit="5", code=5, participant=SimpleNamespace(identity="someone_else")))
    h(SimpleNamespace(digit="5", code=5, participant=None))
    h(SimpleNamespace(digit="", code=10, participant=SimpleNamespace(identity="sip_caller")))
    h(SimpleNamespace(digit="7", code=7, participant=SimpleNamespace(identity="sip_caller")))
    assert pressed == ["*", "7"]
    cv.dtmf_handler("sip_caller", None)(SimpleNamespace(digit="1", participant=SimpleNamespace(identity="sip_caller")))


def test_worker_listens_to_the_sip_dtmf_room_event_not_speech(worker):
    src = (VOICE / "agent_worker.py").read_text(encoding="utf-8")
    assert 'ctx.room.on("sip_dtmf_received"' in src and "GetDtmfTask" not in src
    from livekit import rtc
    assert "sip_dtmf_received" in rtc.room.EventTypes.__args__ and {"digit", "code", "participant"} <= set(
        rtc.SipDTMF.__dataclass_fields__)


def test_worker_logs_no_digits_or_numbers(worker):
    src = (VOICE / "agent_worker.py").read_text(encoding="utf-8")
    for bad in ("log.info(f", "sip.phoneNumber\")}", "print("):
        assert bad not in src
