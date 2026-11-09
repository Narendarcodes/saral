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


def make_agent(worker, cv, record=None, reason=None, on_say=None, prompts=None):
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
    agent = TestAgent(gate, "en", room=None, turn=worker.Turn(), phone=phone, prompts=prompts)
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


@pytest.mark.parametrize("clips", [True, False], ids=["pre-recorded", "missing-clips-fallback"])
def test_full_keypad_flow_through_the_agent_keeps_digits_out_of_chat_and_transcript(worker, cv, directory_env, clips,
                                                                                   tmp_path):
    import auth_prompts as ap
    rec, _, _ = worker.phone_lookup(PHONE, directory_env)
    script = ["1", APP4, "1"]
    holder = {}

    def on_say(text):
        if script and (text in (cv.PROMPT["menu"], cv.PROMPT["enter"]) or text.startswith("You entered")):
            handler = holder["h"]
            for k in script.pop(0):
                handler(SimpleNamespace(digit=k, code=None, participant=SimpleNamespace(identity="sip_caller")))

    store = ap.PromptStore() if clips else ap.PromptStore(root=tmp_path)
    agent, fake, calls = make_agent(worker, cv, rec, on_say=on_say, prompts=store)
    holder["h"] = cv.dtmf_handler("sip_caller", agent.phone.verifier)
    assert asyncio.run(agent.verify_caller()) == cv.VERIFIED and agent.auth.verified
    assert all(s["add_to_chat_ctx"] is False for s in fake.said)
    shown = " ".join(s["text"] for s in fake.said)
    assert "You entered ****." in shown and APP4 not in shown and "4, 7, 2, 9" not in shown
    readback = [s for s in fake.said if s["text"] == "You entered ****. Press 1 to confirm or 2 to retype."]
    assert len(readback) == 1 and readback[0]["frames"]
    if clips:  # every line, the read-back included, came from pre-recorded clips: live TTS was never called
        assert fake.tts.synthesized == []
        upto = [s["text"] for s in fake.said].index(cv.PROMPT["verified"]) + 1
        assert all(s["frames"] and not isinstance(s["frames"][0], str) for s in fake.said[:upto])
    else:  # no clips: the same fixed lines go to live TTS, the real digits only as the fixed template
        assert "You entered 4, 7, 2, 9. Press 1 to confirm or 2 to retype." in fake.tts.synthesized
        assert cv.PROMPT["menu"] in fake.tts.synthesized
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


# ----------------------------------------------------------------------------------------------- pre-recorded auth phase
async def _gate_turn(agent, text):
    """One unverified turn through the real llm_node -> tts_node chain."""
    async def chunks():
        for c in await drain(agent.llm_node(user_ctx(text), [], None)):
            yield c
    return await drain(agent.tts_node(chunks(), None))


def test_auth_phase_turn_plays_a_clip_and_calls_neither_the_graph_nor_tts(worker, cv, directory_env, monkeypatch):
    import aiohttp
    rec, _, _ = worker.phone_lookup(PHONE, directory_env)
    agent, fake, calls = make_agent(worker, cv, rec)
    agent.phone.verifier.locked = True

    def no_http(*a, **k):
        raise AssertionError("the advisory server (graph LLM) was called before verification")
    monkeypatch.setattr(aiohttp, "ClientSession", no_http)
    frames = asyncio.run(_gate_turn(agent, "why was my loan rejected"))
    from livekit import rtc
    assert frames and all(isinstance(f, rtc.AudioFrame) and f.sample_rate == 16000 for f in frames)
    assert fake.tts.synthesized == [] and calls == [] and agent._http is None
    dur = sum(f.samples_per_channel for f in frames) / 16000
    assert abs(dur - agent.prompts.load("locked")[0].__len__() / 2 / 16000) < 0.03


def test_auth_phase_falls_back_to_live_tts_with_a_warning(worker, cv, directory_env, tmp_path, caplog):
    import auth_prompts as ap
    rec, _, _ = worker.phone_lookup(PHONE, directory_env)
    agent, fake, calls = make_agent(worker, cv, rec, prompts=ap.PromptStore(root=tmp_path))
    agent.phone.verifier.locked = True
    with caplog.at_level("WARNING", logger="saral.voice.prompts"):
        frames = asyncio.run(_gate_turn(agent, "hello"))
    assert fake.tts.synthesized == [cv.PROMPT["locked"]] and frames and calls == []
    assert "using live TTS" in caplog.text


def test_greeting_and_unknown_caller_lines_are_clips(worker, cv):
    agent, fake, calls = make_agent(worker, cv, reason=cv.UNKNOWN_CALLER)
    asyncio.run(agent._say(worker.PHONE_GREETING))
    assert asyncio.run(agent.verify_caller()) == cv.UNKNOWN_CALLER
    assert [s["text"] for s in fake.said] == [cv.PROMPT["greeting"], cv.PROMPT["unknown_caller"]]
    assert fake.tts.synthesized == [] and all(s["frames"] for s in fake.said) and calls == []


def test_mixed_rate_fallback_part_is_resampled_to_one_rate(worker, cv):
    import auth_prompts as ap
    from livekit import rtc
    agent, fake, _ = make_agent(worker, cv, reason=cv.UNKNOWN_CALLER)
    tone = b"\x10\x00" * 2205  # 100 ms at 22.05 kHz, as live Timbre would return

    class RateTTS:
        def synthesize(self, text):
            class _S:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    return False

                def __aiter__(self):
                    async def gen():
                        yield SimpleNamespace(frame=rtc.AudioFrame(tone, 22050, 1, 2205))
                    return gen()
            return _S()
    fake.tts = RateTTS()
    parts = [ap.Part("clip", "you_entered", "You entered", b"\x00\x00" * 1600, 16000), ap.Part("tts", "x", "x")]
    frames = asyncio.run(drain(agent._prompt_frames(parts)))
    assert {f.sample_rate for f in frames} == {16000}


# ----------------------------------------------------------------------------------------------- tone after verification
class ToneSpy:
    def __init__(self):
        import tone_wiring as W
        self.inner, self.calls = W.ToneSession(W.make_hook("en")), []

    def __getattr__(self, name):
        attr = getattr(self.inner, name)
        if not callable(attr):
            return attr

        def wrapped(*a, **k):
            self.calls.append(name)
            return attr(*a, **k)
        return wrapped


def test_tone_starts_only_after_keypad_verification(worker, cv, directory_env):
    rec, _, _ = worker.phone_lookup(PHONE, directory_env)
    script = ["1", APP4, "1"]
    holder = {}

    def on_say(text):
        if script and (text in (cv.PROMPT["menu"], cv.PROMPT["enter"]) or text.startswith("You entered")):
            for k in script.pop(0):
                holder["h"](SimpleNamespace(digit=k, code=None, participant=SimpleNamespace(identity="sip_caller")))

    agent, fake, calls = make_agent(worker, cv, rec, on_say=on_say)
    spy = agent.tone = ToneSpy()
    holder["h"] = cv.dtmf_handler("sip_caller", agent.phone.verifier)

    async def go():
        out = await drain(agent.llm_node(user_ctx("why was my loan rejected"), [], None))  # unverified turn
        agent._verify_task.cancel()
        return out
    assert asyncio.run(go()) == [cv.PROMPT["needs_verification"]]
    assert "activate" not in spy.calls and not spy.inner.active  # the gate turn never reached the tone engine
    assert asyncio.run(agent.verify_caller()) == cv.VERIFIED
    assert spy.calls.count("activate") == 1 and spy.inner.active
    assert fake.tts.synthesized == []  # the whole auth phase, "verified" included, came from clips


def test_verified_turn_sets_speed_and_speaks_only_the_budget(worker, cv, monkeypatch):
    agent, fake, calls = make_agent(worker, cv, reason=cv.UNKNOWN_CALLER)
    agent.phone = None  # browser path: the Voice Session row authorised the call
    speeds = []
    fake.tts.set_speed = speeds.append
    published = []

    async def publish_data(data, topic=None):
        published.append(topic)
    agent.room = SimpleNamespace(local_participant=SimpleNamespace(publish_data=publish_data))
    agent._http = object()  # never used: the graph stream is faked below
    graph = ["Your application was declined because the income could not be verified.",
             "The bank statement for March was missing.",
             "You can upload it in the portal and ask for a review.",
             "A person from our team can also call you."]

    async def fake_speakable(events, after, filler, on_final, failure_text=None):
        for s in graph:
            yield s
        on_final({"answer": " ".join(graph), "intent": "explain"})
    monkeypatch.setattr(worker, "speakable", fake_speakable)

    async def go():
        await agent.on_verified()
        out = await drain(agent.llm_node(user_ctx("why was my loan rejected"), [], None))
        await asyncio.sleep(0)
        return out
    out = asyncio.run(go())
    from loan_advisory.agent import tone as T
    kept, deferred = T.fit_budget(graph, T.derive(None, "en"))
    assert out == kept and deferred and agent.turn.info["tone_deferred"] == len(deferred)
    assert speeds and all(0.9 <= s <= 1.05 for s in speeds)
    assert agent.turn.info["tone_tier"] in T.TIERS and published == ["advisor"]
