"""Voice worker wiring for phone (SIP) callers, call flow v2: caller-ID lookup (with logged, masked reasons), the
greeting in the directory language and the press-9 menu, the `verify_caller` tool, the llm_node gate that keeps the
advisory server unreachable and IGNORES speech until the caller is verified (one nudge per prompt at most, refusals
spoken once and never replayed, then the call ends), non-interruptible pre-verification clips, and the chosen language
reaching STT / TTS / the graph after verification together with the first-name greeting. LiveKit is imported but never
connected; Gnani and Frappe are never called."""
import asyncio
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
VOICE = ROOT / "apps" / "voice"
KEY = "test-only-hmac-key-0123456789abcdef"
PHONE, APP4 = "+910000123456", "4729"
ENTRY = {"phone": PHONE, "applicant_id": "CUS-TEST-001", "application_id": "APP-TEST-001",
         "application_last4": APP4, "language": "te-IN"}


@pytest.fixture
def cv(monkeypatch):
    monkeypatch.syspath_prepend(str(VOICE))
    import caller_verification
    return caller_verification


@pytest.fixture
def pt(monkeypatch):
    monkeypatch.syspath_prepend(str(VOICE))
    import prompt_texts
    return prompt_texts


@pytest.fixture
def worker(monkeypatch):
    pytest.importorskip("livekit.agents")
    pytest.importorskip("livekit.plugins.gnani")
    monkeypatch.setenv("GNANI_API_KEY", "test-dummy-gnani-key")
    monkeypatch.syspath_prepend(str(VOICE))
    sys.modules.pop("agent_worker", None)
    import agent_worker
    return agent_worker


def write_directory(tmp_path, cv, entries=(ENTRY,)):
    path = tmp_path / "callers.json"
    path.write_text(json.dumps(cv.build_seed(KEY.encode(), list(entries))))
    return {cv.KEY_ENV: KEY, cv.DIRECTORY_ENV: str(path)}


@pytest.fixture
def directory_env(tmp_path, cv):
    return write_directory(tmp_path, cv)


class FakeTTS:
    def __init__(self, language="te-IN"):
        self.synthesized, self.language, self.languages = [], language, []

    def update_options(self, language=None, **_):
        if language is not None:
            self.language = language

    def synthesize(self, text):
        self.synthesized.append(text)
        self.languages.append(self.language)
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
    """Stands in for AgentSession.say: records text, chat-context flag, interruptibility, and drains direct audio."""

    def __init__(self, on_say=None):
        self.tts, self.said, self.on_say = FakeTTS(), [], on_say

    def say(self, text, *, audio=None, allow_interruptions=None, add_to_chat_ctx=True):
        async def done():
            frames = [f async for f in audio] if audio is not None else []
            self.said.append({"text": text, "add_to_chat_ctx": add_to_chat_ctx, "frames": frames,
                              "allow_interruptions": allow_interruptions})
            if self.on_say:
                self.on_say(text)
        return done()


def make_agent(worker, cv, record=None, reason=None, on_say=None, prompts=None, first_name="Asha"):
    fake = FakeSession(on_say)

    class TestAgent(worker.AdvisorAgent):
        @property
        def session(self):
            return fake

    calls = []

    async def exchange():
        calls.append(1)
        return {"assertion": "x", "first_name": first_name}
    gate = cv.VerifiedGate(None, exchange)
    phone = worker.PhoneCall(record, reason)
    agent = TestAgent(gate, "en", room=None, turn=worker.Turn(), phone=phone, prompts=prompts)
    hangups = []

    async def end_call():
        hangups.append(1)
    agent.end_call = end_call
    kw = dict(speak=agent._say_key, on_language=agent.apply_language, nudge_s=0.05, give_up_s=0.2, window_s=0.05,
              menu_wait_s=0.05)
    if record is not None:
        phone.verifier = phone.keypad = gate.verifier = cv.KeypadVerifier(record, KEY.encode(), **kw)
    else:
        phone.keypad = cv.KeypadCall(**kw)
    return agent, fake, calls, hangups


async def drain(gen):
    return [x async for x in gen]


def user_ctx(text):
    from livekit.agents import llm
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content=text)
    return ctx


def keypad_script(cv, pt, holder, script):
    """on_say hook: when a prompt that waits for keys finishes, key the next scripted presses (None = nothing)."""
    waits = {pt.text(code, k) for code in pt.LANGUAGES for k in ("greeting", "enter", "no_match")} | {pt.menu_text()}

    def on_say(text):
        if script and text in waits:
            for k in script.pop(0) or "":
                holder["h"](SimpleNamespace(digit=k, code=None, participant=SimpleNamespace(identity="sip_caller")))
    return on_say


# ----------------------------------------------------------------------------------------------- caller ID
def test_phone_lookup_known_unknown_and_fail_closed(worker, cv, directory_env):
    rec, key, reason = worker.phone_lookup(PHONE, directory_env)
    assert rec.application_id == "APP-TEST-001" and rec.language == "te-IN" and key == KEY.encode() and reason is None
    assert worker.phone_lookup("+910000999999", directory_env) == (None, None, cv.UNKNOWN_CALLER)
    assert worker.phone_lookup(None, directory_env)[2] == cv.UNKNOWN_CALLER
    assert worker.phone_lookup(PHONE, {cv.DIRECTORY_ENV: directory_env[cv.DIRECTORY_ENV]})[2] == cv.UNAVAILABLE
    assert worker.phone_lookup(PHONE, {cv.KEY_ENV: KEY, cv.DIRECTORY_ENV: "/nonexistent.json"})[2] == cv.UNAVAILABLE


@pytest.mark.parametrize("raw", ["9876543210", "09876543210", "919876543210", "+91 98765 43210",
                                 "sip:+919876543210@trunk.example"])
def test_phone_lookup_matches_every_indian_form(worker, cv, tmp_path, raw):
    env = write_directory(tmp_path, cv, [dict(ENTRY, phone="+919876543210")])
    rec, _, reason = worker.phone_lookup(raw, env)
    assert rec is not None and reason is None


def test_phone_lookup_logs_which_reason_fired_without_the_number(worker, cv, directory_env, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="saral.voice.worker")
    worker.phone_lookup("+919876543210", {cv.KEY_ENV: KEY})
    assert "CALLER_DIRECTORY is not set" in caplog.text
    caplog.clear()
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    worker.phone_lookup("+919876543210", {cv.KEY_ENV: KEY, cv.DIRECTORY_ENV: str(bad)})
    assert "caller directory unreadable" in caplog.text and str(bad) not in caplog.text
    caplog.clear()
    worker.phone_lookup("+919876543210", {cv.DIRECTORY_ENV: directory_env[cv.DIRECTORY_ENV]})
    assert "CALLER_VERIFY_HMAC_KEY missing" in caplog.text
    caplog.clear()
    worker.phone_lookup("+919876543210", directory_env)
    assert "no directory match for +**********10" in caplog.text
    caplog.clear()
    worker.phone_lookup(PHONE, directory_env)
    assert "caller ID matched +**********56 language=te-IN" in caplog.text
    for secret in ("9876543210", "98765", "0000123456", APP4):
        assert secret not in caplog.text


def test_entrypoint_reads_caller_id_with_attribute_fallbacks(worker):
    src = (VOICE / "agent_worker.py").read_text(encoding="utf-8")
    assert "cv.caller_id(participant.attributes, participant.identity)" in src
    assert 'attributes.get("sip.phoneNumber")' not in src


# ----------------------------------------------------------------------------------------------- unknown caller
def test_unknown_caller_hears_greeting_and_one_refusal_then_the_call_ends(worker, cv, pt):
    agent, fake, calls, hangups = make_agent(worker, cv, reason=cv.UNKNOWN_CALLER)
    assert asyncio.run(agent.verify_caller()) == cv.UNKNOWN_CALLER
    assert [s["text"] for s in fake.said] == [pt.text("en-IN", "greeting"), pt.text("en-IN", "unknown_caller")]
    assert all(s["allow_interruptions"] is False and s["add_to_chat_ctx"] is False for s in fake.said)
    assert hangups == [1] and agent.ended and calls == [] and fake.tts.synthesized == []
    # the caller keeps saying "hello": nothing is replayed, nothing reaches the server, no second hang-up
    for text in ("hi", "hello", "what is my loan status"):
        assert asyncio.run(drain(agent.llm_node(user_ctx(text), [], None))) == []
    assert asyncio.run(agent.verify_caller()) == cv.UNKNOWN_CALLER
    assert len(fake.said) == 2 and hangups == [1] and agent._http is None


def test_unavailable_directory_is_the_same_single_refusal(worker, cv, pt):
    agent, fake, calls, hangups = make_agent(worker, cv, reason=cv.UNAVAILABLE)
    asyncio.run(agent.verify_caller())
    assert [s["text"] for s in fake.said][-1] == pt.text("en-IN", "unavailable") and hangups == [1]


def test_unknown_caller_speech_during_greeting_is_ignored(worker, cv):
    agent, fake, _, _ = make_agent(worker, cv, reason=cv.UNKNOWN_CALLER)

    async def go():
        task = asyncio.ensure_future(agent.run_verification())
        await asyncio.sleep(0)
        out = await drain(agent.llm_node(user_ctx("hello"), [], None))
        await task
        return out
    assert asyncio.run(go()) == []
    texts = [s["text"] for s in fake.said]
    assert texts.count(texts[-1]) == 1  # the refusal once


def test_unknown_caller_can_switch_language_before_the_refusal(worker, cv, pt):
    holder, script = {}, ["9", "3"]
    agent, fake, _, _ = make_agent(worker, cv, reason=cv.UNKNOWN_CALLER)
    fake.on_say = keypad_script(cv, pt, holder, script)
    holder["h"] = cv.dtmf_handler("sip_caller", agent.phone.keypad)
    asyncio.run(agent.verify_caller())
    assert fake.said[-1]["text"] == pt.text("te-IN", "unknown_caller")
    assert fake.said[1]["text"] == pt.menu_text()


# ----------------------------------------------------------------------------------------------- gate
def test_unverified_speech_is_ignored_with_at_most_one_nudge_per_prompt(worker, cv, pt, directory_env):
    rec, _, _ = worker.phone_lookup(PHONE, directory_env)
    agent, fake, calls, _ = make_agent(worker, cv, rec)
    agent.phone.verifier.give_up_s = agent.phone.verifier.nudge_s = 5

    async def go():
        task = asyncio.ensure_future(agent.run_verification())
        for _ in range(20):  # wait until the enter-digits prompt is waiting for keys
            await asyncio.sleep(0.02)
            if agent.phone.verifier.step >= 2:
                break
        outs = [await drain(agent.llm_node(user_ctx(t), [], None)) for t in ("hi", "hello", "hello?")]
        await asyncio.sleep(0.05)
        task.cancel()
        return outs
    outs = asyncio.run(go())
    assert outs == [[], [], []] and calls == [] and agent._http is None
    texts = [s["text"] for s in fake.said]
    assert texts.count(pt.text("te-IN", "nudge")) == 1
    assert pt.text("te-IN", "needs_verification") not in texts
    assert all(s["allow_interruptions"] is False for s in fake.said)


def test_locked_caller_hears_locked_once_then_the_call_ends(worker, cv, pt, directory_env):
    rec, _, _ = worker.phone_lookup(PHONE, directory_env)
    holder, script = {}, [None, "0000", "1111", "2222"]
    agent, fake, calls, hangups = make_agent(worker, cv, rec)
    fake.on_say = keypad_script(cv, pt, holder, script)
    holder["h"] = cv.dtmf_handler("sip_caller", agent.phone.verifier)
    assert asyncio.run(agent.verify_caller()) == cv.LOCKED and hangups == [1]
    assert asyncio.run(drain(agent.llm_node(user_ctx("hello"), [], None))) == []
    assert asyncio.run(agent.verify_caller()) == cv.LOCKED
    texts = [s["text"] for s in fake.said]
    assert texts.count(pt.text("te-IN", "locked")) == 1 and calls == [] and hangups == [1]


def test_second_keypad_timeout_ends_the_call(worker, cv, pt, directory_env):
    rec, _, _ = worker.phone_lookup(PHONE, directory_env)
    agent, fake, _, hangups = make_agent(worker, cv, rec)
    async def go():
        first = await agent.verify_caller()
        after_first = list(hangups)
        return first, after_first, await agent.verify_caller()
    first, after_first, second = asyncio.run(go())
    assert first == second == cv.TIMEOUT and after_first == [] and hangups == [1]


def test_browser_session_is_not_gated(worker, cv):
    agent, _, _, _ = make_agent(worker, cv, reason=cv.UNKNOWN_CALLER)
    agent.phone = None
    assert agent._gate_reply() is None and asyncio.run(agent.verify_caller()) == cv.NOT_REQUIRED


# ----------------------------------------------------------------------------------------------- tool + DTMF
def test_verify_caller_is_a_function_tool(worker):
    from livekit.agents.llm import find_function_tools
    names = {getattr(t, "info", None) and t.info.name for t in find_function_tools(worker.AdvisorAgent)}
    assert "verify_caller" in names


@pytest.mark.parametrize("clips", [True, False], ids=["pre-recorded", "missing-clips-fallback"])
def test_full_flow_language_menu_then_digits_then_verified_in_the_chosen_language(worker, cv, pt, directory_env,
                                                                                    clips, tmp_path):
    import auth_prompts as ap
    rec, _, _ = worker.phone_lookup(PHONE, directory_env)
    holder, script = {}, ["9", "4", APP4]  # te-IN record, caller picks Tamil
    store = ap.PromptStore() if clips else ap.PromptStore(root=tmp_path)
    agent, fake, calls, hangups = make_agent(worker, cv, rec, prompts=store)
    fake.on_say = keypad_script(cv, pt, holder, script)
    holder["h"] = cv.dtmf_handler("sip_caller", agent.phone.verifier)
    assert asyncio.run(agent.verify_caller()) == cv.VERIFIED and agent.auth.verified and hangups == []
    texts = [s["text"] for s in fake.said]
    i_verified = texts.index(pt.text("ta-IN", "verified"))
    assert texts[:i_verified + 1] == [pt.text("te-IN", "greeting"), pt.menu_text(), pt.text("ta-IN", "enter"),
                                      pt.text("ta-IN", "verified")]
    pre = fake.said[:i_verified + 1]
    assert all(s["add_to_chat_ctx"] is False and s["allow_interruptions"] is False for s in pre)
    assert not any("Asha" in t for t in texts[:i_verified + 1])  # no name before verification
    assert texts[i_verified + 1] == pt.live_greeting("ta-IN", "Asha") and "Asha" in texts[i_verified + 1]
    assert APP4 not in " ".join(texts)
    if clips:  # the whole auth phase came from clips: live TTS was never called
        assert fake.tts.synthesized == []
        assert all(s["frames"] and not isinstance(s["frames"][0], str) for s in pre)
    else:  # fallback: each option in its own language, then Tamil
        assert fake.tts.synthesized[0] == pt.text("te-IN", "greeting") and fake.tts.languages[0] == "te-IN"
        assert fake.tts.languages[1:11] == list(pt.LANGUAGES)
        assert fake.tts.languages[-1] == "ta-IN"
    # the chosen language now drives STT, TTS and the graph
    assert agent.call_language == "ta-IN" and fake.tts.language == "ta-IN"
    assert agent.stt.language_code == "ta-IN,en-IN" and agent.locale == "en"
    assert agent._gate_reply() is None
    assert asyncio.run(agent.auth.fresh())["assertion"] == "x"


def test_english_caller_gets_english_stt_and_hindi_maps_to_the_graph_locale(worker, cv, pt, tmp_path):
    env = write_directory(tmp_path, cv, [dict(ENTRY, language="en-IN")])
    rec, _, _ = worker.phone_lookup(PHONE, env)
    holder, script = {}, [None, APP4]
    agent, fake, _, _ = make_agent(worker, cv, rec, first_name=None)
    fake.on_say = keypad_script(cv, pt, holder, script)
    holder["h"] = cv.dtmf_handler("sip_caller", agent.phone.verifier)
    assert asyncio.run(agent.verify_caller()) == cv.VERIFIED
    assert agent.stt.language_code == "en-IN" and fake.tts.language == "en-IN" and agent.locale == "en"
    texts = [s["text"] for s in fake.said]
    i = texts.index(pt.text("en-IN", "verified"))
    assert texts[i + 1] == pt.live_greeting("en-IN", None)  # no name available: greeting without one
    agent.apply_language("hi-IN", stt=True)
    assert agent.stt.language_code == "hi-IN,en-IN" and agent.locale == "hi"


def test_verified_turn_sends_the_reply_language_to_the_graph(worker, cv, pt, directory_env, monkeypatch):
    rec, _, _ = worker.phone_lookup(PHONE, directory_env)
    holder, script = {}, [None, APP4]
    agent, fake, _, _ = make_agent(worker, cv, rec)
    fake.on_say = keypad_script(cv, pt, holder, script)
    holder["h"] = cv.dtmf_handler("sip_caller", agent.phone.verifier)
    assert asyncio.run(agent.verify_caller()) == cv.VERIFIED
    sent = []

    class Resp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def json(self):
            return {"answer": "ok", "intent": "other"}

        @property
        def content(self):
            async def gen():
                yield json.dumps({"type": "final", "response": {"answer": "ok", "intent": "other"}}).encode()
            return gen()

    class Http:
        def post(self, url, headers=None, json=None, timeout=None):
            sent.append(json)
            return Resp()
    agent._http = Http()

    async def publish_data(data, topic=None):
        return None
    agent.room = SimpleNamespace(local_participant=SimpleNamespace(publish_data=publish_data))

    async def fake_speakable(events, after, filler, on_final, failure_text=None):
        async for ev in events:
            if ev.get("type") == "final":
                on_final(ev["response"])
        yield "ok"
    monkeypatch.setattr(worker, "speakable", fake_speakable)

    async def go():
        out = await drain(agent.llm_node(user_ctx("why was my loan rejected"), [], None))
        await asyncio.sleep(0)
        return out
    asyncio.run(go())
    assert sent and sent[0]["reply_language"] == "te-IN" and sent[0]["channel"] == "voice"
    assert APP4 not in json.dumps(sent)


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


def test_phone_session_barge_in_settings_use_only_supported_keywords(worker):
    kw = worker.phone_session_kwargs()
    from livekit.agents import AgentSession
    import inspect
    params = inspect.signature(AgentSession.__init__).parameters
    assert set(kw) <= set(params)
    if "min_interruption_duration" in params:
        assert kw["min_interruption_duration"] >= 0.8

    class Old:
        def __init__(self, vad=None, stt=None):
            pass
    assert worker.phone_session_kwargs(Old) == {}


# ----------------------------------------------------------------------------------------------- pre-recorded auth phase
def test_mixed_rate_fallback_part_is_resampled_to_one_rate(worker, cv):
    import auth_prompts as ap
    from livekit import rtc
    agent, fake, _, _ = make_agent(worker, cv, reason=cv.UNKNOWN_CALLER)
    tone = b"\x10\x00" * 2205  # 100 ms at 22.05 kHz, as live Timbre would return

    class RateTTS:
        language = "en-IN"

        def update_options(self, **_):
            pass

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
    parts = [ap.Part("clip", "enter", "x", b"\x00\x00" * 1600, 16000), ap.Part("tts", "x", "x", language="en-IN")]
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


def test_tone_starts_only_after_keypad_verification(worker, cv, pt, directory_env):
    rec, _, _ = worker.phone_lookup(PHONE, directory_env)
    holder, script = {}, [None, APP4]
    agent, fake, calls, _ = make_agent(worker, cv, rec)
    spy = agent.tone = ToneSpy()
    agent._tone_default = False
    assert asyncio.run(drain(agent.llm_node(user_ctx("why was my loan rejected"), [], None))) == []
    agent._verify_task.cancel()
    assert "activate" not in spy.calls and not spy.inner.active  # the ignored turn never reached the tone engine
    agent2, fake2, _, _ = make_agent(worker, cv, rec)
    spy2 = agent2.tone = ToneSpy()
    agent2._tone_default = False
    fake2.on_say = keypad_script(cv, pt, holder, script)
    holder["h"] = cv.dtmf_handler("sip_caller", agent2.phone.verifier)
    assert asyncio.run(agent2.verify_caller()) == cv.VERIFIED
    assert spy2.calls.count("activate") == 1 and spy2.inner.active
    assert fake2.tts.synthesized == []  # the whole auth phase, "verified" included, came from clips


def test_verified_turn_sets_speed_and_speaks_only_the_budget(worker, cv, monkeypatch):
    agent, fake, calls, _ = make_agent(worker, cv, reason=cv.UNKNOWN_CALLER)
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
