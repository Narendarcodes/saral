"""LiveKit Agents voice worker that drives the SAME LangGraph advisory graph as the text chat.

    LIVEKIT_URL=ws://127.0.0.1:7880 LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... GNANI_API_KEY=... \
    AGENT_HTTP=http://127.0.0.1:8095 python agent_worker.py start

Pipeline (speech is Gnani only; see gnani_speech.py and the GNANI_* settings):
  browser mic -> LiveKit -> Silero VAD (turn detection, barge-in) + Gnani Prisma v3.0 streaming STT (WebSocket)
  -> llm_node = POST /api/chat on the advisory server (LangGraph graph: authorization, ZEN evidence, what-if,
  policy RAG, guards; its LLM is configured there, not here) -> Gnani Timbre v3.0 TTS (voice Tarika by default)
  -> LiveKit -> browser.

Authorization: Frappe mints the LiveKit token and writes the `Voice Session` row before the browser joins, so
the row is the authorisation. Participant metadata carries `{session_id, user, customer, application}` and no
credential; for every turn the worker presents that session id (with the room and identity it is serving) to
Frappe's `voice_assertion` method and forwards the fresh 60-second assertion as `Authorization: Bearer`. Voice
therefore uses exactly the same authorization path as text, and the worker holds no borrower cookie. The worker
never sees or chooses an application on its own.

Phone calls (LiveKit SIP participant): there is no Voice Session row, so the caller ID (`sip.phoneNumber`) is matched
to a hashed synthetic applicant directory (CALLER_DIRECTORY, CALLER_VERIFY_HMAC_KEY) and the caller must pass keypad
verification (`verify_caller`, caller_verification.py) before any turn reaches the advisory server. Digits arrive as
`sip_dtmf_received` room events only, and neither the chat context nor any log ever holds a digit.
Everything the caller hears before verification succeeds is PRE-RECORDED (auth_prompts.py, apps/voice/prompts/<locale>/,
rendered once by scripts/voice/render_auth_prompts.py with Timbre v3.0 / Tarika): greeting, menu, enter-digits, the
read-back stitched from "you entered" + one clip per digit + "press 1 to confirm", mismatch / retry / locked / timeout
lines and "verified, connecting you to Saral". The auth phase never calls the advisory graph (its LLM) or live TTS; a
missing or stale clip falls back to live TTS of the same fixed text with a warning. After verification the same
session continues as the normal dynamic agent. An unknown number is told no account details can be shared and offered a
person. DTMF must be enabled on the SIP trunk; this is demo-grade authentication (4 digits).

Every turn writes one JSON line (VOICE_TURN_LOG) with: end-of-user-speech -> STT done -> graph answer (and the
graph's LLM TTFT) -> first TTS audio -> agent speaking. It keeps timings, locale, STT engine, audio duration,
filler and the session id only, never the transcript. The same answer card (decision, citations) is published on
the data channel (topic "advisor") for the page to render.
"""
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import sys

import aiohttp
from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli, llm
from livekit.plugins import silero

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "api", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loan_advisory.agent import observability as obs  # noqa: E402  (OTel -> Phoenix; no-op unless AGENT_TRACING=1)
from loan_advisory.agent.settings import VoiceSettings  # noqa: E402
from loan_advisory.agent.streaming import speakable  # noqa: E402
from loan_advisory.agent.voice_log import turn_row  # noqa: E402
from loan_advisory.state_paths import append_private  # noqa: E402  (P0c private creation)
from voice_assertion_exchange import Assertions, PhoneAssertions  # noqa: E402  (Frappe exchange; no LiveKit imports)
import caller_verification as cv  # noqa: E402  (keypad caller verification; no LiveKit imports)
import auth_prompts as ap  # noqa: E402  (pre-recorded auth-phase clips; no LiveKit imports)
from gnani_speech import build_stt, build_tts  # noqa: E402  (Gnani Prisma STT + Timbre TTS)

CFG = VoiceSettings.from_env()
AGENT_HTTP, TURN_LOG = CFG.agent_http, CFG.turn_log
SESSION_ID = re.compile(r"^[0-9a-f]{32}$")
GREETING = {
    "en": "Hello. I can explain your loan decision, suggest what to work on, or answer policy questions. What would you like to know?",
    "hi": "नमस्ते। मैं आपके लोन फ़ैसले के कारण समझा सकता हूँ, सुधार के उपाय बता सकता हूँ, या नीति के सवालों का जवाब दे सकता हूँ। आप क्या जानना चाहेंगे?",
    "te": "నమస్కారం. మీ లోన్ నిర్ణయానికి కారణాలు వివరించగలను, ఏమి మెరుగుపరచాలో చెప్పగలను, లేదా నియమాల ప్రశ్నలకు జవాబివ్వగలను. మీరు ఏమి తెలుసుకోవాలనుకుంటున్నారు?",
}
PHONE_GREETING = cv.PROMPT["greeting"]
FRAME_MS = 20
log = logging.getLogger("saral.voice.worker")


class PhoneCall:
    """What the worker knows about a SIP caller: the matched record (or why there is none) and its verifier."""

    def __init__(self, record: cv.CallerRecord | None, reason: str | None, verifier: cv.KeypadVerifier | None = None):
        self.record, self.reason, self.verifier = record, reason, verifier


def phone_lookup(caller_id: str | None, env=None) -> tuple[cv.CallerRecord | None, bytes | None, str | None]:
    """(record, key, reason). Fails closed: no key or no directory means nobody can be verified on this call."""
    env = os.environ if env is None else env
    key, path = cv.load_key(env), env.get(cv.DIRECTORY_ENV, "")
    if key is None or not path:
        return None, None, cv.UNAVAILABLE
    try:
        directory = cv.CallerDirectory.load(path, key)
    except (OSError, ValueError, KeyError):
        log.warning("caller directory could not be loaded")  # the path and contents stay out of the log
        return None, None, cv.UNAVAILABLE
    record = directory.lookup(caller_id)
    return (record, key, None) if record is not None else (None, None, cv.UNKNOWN_CALLER)


class Turn:
    """Timestamps of the current turn (perf_counter seconds)."""

    def __init__(self):
        self.reset()

    def reset(self):
        sp = getattr(self, "span", None)
        if sp is not None:  # turn abandoned (e.g. barge-in) — close its span
            sp.set_attribute("voice.abandoned", True); sp.end()
        self.t = {}
        self.info = {}
        self.span = None

    @staticmethod
    def epoch_ns(t_perf):
        return time.time_ns() - int((time.perf_counter() - t_perf) * 1e9)

    def start_span(self):
        """voice.turn span, back-dated to end of user speech, with an explicit voice.stt child."""
        tr = obs.tracer()
        if tr is None or self.span is not None or "user_stopped" not in self.t:
            return None
        from opentelemetry import trace
        self.span = tr.start_span("voice.turn", start_time=self.epoch_ns(self.t["user_stopped"]),
                                  attributes={"openinference.span.kind": "CHAIN", "voice.locale": self.info.get("locale", "")})
        ctx = trace.set_span_in_context(self.span)
        if "stt_done" in self.t:
            st = tr.start_span("voice.stt", context=ctx, start_time=self.epoch_ns(self.t["user_stopped"]),
                               attributes={"openinference.span.kind": "TOOL", "stt.engine": str(self.info.get("stt_engine")),
                                           "stt.latency_ms": self.info.get("stt_ms") or 0, "stt.user_audio_s": self.info.get("user_audio_s") or 0})
            st.end(end_time=self.epoch_ns(self.t["stt_done"]))
        return ctx

    def mark(self, k):
        self.t.setdefault(k, time.perf_counter())

    def flush(self):
        t = self.t
        if "user_stopped" not in t or "graph_done" not in t:
            self.reset(); return
        row = turn_row(self.info, t, time.time())
        if self.span is not None:
            self.span.set_attributes({f"voice.{k}": v for k, v in row.items() if isinstance(v, (int, float, str)) and k != "transcript"})
            if "tts_first_audio" in t:
                self.span.add_event("tts_first_audio", timestamp=self.epoch_ns(t["tts_first_audio"]))
            self.span.end(end_time=self.epoch_ns(t.get("agent_speaking", time.perf_counter())))
            self.span = None
        with append_private(Path(TURN_LOG)) as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.reset()


class GraphLLM(llm.LLM):
    """Placeholder so AgentSession runs the pipeline; generation happens in AdvisorAgent.llm_node (the graph)."""

    @property
    def model(self) -> str:
        return "langgraph-advisor(gpt-oss-120b)"

    def chat(self, **kw):
        raise RuntimeError("generation is handled by the advisory graph in llm_node")


class AdvisorAgent(Agent):
    def __init__(self, auth, locale: str, room: rtc.Room, turn: Turn, phone: PhoneCall | None = None,
                 prompts: ap.PromptStore | None = None):
        super().__init__(instructions="Loan advisory assistant (replies come from the LangGraph advisory graph). "
                                      "Phone callers must pass verify_caller before any account detail is discussed.")
        self.auth, self.locale, self.room, self.turn, self.phone = auth, locale, room, turn, phone
        self.conv = f"voice-{auth.session_id}"
        self._http: aiohttp.ClientSession | None = None
        self._verify_task: asyncio.Task | None = None
        self.prompts = prompts or ap.PromptStore(locale=locale)
        self._gate_pending = False

    # ------------------------------------------------------------------------------------- phone verification
    @llm.function_tool
    async def verify_caller(self) -> str:
        """Verify a phone caller with their phone KEYPAD before any account or application detail is discussed.
        The caller picks one factor (1: loan application number, 2: bank account number) and keys its last 4 digits.
        Never ask the caller to say digits aloud. Returns only the outcome: verified, locked, timeout,
        unknown_caller, unavailable or not_required; it never returns digits."""
        return await self.run_verification()

    async def run_verification(self) -> str:
        if self.phone is None:
            return cv.NOT_REQUIRED  # browser sessions are authorised by Frappe's Voice Session row
        if self.phone.verifier is None:
            await self._say(cv.PROMPT[self.phone.reason or cv.UNAVAILABLE])
            return self.phone.reason or cv.UNAVAILABLE
        return await self.phone.verifier.run()

    def start_verification(self) -> None:
        if self._verify_task is None or self._verify_task.done():
            self._verify_task = asyncio.ensure_future(self.run_verification())
            self._verify_task.add_done_callback(lambda t: t.cancelled() or t.exception())

    def _gate_reply(self) -> str | None:
        """None when this turn may reach the advisory server; otherwise the fixed line to speak instead."""
        if self.phone is None or getattr(self.auth, "verified", False):
            return None
        v = self.phone.verifier
        if v is None:
            return cv.PROMPT[self.phone.reason or cv.UNAVAILABLE]
        if v.locked:
            return cv.PROMPT["locked"]
        if v.busy:
            return ""  # the keypad flow is speaking; say nothing over it
        self.start_verification()
        return cv.PROMPT["needs_verification"]

    def _auth_phase(self) -> bool:
        return self.phone is not None and not getattr(self.auth, "verified", False)

    async def _prompt_frames(self, parts: list[ap.Part]):
        """LiveKit frames for a planned prompt: clip and silence PCM in 20 ms frames; a `tts` part (missing clip) is
        synthesized live. Every frame leaves at the first frame's rate, because the session's resampler is chosen once
        per utterance."""
        rate, resampler, resampler_in = None, None, None
        for part in parts:
            if part.kind == "tts":
                async with self.session.tts.synthesize(part.text) as stream:
                    frames = [ev.frame async for ev in stream]
            else:
                n = part.sample_rate * FRAME_MS // 1000
                step = n * ap.SAMPLE_WIDTH
                frames = []
                for i in range(0, len(part.pcm), step):
                    chunk = part.pcm[i:i + step]
                    frames.append(rtc.AudioFrame(chunk, part.sample_rate, 1, len(chunk) // ap.SAMPLE_WIDTH))
            for frame in frames:
                fr = getattr(frame, "sample_rate", None)
                if rate is None:
                    rate = fr
                if fr is None or rate is None or fr == rate:
                    yield frame
                    continue
                if resampler is None or resampler_in != fr:
                    resampler, resampler_in = rtc.AudioResampler(input_rate=fr, output_rate=rate, num_channels=1), fr
                for out in resampler.push(frame):
                    yield out
            if resampler is not None:
                for out in resampler.flush():
                    yield out
                resampler = None

    async def _say(self, text: str) -> None:
        """Fixed verification prompts: played from their pre-recorded clip (live TTS of the same text only when the
        clip is missing), kept out of the chat context."""
        await self.session.say(text, audio=self._prompt_frames(self.prompts.plan(text)), add_to_chat_ctx=False,
                               allow_interruptions=True)

    async def _say_private(self, spoken: str, shown: str) -> None:
        """The digit read-back, stitched from clips ("you entered", one clip per digit, "press 1 to confirm"); the
        whole `spoken` line goes to live TTS only when a clip is missing. Never through llm_node or the chat context;
        the transcript the room and chat context could show is the masked `shown` text."""
        await self.session.say(shown, audio=self._prompt_frames(self.prompts.plan_readback(spoken)),
                               add_to_chat_ctx=False, allow_interruptions=False)

    async def llm_node(self, chat_ctx: llm.ChatContext, tools, model_settings):
        gate = self._gate_reply()
        if gate is not None:  # unverified phone caller: the advisory server is never called, so no fact can leak
            if gate:
                self._gate_pending = True  # tts_node plays this fixed line from its clip
                yield gate
            return
        msg = ""
        for item in reversed(chat_ctx.items):
            if getattr(item, "role", None) == "user":
                msg = item.text_content or ""
                break
        self.turn.mark("graph_start")
        self.turn.info["locale"] = self.locale
        self.turn.info["session_id"] = self.auth.session_id
        ctx = self.turn.start_span()
        self._http = self._http or aiohttp.ClientSession()
        # The body carries the question only: identity, application and language come from the
        # assertion, so a tampered worker cannot change whose file is explained.
        body = {"message": msg or "hello", "channel": "voice", "conversation_id": self.conv}
        final: dict = {}

        async def events():
            """One fresh assertion, then NDJSON events from the advisory server (sentence streaming)."""
            assertion = await self.auth.fresh()
            self.locale = assertion.get("locale") or self.locale
            headers = {"Authorization": f"Bearer {assertion['assertion']}"}
            if ctx is not None:  # W3C traceparent -> the server's advisor.turn becomes a child of this voice.turn
                from opentelemetry.propagate import inject
                inject(headers, context=ctx)
            path = "/api/chat/stream" if CFG.stream else "/api/chat"
            async with self._http.post(f"{AGENT_HTTP}{path}", headers=headers, json=body,
                                       timeout=aiohttp.ClientTimeout(total=CFG.http_timeout_s)) as r:
                if not CFG.stream:
                    yield {"type": "final", "response": await r.json()}
                    return
                async for raw in r.content:
                    line = raw.decode("utf-8", "replace").strip()
                    if line:
                        yield json.loads(line)

        def on_final(resp):
            final.update(resp)

        n = 0
        async for text in speakable(events(), CFG.filler_after_s, CFG.filler.get(self.locale, CFG.filler["en"]), on_final,
                                    failure_text=CFG.unreachable.get(self.locale, CFG.unreachable["en"])):
            n += 1
            if text == CFG.filler.get(self.locale):
                self.turn.info["filler"] = True
            else:
                self.turn.mark("first_sentence")
            yield text
        self.turn.mark("graph_done")
        resp = final or {"answer": "", "intent": "other", "error": "no final response"}
        self._report(resp, msg, n)

    def _report(self, resp, msg, n_chunks):
        timings = resp.get("timings") or {}
        calls = timings.get("llm_calls") or []
        comp = [c for c in calls if str(c.get("tag", "")).startswith("compose")]
        self.turn.info.update(intent=resp.get("intent"), action=resp.get("action"), locale=self.locale,
                              graph_total_ms=timings.get("total_ms"), graph_first_sentence_ms=timings.get("first_sentence_ms"),
                              spoken_chunks=n_chunks,
                              llm_ttft_ms=round(comp[-1]["ttft_s"] * 1000, 1) if comp and comp[-1].get("ttft_s") else None,
                              classify_ms=round(calls[0]["total_s"] * 1000, 1) if calls and calls[0].get("tag") == "classify" else None,
                              guard_flags=resp.get("guard_flags"), answer_chars=len(resp.get("answer", "")))
        task = asyncio.ensure_future(self.room.local_participant.publish_data(json.dumps(
            {k: resp.get(k) for k in ("answer", "intent", "action", "abstained", "citations", "decision_card", "improvement_plan",
                                      "what_if", "handoff_ref")} | {"transcript": msg}, ensure_ascii=False).encode(), topic="advisor"))
        task.add_done_callback(lambda t: t.exception())  # best effort; never break the turn

    async def tts_node(self, text, model_settings):
        """Every chunk from llm_node is already a complete, guarded sentence: synthesize each one as soon as it
        arrives (no extra sentence buffering by the default tokenizer)."""
        tts_engine = self.session.tts
        async for sentence in text:
            if not sentence.strip():
                continue
            if self._gate_pending or self._auth_phase():  # auth phase: fixed lines are clips, never live synthesis
                self._gate_pending = False
                async for frame in self._prompt_frames(self.prompts.plan(sentence)):
                    yield frame
                continue
            t0 = time.perf_counter()
            async with tts_engine.synthesize(sentence) as stream:
                async for ev in stream:
                    if "tts_first_audio" not in self.turn.t:
                        self.turn.mark("tts_first_audio")
                        self.turn.info.setdefault("tts_ttfa_ms", round((time.perf_counter() - t0) * 1000, 1))
                    yield ev.frame


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=CFG.vad_min_silence_s)


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        await phone_entrypoint(ctx, participant)
        return
    meta = json.loads(participant.metadata or "{}")
    session_id = meta.get("session_id")
    if not isinstance(session_id, str) or not SESSION_ID.match(session_id):
        # No session row means no authorisation: refuse the room instead of greeting a stranger.
        raise RuntimeError("voice session metadata is missing; leaving the room")
    http = aiohttp.ClientSession()
    auth = Assertions(http, session_id, ctx.room.name, participant.identity,
                      frappe_url=CFG.frappe_url, api_key=CFG.frappe_api_key,
                      api_secret=CFG.frappe_api_secret, timeout_s=CFG.http_timeout_s)
    first = await auth.fresh()  # fail fast: an unverified session never hears a greeting
    locale = first.get("locale") if first.get("locale") in ("en", "hi", "te") else "en"
    turn = Turn()
    stt_engine, tts_engine = build_stt(CFG.speech, locale), build_tts(CFG.speech, locale)
    session = AgentSession(vad=ctx.proc.userdata["vad"], stt=stt_engine, tts=tts_engine, llm=GraphLLM(),
                           min_endpointing_delay=CFG.min_endpointing_s)

    @session.on("user_input_transcribed")
    def _t(ev):
        if getattr(ev, "is_final", False):  # timings and engine only; the words never enter the turn info
            turn.mark("stt_done")
            turn.info.update(stt_engine=f"gnani/{stt_engine.model}")
            if "user_stopped" in turn.t:
                turn.info["stt_ms"] = round((turn.t["stt_done"] - turn.t["user_stopped"]) * 1000, 1)

    @session.on("user_state_changed")
    def _u(ev):
        if getattr(ev, "new_state", None) == "speaking":
            turn.reset()
        elif getattr(ev, "new_state", None) == "listening":
            turn.mark("user_stopped")

    @session.on("agent_state_changed")
    def _a(ev):
        if getattr(ev, "new_state", None) == "speaking":
            turn.mark("agent_speaking")
            turn.flush()

    agent = AdvisorAgent(auth, locale, ctx.room, turn)
    await session.start(agent=agent, room=ctx.room)
    await session.say(GREETING[locale], allow_interruptions=True)
    await asyncio.sleep(0)


async def phone_entrypoint(ctx: JobContext, participant: rtc.RemoteParticipant):
    """A SIP caller: caller-ID lookup, then keypad verification before any advisory turn. Prompts are English."""
    record, key, reason = phone_lookup(participant.attributes.get("sip.phoneNumber"))
    http = aiohttp.ClientSession()
    phone = PhoneCall(record, reason)
    exchange = None
    if record is not None:
        pa = PhoneAssertions(http, record.applicant_id, record.application_id, ctx.room.name, participant.identity,
                             frappe_url=CFG.frappe_url, api_key=CFG.frappe_api_key,
                             api_secret=CFG.frappe_api_secret, timeout_s=CFG.http_timeout_s)

        async def exchange():
            return await pa.fresh(phone.verifier.factor)
    gate = cv.VerifiedGate(None, exchange)
    locale, turn = "en", Turn()
    stt_engine, tts_engine = build_stt(CFG.speech, locale), build_tts(CFG.speech, locale)
    session = AgentSession(vad=ctx.proc.userdata["vad"], stt=stt_engine, tts=tts_engine, llm=GraphLLM(),
                           min_endpointing_delay=CFG.min_endpointing_s)
    agent = AdvisorAgent(gate, locale, ctx.room, turn, phone=phone)
    if record is not None:
        phone.verifier = gate.verifier = cv.KeypadVerifier(record, key, speak=agent._say,
                                                           speak_private=agent._say_private)
    # Keypad digits come only from LiveKit SIP DTMF room events, never from the speech path.
    ctx.room.on("sip_dtmf_received", cv.dtmf_handler(participant.identity, phone.verifier))
    await session.start(agent=agent, room=ctx.room)
    await agent._say(PHONE_GREETING)  # pre-recorded, like every line before verification
    agent.start_verification()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, port=int(os.environ.get("VOICE_WORKER_PORT", "8081"))))
