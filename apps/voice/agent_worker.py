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

Routing (call_routing.py, decided once per room): a SIP participant, or the portal's WEB PHONE (a browser participant
whose Frappe-minted token sets `saral.mode=web_phone` and `sip.phoneNumber`, in a `webcall-` room), takes the phone
flow below; a participant whose metadata names a Voice Session takes voice mode; anything else is refused.

Phone calls (LiveKit SIP participant, or the web phone), call flow v2 "language first, then 4 digits": there is no Voice Session row, so
the caller ID (`sip.phoneNumber`) is matched to a hashed synthetic applicant directory (CALLER_DIRECTORY,
CALLER_VERIFY_HMAC_KEY) whose record names the caller's language (ten Gnani codes, default en-IN). The caller hears a
greeting in that language with no name or account data ("Welcome to Saral. To change language, press nine."), may
press 9 for a ten-language menu (each option in its own language), then keys the last 4 digits of the loan
application number (`verify_caller`, caller_verification.py) before any turn reaches the advisory server. No
read-back; three mismatches lock the call. Digits arrive as `sip_dtmf_received` room events only, and neither the chat
context nor any log ever holds a digit. Everything the caller hears before verification succeeds is PRE-RECORDED
(auth_prompts.py, apps/voice/prompts/<locale>/, rendered once by scripts/voice/render_auth_prompts.py with Timbre v3.0 /
Tarika). The auth phase never calls the advisory graph (its LLM) or live TTS; a missing or stale clip falls back to
live TTS of the same fixed text, in its language, with a warning. After the pre-recorded "Thank you, you're verified."
the session switches Prisma STT to "<language>,en-IN" (en-IN alone for English) and Timbre TTS to the language, greets
the caller by first name (learnt only now, from the phone assertion) and asks the graph to answer in that language.

Tone (tone_wiring.py -> tone_hook.py) starts only after verification (phone: keypad verified; browser: the Voice
Session row): per turn it sets the Timbre speed (0.90-1.05), adds closed-set acknowledgement / hand-over lines and
speaks only the whole graph sentences that fit the tier's word budget (the rest are deferred; the page still gets the
full answer card). Decision prefetch from ZEN is not wired yet (UnwiredDecisionPort). An unknown number is told no account details can be shared and offered a
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
import call_routing  # noqa: E402  (which door a participant came in by; no LiveKit imports)
import auth_prompts as ap  # noqa: E402  (pre-recorded auth-phase clips; no LiveKit imports)
import prompt_texts as pt  # noqa: E402  (fixed call lines in ten languages; no LiveKit imports)
from tone_wiring import ToneSession, make_hook  # noqa: E402  (tone engine, used only after verification)
from gnani_speech import build_stt, build_stt_for_language, build_tts  # noqa: E402  (Gnani Prisma STT + Timbre TTS)

CFG = VoiceSettings.from_env()
AGENT_HTTP, TURN_LOG = CFG.agent_http, CFG.turn_log
SESSION_ID = call_routing.SESSION_ID  # Voice Session ids (voice mode)
GREETING = {
    "en": "Hello. I can explain your loan decision, suggest what to work on, or answer policy questions. What would you like to know?",
    "hi": "नमस्ते। मैं आपके लोन फ़ैसले के कारण समझा सकता हूँ, सुधार के उपाय बता सकता हूँ, या नीति के सवालों का जवाब दे सकता हूँ। आप क्या जानना चाहेंगे?",
    "te": "నమస్కారం. మీ లోన్ నిర్ణయానికి కారణాలు వివరించగలను, ఏమి మెరుగుపరచాలో చెప్పగలను, లేదా నియమాల ప్రశ్నలకు జవాబివ్వగలను. మీరు ఏమి తెలుసుకోవాలనుకుంటున్నారు?",
}
FRAME_MS = 20
PHONE_MIN_INTERRUPTION_S = 0.8   # SIP: echo of our own TTS shorter than this never barges in
MAX_TIMEOUTS = 2                 # a second keypad timeout ends the call
GRAPH_LOCALES = {"en-IN": "en", "hi-IN": "hi", "te-IN": "te"}  # call languages the graph also has templates for
FIRST_NAME = re.compile(r"^[^\W\d_][^\W\d_'\-]{0,39}$")
log = logging.getLogger("saral.voice.worker")


class PhoneCall:
    """What the worker knows about a SIP caller: the matched record (or why there is none), its verifier, and the
    keypad session that carries the call's language (the verifier itself for a known caller)."""

    def __init__(self, record: cv.CallerRecord | None, reason: str | None, verifier: cv.KeypadVerifier | None = None,
                 keypad: cv.KeypadCall | None = None):
        self.record, self.reason, self.verifier = record, reason, verifier
        self.keypad = keypad or verifier


def graph_locale(language: str) -> str:
    """The graph's template locale for a call language: en / hi / te; the other seven fall back to en templates (their
    composed answers are still written in the call language, see `reply_language`)."""
    return GRAPH_LOCALES.get(language, "en")


def first_name(raw) -> str | None:
    """A first name fit to speak, or None (then the greeting has no name)."""
    name = str(raw or "").strip().split(" ")[0] if raw else ""
    return name if FIRST_NAME.match(name) else None


def phone_lookup(caller_id: str | None, env=None) -> tuple[cv.CallerRecord | None, bytes | None, str | None]:
    """(record, key, reason). Fails closed: no key or no directory means nobody can be verified on this call. Logs
    which reason fired; the caller number appears masked (last 2 digits only) and no digit is ever logged."""
    env = os.environ if env is None else env
    key, path = cv.load_key(env), env.get(cv.DIRECTORY_ENV, "")
    if key is None:
        log.warning("caller verification unavailable: %s missing or shorter than %d bytes", cv.KEY_ENV,
                    cv.MIN_KEY_BYTES)
        return None, None, cv.UNAVAILABLE
    if not path:
        log.warning("caller verification unavailable: %s is not set", cv.DIRECTORY_ENV)
        return None, None, cv.UNAVAILABLE
    try:
        directory = cv.CallerDirectory.load(path, key)
    except (OSError, ValueError, KeyError, TypeError) as e:
        # the path and contents stay out of the log; the error class says which kind of failure it was
        log.warning("caller verification unavailable: caller directory unreadable (%s)", type(e).__name__)
        return None, None, cv.UNAVAILABLE
    phone = cv.normalize_phone(caller_id)
    if phone is None:
        log.info("unknown caller: no usable caller ID on the SIP participant")
        return None, None, cv.UNKNOWN_CALLER
    record = directory.lookup(phone)
    if record is None:
        log.info("unknown caller: no directory match for %s (%d records)", cv.mask_phone(phone), len(directory))
        return None, None, cv.UNKNOWN_CALLER
    log.info("caller ID matched %s language=%s", cv.mask_phone(phone), record.language)
    return record, key, None


def phone_session_kwargs(cls=None) -> dict:
    """Barge-in settings for SIP calls, where the phone line can echo our own TTS back: a longer minimum interruption
    and false-interruption resume. Only the keywords this livekit-agents version accepts are passed."""
    import inspect
    cls = cls or AgentSession
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return {}
    want = {"min_interruption_duration": PHONE_MIN_INTERRUPTION_S, "false_interruption_timeout": 1.5,
            "resume_false_interruption": True}
    return {k: v for k, v in want.items() if k in params}


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
                 prompts: ap.PromptStore | None = None, tone: ToneSession | None = None):
        super().__init__(instructions="Loan advisory assistant (replies come from the LangGraph advisory graph). "
                                      "Phone callers must pass verify_caller before any account detail is discussed.")
        self.auth, self.locale, self.room, self.turn, self.phone = auth, locale, room, turn, phone
        self.conv = f"voice-{auth.session_id}"
        self._http: aiohttp.ClientSession | None = None
        self._verify_task: asyncio.Task | None = None
        self.prompts = prompts or ap.PromptStore(locale=locale)
        self._gate_pending = False
        self._tone_default = tone is None
        self.tone = tone or ToneSession(make_hook(locale))
        self.call_language: str | None = None  # set for phone calls once verified (the language the call ended up in)
        self.timeouts = 0
        self.ended = False
        self.end_call = None  # async () -> None, set by phone_entrypoint: hang up (delete the room)

    @property
    def language(self) -> str:
        """The phone call's current language (Gnani code): the keypad's until verification, then fixed."""
        if self.call_language:
            return self.call_language
        keypad = getattr(self.phone, "keypad", None) if self.phone is not None else None
        return keypad.language if keypad is not None else pt.DEFAULT_LANGUAGE

    def apply_language(self, language: str, *, stt: bool = False) -> None:
        """Timbre TTS speaks `language` from the next synthesis (the live-TTS fallback before verification, every
        reply after it). With `stt`, Prisma STT is swapped for one listening for "<language>,en-IN"."""
        upd = getattr(self.session.tts, "update_options", None)
        if callable(upd):
            upd(language=language)
        if stt:
            self.update_options(stt=build_stt_for_language(CFG.speech, language))
        self.locale = graph_locale(language)

    # ------------------------------------------------------------------------------------- phone verification
    @llm.function_tool
    async def verify_caller(self) -> str:
        """Verify a phone caller with their phone KEYPAD before any account or application detail is discussed.
        The caller keys the last 4 digits of their loan application number (9 during the greeting changes language).
        Never ask the caller to say digits aloud. Returns only the outcome: verified, locked, timeout,
        unknown_caller, unavailable or not_required; it never returns digits."""
        return await self.run_verification()

    async def run_verification(self) -> str:
        if self.phone is None:
            return cv.NOT_REQUIRED  # browser sessions are authorised by Frappe's Voice Session row
        reason = self.phone.reason or cv.UNAVAILABLE
        if self.ended:
            return reason if self.phone.verifier is None else cv.LOCKED
        if self.phone.verifier is None:
            keypad = self.phone.keypad
            if keypad is not None and keypad.busy:
                return reason
            if keypad is not None and not keypad.greeted:
                await keypad.run_unverifiable(reason)  # en-IN greeting + press 9, then the refusal, once
            else:
                await self._say_key(reason)
            await self._hang_up()
            return reason
        result = await self.phone.verifier.run()
        if result == cv.VERIFIED:
            await self.on_verified()
        elif result == cv.LOCKED:
            await self._hang_up()
        elif result == cv.TIMEOUT:
            self.timeouts += 1
            if self.timeouts >= MAX_TIMEOUTS:
                await self._hang_up()
        return result

    async def _hang_up(self) -> None:
        """End the call politely once the refusal / locked line has finished playing (session.say is awaited to
        playout before this runs). Idempotent."""
        if self.ended:
            return
        self.ended = True
        log.info("phone call ended before verification")
        if self.end_call is not None:
            try:
                await self.end_call()
            except Exception:
                log.warning("could not end the call cleanly")

    async def on_verified(self, speak: bool = True) -> None:
        """Hand-over from the pre-recorded auth phase to the dynamic agent. Phone: fix the call language on STT, TTS
        and the graph, then greet the caller by first name (live TTS; the name is fetched only now). Then start the
        tone engine (never before). `speak=False` (browser calls, which open with their own greeting) starts tone
        without its opening lines."""
        if self.phone is not None:
            self.call_language = self.language
            self.apply_language(self.call_language, stt=True)
            if self._tone_default:
                self.tone = ToneSession(make_hook(self.locale))
            name = None
            try:
                name = first_name((await self.auth.fresh()).get("first_name"))
            except Exception:  # the greeting goes on without a name; the reason stays out of the log
                log.warning("caller first name unavailable; greeting without it")
            if speak:
                await self.session.say(pt.live_greeting(self.call_language, name), allow_interruptions=True)
        try:
            lines = await self.tone.activate()
        except Exception:  # tone is an enhancement; the verified call goes on without it
            log.warning("tone engine could not start; continuing with the neutral tone")
            return
        for line in lines if speak and self._tone_lines_fit() else []:
            await self.session.say(line, add_to_chat_ctx=False, allow_interruptions=True)

    def _tone_lines_fit(self) -> bool:
        """Tone's closed-set lines exist in en / hi / te only: a call in another language skips them."""
        return self.phone is None or self.language in GRAPH_LOCALES

    def start_verification(self) -> None:
        if self._verify_task is None or self._verify_task.done():
            self._verify_task = asyncio.ensure_future(self.run_verification())
            self._verify_task.add_done_callback(lambda t: t.cancelled() or t.exception())

    def _gate_reply(self) -> str | None:
        """None when this turn may reach the advisory server. Otherwise "" (nothing is spoken for the turn): before
        verification the caller's speech is ignored. The only reaction is the keypad nudge, at most once per prompt,
        or (after a keypad timeout) restarting verification. Refusal lines are never replayed per utterance."""
        if self.phone is None or getattr(self.auth, "verified", False):
            return None
        if self.ended:
            return ""
        v, keypad = self.phone.verifier, self.phone.keypad
        if keypad is not None and keypad.busy:
            if keypad.may_nudge():
                self._spawn(self._say_key("nudge"))
            return ""
        if v is None or v.locked:
            return ""  # the refusal is (being) spoken once and the call ends after it
        self.start_verification()  # e.g. "say something to try again" after a keypad timeout
        return ""

    def _spawn(self, coro) -> None:
        task = asyncio.ensure_future(coro)
        task.add_done_callback(lambda t: t.cancelled() or t.exception())

    def _auth_phase(self) -> bool:
        return self.phone is not None and not getattr(self.auth, "verified", False)

    async def _prompt_frames(self, parts: list[ap.Part]):
        """LiveKit frames for a planned prompt: clip and silence PCM in 20 ms frames; a `tts` part (missing clip) is
        synthesized live. Every frame leaves at the first frame's rate, because the session's resampler is chosen once
        per utterance."""
        rate, resampler, resampler_in = None, None, None
        for part in parts:
            if part.kind == "tts":
                frames = await self._synthesize_in(part.text, part.language)
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

    async def _synthesize_in(self, text: str, language: str | None) -> list:
        """Live TTS of a fixed line whose clip is missing, in that line's language (a language-menu option can be in
        a language other than the call's); the TTS language is put back afterwards."""
        tts = self.session.tts
        current = getattr(tts, "language", None)
        upd = getattr(tts, "update_options", None)
        switch = bool(language) and callable(upd) and current is not None and language != current
        if switch:
            upd(language=language)
        try:
            async with tts.synthesize(text) as stream:
                return [ev.frame async for ev in stream]
        finally:
            if switch:
                upd(language=current)

    async def _say_key(self, key: str) -> None:
        """A fixed auth-phase line by key, in the call's current language: played from its pre-recorded clip (live
        TTS of the same text only when the clip is missing), kept out of the chat context."""
        lang = self.language
        shown = pt.menu_text() if key == cv.LANGUAGE_MENU else pt.text(lang, key)
        # Never interruptible: a "hello" over a refusal or a prompt must not cut it off and make it restart.
        await self.session.say(shown, audio=self._prompt_frames(self.prompts.plan_key(key, lang)),
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
        plan = self.tone.plan_turn(msg)  # None before verification / for an inactive tone engine
        speed = self.tone.speed(plan)
        set_speed = getattr(self.session.tts, "set_speed", None)
        if speed is not None and callable(set_speed):
            set_speed(speed)
        budget = self.tone.budget(plan)
        speak_answer = self.tone.speak_answer(plan)
        lines_fit = self._tone_lines_fit()
        if plan is not None:
            self.turn.info.update(tone_tier=plan.tone.tier, tone_speed=speed, tone_fired=list(plan.fired))
        for line in self.tone.lead_lines(plan) if lines_fit else []:
            yield line
        self.turn.mark("graph_start")
        self.turn.info["locale"] = self.locale
        self.turn.info["session_id"] = self.auth.session_id
        ctx = self.turn.start_span()
        self._http = self._http or aiohttp.ClientSession()
        # The body carries the question only: identity, application and language come from the
        # assertion, so a tampered worker cannot change whose file is explained.
        body = {"message": msg or "hello", "channel": "voice", "conversation_id": self.conv}
        if self.phone is not None and self.call_language:
            # The language the caller chose on the keypad: the graph writes its answer in it. Not an identity claim.
            body["reply_language"] = self.call_language
        final: dict = {}

        async def events():
            """One fresh assertion, then NDJSON events from the advisory server (sentence streaming)."""
            assertion = await self.auth.fresh()
            self.locale = graph_locale(self.call_language) if self.call_language else (assertion.get("locale") or self.locale)
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
            elif not speak_answer or (budget is not None and not budget.admit(text)):
                continue  # deferred by the tone budget (whole sentences only; the page still gets the answer card)
            else:
                self.turn.mark("first_sentence")
            yield text
        for line in self.tone.closing_lines(plan) if lines_fit else []:
            yield line
        if budget is not None:
            self.turn.info["tone_deferred"] = budget.deferred
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
                async for frame in self._prompt_frames(self.prompts.plan(sentence, self.language)):
                    yield frame
                continue
            t0 = time.perf_counter()
            async with tts_engine.synthesize(sentence) as stream:
                async for ev in stream:
                    if "tts_first_audio" not in self.turn.t:
                        self.turn.mark("tts_first_audio")
                        self.turn.info.setdefault("tts_ttfa_ms", round((time.perf_counter() - t0) * 1000, 1))
                    yield ev.frame


def wire_tone_events(session: AgentSession, agent: AdvisorAgent) -> None:
    """Barge-in feeds the tone engine (a no-op until the call is verified)."""

    @session.on("user_state_changed")
    def _barge(ev):
        if getattr(ev, "new_state", None) == "speaking" and getattr(session, "agent_state", None) == "speaking":
            agent.tone.on_barge_in()


def _end_tone(agent: AdvisorAgent):
    async def _end(*_a):
        agent.tone.end()  # wipes the call cache
    return _end


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=CFG.vad_min_silence_s)


def participant_route(participant, room_name: str) -> str:
    """call_routing.route for a LiveKit participant: PHONE (SIP), WEB_PHONE (portal web phone), VOICE or REFUSE."""
    return call_routing.route(is_sip=participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
                              attributes=participant.attributes, metadata=participant.metadata, room=room_name)


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    route = participant_route(participant, ctx.room.name)
    if call_routing.is_phone_flow(route):
        # A SIP call and the portal's web phone run the same keypad flow; only the web phone may also key by data.
        await phone_entrypoint(ctx, participant, web_phone=route == call_routing.WEB_PHONE)
        return
    session_id = call_routing.session_id(participant.metadata)
    if route != call_routing.VOICE or session_id is None:
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

    agent = AdvisorAgent(auth, locale, ctx.room, turn)

    @session.on("agent_state_changed")
    def _a(ev):
        if getattr(ev, "new_state", None) == "speaking":
            turn.mark("agent_speaking")
            turn.flush()
        elif getattr(ev, "old_state", None) == "speaking":
            agent.tone.on_agent_turn_end()

    wire_tone_events(session, agent)
    ctx.add_shutdown_callback(_end_tone(agent))
    await session.start(agent=agent, room=ctx.room)
    await agent.on_verified(speak=False)  # the Voice Session row authorised this call (auth.fresh() above)
    await session.say(GREETING[locale], allow_interruptions=True)
    await asyncio.sleep(0)


async def phone_entrypoint(ctx: JobContext, participant: rtc.RemoteParticipant, web_phone: bool = False):
    """A SIP caller: caller-ID lookup, greeting in the record's language (press 9 for the language menu), then keypad
    verification before any advisory turn. An unknown caller hears the en-IN greeting and the unknown-caller line.

    `web_phone`: the portal's browser phone (call_routing.WEB_PHONE). Same flow; its caller ID is the `sip.phoneNumber`
    attribute Frappe put in the token, and its keypad presses arrive as `sip_dtmf_received` (livekit-client
    `publishDtmf`) or, as a site-selectable fallback, as data packets on topic "dtmf" from that participant only."""
    if web_phone:
        log.info("web phone call joined")
    record, key, reason = phone_lookup(cv.caller_id(participant.attributes, participant.identity))
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
    language = record.language if record is not None else pt.DEFAULT_LANGUAGE
    locale, turn = graph_locale(language), Turn()
    stt_engine = build_stt_for_language(CFG.speech, language)
    tts_engine = build_tts(CFG.speech, locale)
    tts_engine.update_options(language=language)
    session = AgentSession(vad=ctx.proc.userdata["vad"], stt=stt_engine, tts=tts_engine, llm=GraphLLM(),
                           min_endpointing_delay=CFG.min_endpointing_s, **phone_session_kwargs())
    agent = AdvisorAgent(gate, locale, ctx.room, turn, phone=phone)

    async def hang_up():
        try:
            await ctx.delete_room()
        except Exception:
            ctx.shutdown("call ended before verification")
    agent.end_call = hang_up
    on_language = lambda code: agent.apply_language(code)  # noqa: E731  (fallback TTS follows the menu choice)
    if record is not None:
        phone.verifier = phone.keypad = gate.verifier = cv.KeypadVerifier(record, key, speak=agent._say_key,
                                                                          on_language=on_language)
    else:
        phone.keypad = cv.KeypadCall(speak=agent._say_key, language=pt.DEFAULT_LANGUAGE, on_language=on_language)
    # Keypad digits come only from LiveKit SIP DTMF room events, never from the speech path.
    ctx.room.on("sip_dtmf_received", cv.dtmf_handler(participant.identity, phone.keypad))
    if web_phone:
        ctx.room.on("data_received", cv.dtmf_data_handler(participant.identity, phone.keypad))

    @session.on("agent_state_changed")
    def _a(ev):
        if getattr(ev, "old_state", None) == "speaking":
            agent.tone.on_agent_turn_end()

    wire_tone_events(session, agent)
    ctx.add_shutdown_callback(_end_tone(agent))
    await session.start(agent=agent, room=ctx.room)
    agent.start_verification()  # greeting (pre-recorded, record's language) -> press 9 window -> 4 digits


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, port=int(os.environ.get("VOICE_WORKER_PORT", "8081"))))
