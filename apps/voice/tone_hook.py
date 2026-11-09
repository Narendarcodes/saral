"""Integration hook between the voice worker (agent_worker.py) and the tone engine. NOT wired in yet.

The worker owns LiveKit, Gnani STT/TTS and the turn loop; this hook owns tone and the fast/slow thinker. The worker
calls the hook at seven points and plays what comes back. Nothing here imports LiveKit, so it is unit-tested alone.

    hook = ToneEngineHook(port, slow_path=..., locale=meta_locale, literacy=meta_literacy, channel="voice",
                          employment_type=None, product_id=None)
    # after keypad / OTP verification succeeds (never before):
    for ev in await hook.on_verified():           # prefetch + "verified" + consent check
        play(ev)
    # each final Prisma transcript:
    plan = hook.on_user_final(text, detected_language_code)
    tts.update_options(**hook.tts_options(plan))    # speed 0.9-1.05, language en-IN/hi-IN/te-IN
    async for ev in hook.respond(plan, text=text):  # line / filler / sentence / check / handover / safety_flag
        play_or_route(ev)
    # LiveKit events:
    hook.on_barge_in(false_interruption=False)      # user_state_changed -> speaking while the agent speaks
    hook.on_agent_turn_end()                        # agent finished (or was cut off)
    plan = hook.on_silence(seconds)                 # worker timer after a question: 8 s and 15 s ("away")
    plan = hook.on_close(); hook.on_call_end()      # hand-over floor, then wipe the call cache

Events the worker must act on:
- `handover`    -> call the existing human-review / handoff path (ticket + reference) this turn.
- `safety_flag` -> route to a person urgently. The safety wording is an unapproved TODO placeholder and is never
                   returned as text; until it is approved the worker speaks only the hand-over line.
- `needs_auth`  -> a decision question before verification; answer only with the verification prompt.
- `line` / `filler` / `sentence` / `check` -> speak in order (text may carry `<pauses>` / `<slow>` tags inserted by
                   code; the LLM's own text has had every tag removed).
"""
from __future__ import annotations

import re
from typing import AsyncIterator, Mapping, Protocol

from loan_advisory.agent import explain_profile as EP
from loan_advisory.agent import live_signals as LS
from loan_advisory.agent import tone as T
from loan_advisory.agent import tone_cards as C
from tone_thinker import DecisionPort, FastThinker, SlowPath, SlowThinker, TurnEvent

_INTENT_PATTERNS = (
    ("offer", re.compile(r"\b(how\s+much|offer|emi|kitna|kitni|kitne|entha|enta|amount)\b|कितना|कितनी|ఎంత", re.I)),
    ("documents", re.compile(r"\b(documents?|papers?|kagaz|dastavez|patralu|upload|form\s*16|salary\s+slips?)\b|"
                             r"दस्तावेज़?|कागज़?|పత్రాలు", re.I)),
    ("improve", re.compile(r"\b(what\s+(can|should)\s+i\s+do|how\s+(can|do)\s+i\s+(improve|get)|next\s+time|"
                           r"what\s+helps|kya\s+karna|em\s+cheyali|improve)\b|क्या\s+करना|ఏం\s+చేయాలి", re.I)),
    ("explain", re.compile(r"\b(why|reason|reject\w*|declin\w*|kyun|kyon|kyu|enduku|endhuku)\b|क्यों|ఎందుకు", re.I)),
)


def guess_intent(text: str) -> str | None:
    """A cheap keyword intent for the fast path. None -> the slow path (agent server) decides."""
    for intent, pat in _INTENT_PATTERNS:
        if pat.search(text or ""):
            return intent
    return None


class ToneHook(Protocol):
    async def on_verified(self) -> list[TurnEvent]: ...
    def on_user_final(self, text: str, lang_code: str | None = None) -> T.SpeakingPlan: ...
    def respond(self, plan: T.SpeakingPlan, intent: str | None = None, text: str = "") -> AsyncIterator[TurnEvent]: ...
    def on_barge_in(self, false_interruption: bool = False) -> None: ...
    def on_agent_turn_end(self) -> None: ...
    def on_silence(self, seconds: float) -> T.SpeakingPlan | None: ...
    def on_close(self) -> T.SpeakingPlan: ...
    def tts_options(self, plan: T.SpeakingPlan) -> dict: ...
    def on_call_end(self) -> None: ...


class ToneEngineHook:
    def __init__(self, port: DecisionPort, *, slow_path: SlowPath | None = None, locale: str = "en",
                 literacy: str = "medium", channel: str = "voice", employment_type: str | None = None,
                 product_id: str | None = None, lexicon: Mapping | None = None) -> None:
        self.locale = locale if locale in ("en", "hi", "te") else "en"
        self.literacy = literacy
        self.channel = channel
        self.lexicon = lexicon
        self.slow = SlowThinker(port, employment_type=employment_type, channel=channel, literacy=literacy,
                                product_id=product_id)
        self.fast = FastThinker(self.slow, slow_path)
        self.state = T.ToneState()
        self.base = T.derive(None, self.locale, literacy, channel)   # until the record is known
        self.awaiting_income = False

    # ------------------------------------------------------------------ call start
    async def on_verified(self) -> list[TurnEvent]:
        self.slow.mark_verified()
        await self.slow.prefetch()
        self._rebase()
        loc = self.state.reply_locale or self.locale
        out = [TurnEvent("line", C.line("verified", loc), "template", "verified")]
        if self.base.tier != "plain":       # guided voice asks before reading personal detail aloud (v2 Axis C)
            out.append(TurnEvent("line", C.line("consent_check", loc), "template", "consent_check"))
        return out

    def _rebase(self) -> None:
        self.base = T.derive(self.slow.cache.explain, self.locale, self.literacy, self.channel)

    @property
    def explain(self) -> EP.ExplainProfile | None:
        return self.slow.cache.explain

    # ------------------------------------------------------------------ each turn
    def on_user_final(self, text: str, lang_code: str | None = None) -> T.SpeakingPlan:
        sig = LS.detect(text, lang_code, self.lexicon)
        if self.awaiting_income:
            stated = EP.parse_income_answer(text)
            if stated:
                self.awaiting_income = False
                self.slow.update_slots(stated_income_type=stated)
                self._rebase()
        if sig.human and self.explain is not None and not self.explain.assist == "WANTS_PERSON":
            self.slow.update_slots(stated_prefers_person=True)
            self._rebase()
        return T.plan_user_turn(self.base, self.state, sig, self.literacy)

    async def respond(self, plan: T.SpeakingPlan, intent: str | None = None, text: str = "") -> AsyncIterator[TurnEvent]:
        intent = intent or guess_intent(text)
        async for ev in self.fast.respond(plan, intent, text):
            yield ev
        ex = self.explain
        if ex and ex.ask_income and intent in ("documents", "improve") and not self.awaiting_income \
                and not plan.handover and plan.tone.tier != "distress":
            self.awaiting_income = True     # asked once per call, only for S1 / S5
            yield TurnEvent("line", EP.INCOME_QUESTION.get(plan.tone.reply_locale, EP.INCOME_QUESTION["en"]),
                            "template", "income_question")

    def tts_options(self, plan: T.SpeakingPlan) -> dict:
        return T.tts_params(plan.tone)

    # ------------------------------------------------------------------ LiveKit events
    def on_barge_in(self, false_interruption: bool = False) -> None:
        T.note_barge_in(self.state, false_interruption)

    def on_agent_turn_end(self) -> None:
        T.note_agent_turn_end(self.state)

    def on_silence(self, seconds: float) -> T.SpeakingPlan | None:
        return T.plan_silence(self.base, self.state, seconds, self.literacy)

    def on_close(self) -> T.SpeakingPlan:
        return T.plan_close(self.base, self.state, self.literacy)

    def on_call_end(self) -> None:
        self.slow.wipe()
        self.state = T.ToneState()


class NullToneHook:
    """Default for the worker until the tone engine is wired: plans a neutral turn and says nothing itself."""

    def __init__(self, locale: str = "en") -> None:
        self.base = T.derive(None, locale if locale in ("en", "hi", "te") else "en")

    async def on_verified(self) -> list[TurnEvent]:
        return []

    def on_user_final(self, text: str, lang_code: str | None = None) -> T.SpeakingPlan:
        return T.SpeakingPlan(tone=self.base)

    async def respond(self, plan, intent=None, text=""):
        return
        yield  # pragma: no cover

    def on_barge_in(self, false_interruption: bool = False) -> None:
        pass

    def on_agent_turn_end(self) -> None:
        pass

    def on_silence(self, seconds: float):
        return None

    def on_close(self) -> T.SpeakingPlan:
        return T.SpeakingPlan(tone=self.base)

    def tts_options(self, plan: T.SpeakingPlan) -> dict:
        return T.tts_params(plan.tone)

    def on_call_end(self) -> None:
        pass
