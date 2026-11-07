"""Fast/slow thinker for the Saral voice agent (VoiceAgentRAG-style; research/tone-segmentation-v3.md §5).

SLOW THINKER (after keypad/OTP verification, never before): deterministically prefetches the applicant's decision
record and its version stamp from ZEN (through the agent server), resolves the explain profile, builds the fact card
(reasons, document checklist, offer and stored KFS facts as exact ₹ strings) and the lead sentences for the likely
intents, per locale and tier, and refreshes them whenever the stamp changes. No LLM.

FAST THINKER (every turn): answers decision intents from the cache with deterministic templates (no LLM). Before
every decision turn it probes ZEN's version stamp; a stale stamp drops the cache and refetches before anything is
said. On a cache miss (an intent it has no template for) it hands the turn to the slow path (the agent server's
compose call) and plays a filler line if nothing is ready by 700 ms.

Rules that hold everywhere:
- Decision facts always come from ZEN's record with a fresh stamp. The cache holds them only for the call.
- Tone never changes a decision or a fact: leads are built from the card, and a tier only chooses how many of the
  card's sentences go in this turn (the rest are deferred, never rewritten).
- Slow-path text is stripped of tags and dropped if it trips the never-say patterns (thresholds, "you were close").

No LiveKit import: the worker feeds events in and plays what comes out.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Mapping, Protocol

from loan_advisory.agent import explain_profile as EP
from loan_advisory.agent import tone as T
from loan_advisory.agent import tone_cards as C

FILLER_DEADLINE_S = 0.7
DECISION_INTENTS = frozenset({"explain", "offer", "improve", "documents"})
OFFER_LINES = ("offer_person", "confusion_offer_person")
LOCALES = ("en", "hi", "te")


class NotVerified(RuntimeError):
    """Decision data was requested before the caller was verified."""


class DecisionPort(Protocol):
    """How the voice worker reaches ZEN's stored decision (through the agent server; mocked in tests)."""

    async def stamp(self) -> str | None:
        """The current version stamp of the applicant's latest decision record (cheap)."""

    async def fetch(self) -> tuple[Mapping | None, str | None]:
        """(record, stamp) of the applicant's latest decision record."""


SlowPath = Callable[[str, str, T.SpeakingPlan], AsyncIterator[str]]   # (intent or "", text, plan) -> sentences


@dataclass(frozen=True)
class TurnEvent:
    kind: str            # line | filler | sentence | check | handover | safety_flag | needs_auth | stale_refetch
    text: str | None = None
    source: str = ""     # template | cache | slow | ""
    key: str | None = None


@dataclass
class CallCache:
    stamp: str | None = None
    record: Mapping | None = None              # call-scoped only; wiped at call end
    explain: EP.ExplainProfile | None = None
    cards: dict[str, dict] = field(default_factory=dict)            # locale -> fact card
    leads: dict[tuple, list[str]] = field(default_factory=dict)     # (stamp, locale, tier, intent) -> sentences
    fetched_at: float = 0.0
    fetches: int = 0

    def clear(self) -> None:
        self.stamp, self.record, self.explain = None, None, None
        self.cards.clear()
        self.leads.clear()


class SlowThinker:
    def __init__(self, port: DecisionPort, *, employment_type: str | None = None, channel: str = "voice",
                 literacy: str = "medium", product_id: str | None = None, locales=LOCALES,
                 slots: Mapping | None = None) -> None:
        self.port = port
        self.kw = {"employment_type": employment_type, "channel": channel, "literacy": literacy,
                   "product_id": product_id}
        self.locales = tuple(locales)
        self.slots = dict(slots or {})
        self.cache = CallCache()
        self.verified = False
        self._lock = asyncio.Lock()

    def mark_verified(self) -> None:
        self.verified = True

    async def prefetch(self) -> CallCache:
        """Call start, after verification: record + stamp, explain profile, cards and leads for every locale/tier."""
        if not self.verified:
            raise NotVerified("prefetch before verification")
        async with self._lock:
            record, stamp = await self.port.fetch()
            self._build(record, stamp)
        return self.cache

    async def refresh_if_stale(self) -> bool:
        """Probe the stamp; on change drop everything and refetch. True when a refetch happened."""
        if not self.verified:
            raise NotVerified("refresh before verification")
        current = await self.port.stamp()
        if current is not None and current == self.cache.stamp and self.cache.record is not None:
            return False
        await self.prefetch()
        return True

    def update_slots(self, **slots) -> None:
        """A caller-stated slot changed (income answer, prefers a person): recompute profile and leads, same record."""
        self.slots.update(slots)
        if self.cache.record is not None:
            self._build(self.cache.record, self.cache.stamp)

    def _build(self, record: Mapping | None, stamp: str | None) -> None:
        self.cache.clear()
        self.cache.record, self.cache.stamp = record, stamp
        self.cache.fetched_at, self.cache.fetches = time.monotonic(), self.cache.fetches + 1
        if record is None:
            return
        self.cache.explain = EP.resolve(record, slots=self.slots, **self.kw)
        for loc in self.locales:
            card = C.fact_card(record, self.cache.explain, loc)
            self.cache.cards[loc] = card
            for tier in T.TIERS:
                for intent in C.INTENTS:
                    self.cache.leads[(stamp, loc, tier, intent)] = C.lead(intent, card, tier, loc)

    def lead(self, intent: str, locale: str, tier: str, simplify: bool = False) -> list[str] | None:
        if self.cache.record is None:
            return None
        if simplify:
            card = self.cache.cards.get(locale)
            return C.lead(intent, card, tier, locale, simplify=True) if card else None
        return self.cache.leads.get((self.cache.stamp, locale, tier, intent))

    def wipe(self) -> None:
        self.cache.clear()
        self.verified = False


class FastThinker:
    def __init__(self, slow: SlowThinker, slow_path: SlowPath | None = None,
                 filler_deadline_s: float = FILLER_DEADLINE_S) -> None:
        self.slow = slow
        self.slow_path = slow_path
        self.filler_deadline_s = filler_deadline_s
        self.deferred: list[str] = []          # sentences a short tier deferred (next turn or written summary)
        self.last_intent: str | None = None

    async def respond(self, plan: T.SpeakingPlan, intent: str | None = None, text: str = "") -> AsyncIterator[TurnEvent]:
        tone = plan.tone
        loc = tone.reply_locale
        if plan.safety_flag:
            yield TurnEvent("safety_flag", key="safety")
        tags_used = 0
        offers = [k for k in plan.lines if k in OFFER_LINES]
        acked = any(k.startswith("ack_") for k in plan.lines)
        for key in [k for k in plan.lines if k not in OFFER_LINES]:
            said = C.line(key, loc)
            if said is None:
                continue
            if key.startswith("ack_") and key != "ack_confusion" and tone.tier == "distress" and tags_used == 0:
                said = "<slow> " + said          # the only <slow>: on the acknowledgement, scoped to that sentence
                tags_used = 1
            yield TurnEvent("line", said, "template", key)
        if plan.handover:
            yield TurnEvent("handover", key="handover")
        if not plan.no_explain:
            if plan.simplify and intent is None:
                intent = self.last_intent
            if intent is None and plan.tone.tier == "distress" and self.slow.cache.record is not None:
                intent = "explain"                      # distress: the one fact, then the person offer
            async for ev in self._content(plan, intent, text, tags_used, acked):
                yield ev
        for key in offers:                              # the person offer closes the turn
            yield TurnEvent("line", C.line(key, loc), "template", key)

    async def _content(self, plan: T.SpeakingPlan, intent: str | None, text: str, tags_used: int, acked: bool):
        tone = plan.tone
        if plan.handover and intent not in DECISION_INTENTS:
            return
        if intent is None and not plan.llm_allowed:
            return
        if intent in DECISION_INTENTS:
            if not self.slow.verified:
                yield TurnEvent("needs_auth")
                return
            filled = False
            task = asyncio.ensure_future(self.slow.refresh_if_stale())
            try:
                refetched = await asyncio.wait_for(asyncio.shield(task), self.filler_deadline_s)
            except asyncio.TimeoutError:
                filled = True
                if not acked:                           # never a filler after an acknowledgement
                    yield self._filler(tone)
                refetched = await task
            if refetched:
                yield TurnEvent("stale_refetch", key=self.slow.cache.stamp)
            sentences = self.slow.lead(intent, tone.reply_locale, tone.tier, simplify=plan.simplify)
            if sentences:
                self.last_intent = intent
                async for ev in self._speak(sentences, plan, "cache", tags_used=tags_used):
                    yield ev
                return
            if not plan.llm_allowed:
                return
            async for ev in self._slow(intent, text, plan, filled or acked):
                yield ev
            return
        if plan.llm_allowed:
            async for ev in self._slow(intent or "", text, plan, acked):
                yield ev

    def _filler(self, tone: T.ToneProfile) -> TurnEvent:
        key = C.filler_key(tone.tier, tone.tier == "distress")
        return TurnEvent("filler", C.line(key, tone.reply_locale), "template", key)

    async def _speak(self, sentences: list[str], plan: T.SpeakingPlan, source: str, tags_used: int = 0):
        check = C.check_line(plan.tone.reply_locale) if plan.end_with_check else None
        body = list(sentences) + ([check] if check else [])
        kept, deferred = T.fit_budget(body, plan.tone, checks=1 if check else 0)
        self.deferred = [s for s in deferred if s != check]
        rendered = T.render_for_speech(kept, plan.tone, tags_used=tags_used)
        for raw, s in zip(kept, rendered):
            yield TurnEvent("check" if check is not None and raw == check else "sentence", s, source)

    async def _slow(self, intent: str, text: str, plan: T.SpeakingPlan, filled: bool):
        if self.slow_path is None:
            if not filled:
                yield self._filler(plan.tone)
            return
        agen = self.slow_path(intent, text, plan).__aiter__()
        first = asyncio.ensure_future(agen.__anext__())
        try:
            sentence = await asyncio.wait_for(asyncio.shield(first), self.filler_deadline_s)
        except asyncio.TimeoutError:
            if not filled:
                yield self._filler(plan.tone)
            try:
                sentence = await first
            except StopAsyncIteration:
                return
        except StopAsyncIteration:
            return
        budget_words = 0
        segment = plan.tone.segment
        while True:
            clean = T.strip_tags(sentence)
            if clean and not C.never_say_hits(clean, segment):
                budget_words += T.word_count(clean)
                if budget_words > plan.tone.max_words:
                    self.deferred.append(clean)
                else:
                    for s in T.render_for_speech([clean], plan.tone):
                        yield TurnEvent("sentence", s, "slow")
            try:
                sentence = await agen.__anext__()
            except StopAsyncIteration:
                break


async def collect(gen: AsyncIterator[TurnEvent]) -> list[TurnEvent]:
    return [ev async for ev in gen]
