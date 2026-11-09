"""How agent_worker.py uses the tone engine (tone_hook.py), and the rule that it does so only AFTER verification.

    tone = ToneSession(make_hook(locale))
    await tone.activate()             # phone: keypad verified; browser: Frappe's Voice Session row authorised the call
    plan = tone.plan_turn(text)       # None until activated: the auth phase never touches the tone engine
    tts.set_speed(tone.speed(plan))   # Timbre speed from the tier, kept inside the tone band 0.90-1.05
    for line in tone.lead_lines(plan): say(line)      # closed-set acknowledgements / hand-over lines
    budget = tone.budget(plan)
    for sentence in graph_answer: say(sentence) if budget.admit(sentence)   # word + sentence budget of the tier
    for line in tone.closing_lines(plan): say(line)   # the person offer closes the turn

The graph (advisory server) still writes every answer; tone only chooses the speaking speed, which closed-set lines
go around it and how many of its whole sentences are spoken this turn. A sentence over budget is deferred, never cut
or rewritten, and the full answer card still goes to the page. The self-harm safety line is an unapproved placeholder
and is never spoken (tone_cards.line returns None for it).

Decision prefetch is NOT wired: the worker has no ZEN decision-record port yet, so the hook runs with
`UnwiredDecisionPort` (no record), i.e. the segment-free base tone plus the live per-turn rules (confusion, anger,
distress, barge-in, language). No LiveKit import, so this is unit-tested alone.
"""
from __future__ import annotations

from typing import Mapping

from loan_advisory.agent import tone as T
from loan_advisory.agent import tone_cards as C
from tone_hook import ToneEngineHook
from tone_thinker import OFFER_LINES


class UnwiredDecisionPort:
    """DecisionPort with no record: the voice worker cannot reach ZEN's stored decision yet."""

    async def stamp(self) -> str | None:
        return None

    async def fetch(self) -> tuple[Mapping | None, str | None]:
        return None, None


def make_hook(locale: str = "en", literacy: str = "medium") -> ToneEngineHook:
    return ToneEngineHook(UnwiredDecisionPort(), locale=locale, literacy=literacy, channel="voice")


class Budget:
    """Streaming form of tone.fit_budget: whole sentences while they fit the tier's words and sentences; the first
    sentence always; once one is deferred, every later one is too (order is never changed)."""

    def __init__(self, tone: T.ToneProfile):
        self.tone, self.words, self.kept, self.deferred = tone, 0, 0, 0

    def admit(self, sentence: str) -> bool:
        n = T.word_count(sentence)
        if self.kept == 0 or (not self.deferred and self.words + n <= self.tone.max_words
                              and self.kept < self.tone.max_sentences):
            self.kept, self.words = self.kept + 1, self.words + n
            return True
        self.deferred += 1
        return False


class ToneSession:
    """The tone hook for one call, inert until `activate()` (called only once the caller is verified)."""

    def __init__(self, hook):
        self.hook, self.active = hook, False

    async def activate(self) -> list[str]:
        """Start tone for this call (prefetch, base tone). Returns lines to speak after the fixed 'verified' clip."""
        if self.active:
            return []
        self.active = True
        events = await self.hook.on_verified()
        return [e.text for e in events if e.kind == "line" and e.text and e.key != "verified"]

    def plan_turn(self, text: str, lang_code: str | None = None) -> T.SpeakingPlan | None:
        return self.hook.on_user_final(text, lang_code) if self.active else None

    def speed(self, plan: T.SpeakingPlan | None) -> float | None:
        if plan is None:
            return None
        return min(T.SPEED_CEIL, max(T.SPEED_FLOOR, float(self.hook.tts_options(plan)["speed"])))

    @staticmethod
    def _lines(plan: T.SpeakingPlan | None, offers: bool) -> list[str]:
        if plan is None:
            return []
        keys = [k for k in plan.lines if (k in OFFER_LINES) == offers]
        return [s for s in (C.line(k, plan.tone.reply_locale) for k in keys) if s]

    def lead_lines(self, plan: T.SpeakingPlan | None) -> list[str]:
        return self._lines(plan, offers=False)

    def closing_lines(self, plan: T.SpeakingPlan | None) -> list[str]:
        return self._lines(plan, offers=True)

    @staticmethod
    def budget(plan: T.SpeakingPlan | None) -> Budget | None:
        return Budget(plan.tone) if plan is not None else None

    @staticmethod
    def speak_answer(plan: T.SpeakingPlan | None) -> bool:
        """False on a turn where tone says nothing about the decision (safety / urgent hand-over)."""
        return plan is None or not plan.no_explain

    def on_barge_in(self) -> None:
        if self.active:
            self.hook.on_barge_in()

    def on_agent_turn_end(self) -> None:
        if self.active:
            self.hook.on_agent_turn_end()

    def end(self) -> None:
        if self.active:
            self.hook.on_call_end()
            self.active = False


__all__ = ["Budget", "ToneSession", "UnwiredDecisionPort", "make_hook"]
