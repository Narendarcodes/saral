"""Tone engine wiring for the voice worker (apps/voice/tone_wiring.py): inert until verification, Timbre speed kept in
the 0.90-1.05 band, closed-set lines around the graph answer, whole-sentence word budgets, safety line never spoken.
No LiveKit, no network."""
import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "voice"))
import tone_wiring as W  # noqa: E402
from loan_advisory.agent import tone as T  # noqa: E402
from loan_advisory.agent import tone_cards as C  # noqa: E402


class SpyHook:
    """Records every call; plans come from a real hook."""

    def __init__(self, locale="en"):
        self.real, self.calls = W.make_hook(locale), []

    def __getattr__(self, name):
        attr = getattr(self.real, name)
        if not callable(attr):
            return attr

        def wrapped(*a, **k):
            self.calls.append(name)
            return attr(*a, **k)
        return wrapped


def test_inert_until_activated():
    spy = SpyHook()
    tone = W.ToneSession(spy)
    assert tone.plan_turn("why was my loan rejected") is None
    assert tone.speed(None) is None and tone.budget(None) is None and tone.lead_lines(None) == []
    tone.on_barge_in(); tone.on_agent_turn_end(); tone.end()
    assert spy.calls == []


def test_activation_prefetches_once_and_never_repeats_the_verified_line():
    spy = SpyHook()
    tone = W.ToneSession(spy)
    lines = asyncio.run(tone.activate())
    assert C.line("verified", "en") not in lines and spy.calls == ["on_verified"]
    assert asyncio.run(tone.activate()) == [] and spy.calls == ["on_verified"]
    assert spy.real.slow.verified is True and spy.real.slow.cache.fetches == 1
    assert spy.real.slow.cache.record is None  # decision prefetch not wired: no record, neutral base tone


def test_speed_stays_in_the_tone_band_and_reacts_to_live_signals():
    tone = W.ToneSession(W.make_hook("en"))
    asyncio.run(tone.activate())
    calm = tone.plan_turn("why was my loan rejected")
    assert T.SPEED_FLOOR <= tone.speed(calm) <= T.SPEED_CEIL
    confused = None
    for _ in range(2):
        confused = tone.plan_turn("I don't understand, what does that mean?")
    assert confused.tone.tier == "guided" and tone.speed(confused) <= tone.speed(calm)
    assert 0.9 <= tone.speed(confused) <= 1.05


def test_budget_matches_fit_budget_on_whole_sentences():
    prof = T.derive(None, "en")
    sentences = ["Your application was declined because the income could not be verified.",
                 "The bank statement for March was missing.",
                 "You can upload it in the portal and ask for a review.",
                 "A person from our team can also call you."]
    b = W.Budget(prof)
    streamed = [s for s in sentences if b.admit(s)]
    kept, deferred = T.fit_budget(sentences, prof)
    assert streamed == kept and b.deferred == len(deferred)
    one = W.Budget(T.derive(None, "en"))
    assert one.admit("word " * 200)  # the first sentence is always spoken, never cut


def test_budget_never_reorders_after_a_deferral():
    prof = T.derive(None, "en")
    b = W.Budget(prof)
    assert b.admit("short one.")
    assert not b.admit("long " * prof.max_words)
    assert not b.admit("tiny.")  # would fit, but speaking it would skip the deferred sentence


def test_safety_line_is_never_spoken_and_answer_is_withheld():
    tone = W.ToneSession(W.make_hook("en"))
    asyncio.run(tone.activate())
    plan = tone.plan_turn("I want to kill myself")
    assert plan.safety_flag and not tone.speak_answer(plan)
    spoken = tone.lead_lines(plan) + tone.closing_lines(plan)
    assert all("TODO" not in s for s in spoken)
    assert C.line("safety", "en") is None


def test_person_offer_closes_the_turn():
    tone = W.ToneSession(W.make_hook("en"))
    asyncio.run(tone.activate())
    plan = None
    for _ in range(3):
        plan = tone.plan_turn("I don't understand anything you are saying")
    offers = tone.closing_lines(plan)
    assert offers and all(s not in tone.lead_lines(plan) for s in offers)


@pytest.mark.parametrize("locale", ["en", "hi", "te"])
def test_lines_follow_the_reply_locale(locale):
    tone = W.ToneSession(W.make_hook(locale))
    asyncio.run(tone.activate())
    plan = tone.plan_turn("why")
    assert plan.tone.reply_locale == locale
