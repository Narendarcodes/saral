"""Fast/slow thinker, tone cards and the worker hook, with a mocked ZEN port (synthetic records, no network)."""
import asyncio
import itertools
import re
import sys
from pathlib import Path

import pytest

from loan_advisory.agent import explain_profile as EP
from loan_advisory.agent import tone as T
from loan_advisory.agent import tone_cards as C

VOICE = Path(__file__).resolve().parents[1] / "apps" / "voice"
sys.path.insert(0, str(VOICE))

import tone_hook as H  # noqa: E402
import tone_thinker as TT  # noqa: E402

COUNTER = {"amount": 150000, "emi": 5200, "tenure_months": 36, "counter_offer": True}


def record(cats, offer=None, outcome=None, bureau=None, **extra):
    return {"outcome": outcome or ("COUNTER_OFFER" if offer else "DECLINE"), "reason_categories": cats,
            "offer": offer, "rule_evaluations": [], "bureau": bureau or {"match_status": "HIT"},
            "application_ref": "APP-SYN-9", **extra}


DEMO = {
    "S4": record(["REPAYMENTS_TOO_HIGH"]),
    "S1": record(["CREDIT_HISTORY"], bureau={"match_status": "NO_HIT"}),
    "S5": record(["INCOME_NOT_VERIFIED_OR_INSUFFICIENT"]),
    "S7": record(["DETAILS_NOT_VERIFIED"]),
    "S3": record(["CREDIT_HISTORY"]),
    "S6+S8": record(["REQUEST_OUTSIDE_OFFER"], offer=COUNTER),
}


class FakePort:
    def __init__(self, rec, stamp="s1", delay=0.0):
        self.rec, self._stamp, self.delay = rec, stamp, delay
        self.fetches = self.stamps = 0

    async def stamp(self):
        self.stamps += 1
        await asyncio.sleep(self.delay)
        return self._stamp

    async def fetch(self):
        self.fetches += 1
        await asyncio.sleep(self.delay)
        return self.rec, self._stamp


def run(coro):
    return asyncio.run(coro)


async def events(gen):
    return [e async for e in gen]


def texts(evs, kinds=("line", "sentence", "check", "filler")):
    return [e.text for e in evs if e.kind in kinds]


def make_hook(rec, locale="en", **kw):
    port = FakePort(rec, delay=kw.pop("delay", 0.0))
    return H.ToneEngineHook(port, locale=locale, **kw), port


# --------------------------------------------------------------------------------------------- slow thinker
def test_no_decision_fetch_before_verification():
    port = FakePort(DEMO["S4"])
    slow = TT.SlowThinker(port)
    with pytest.raises(TT.NotVerified):
        run(slow.prefetch())
    assert port.fetches == 0
    fast = TT.FastThinker(slow)
    plan = T.SpeakingPlan(tone=T.derive(None, "en"))
    evs = run(events(fast.respond(plan, "explain")))
    assert [e.kind for e in evs] == ["needs_auth"] and port.fetches == 0


def test_prefetch_builds_cards_and_leads_for_every_locale_tier_intent():
    port = FakePort(DEMO["S6+S8"])
    slow = TT.SlowThinker(port)
    slow.mark_verified()
    cache = run(slow.prefetch())
    assert cache.stamp == "s1" and cache.explain.primary == "S6" and set(cache.cards) == {"en", "hi", "te"}
    assert len(cache.leads) == 3 * len(T.TIERS) * len(C.INTENTS)
    assert cache.cards["en"]["offer"]["amount"] == "₹1,50,000"


def test_cache_hit_uses_templates_and_checks_stamp_each_decision_turn():
    port = FakePort(DEMO["S4"])
    slow = TT.SlowThinker(port)
    slow.mark_verified()
    run(slow.prefetch())
    fast = TT.FastThinker(slow, slow_path=None)
    plan = T.SpeakingPlan(tone=T.derive(slow.cache.explain, "en"))
    evs = run(events(fast.respond(plan, "explain")))
    assert evs and all(e.source in ("cache", "template") for e in evs if e.kind == "sentence")
    assert port.stamps == 1 and port.fetches == 1
    run(events(fast.respond(plan, "offer")))
    assert port.stamps == 2 and port.fetches == 1


def test_stale_stamp_refetches_before_speaking():
    port = FakePort(DEMO["S4"])
    slow = TT.SlowThinker(port)
    slow.mark_verified()
    run(slow.prefetch())
    port.rec, port._stamp = DEMO["S6+S8"], "s2"          # a new evaluation landed mid-call
    fast = TT.FastThinker(slow)
    plan = T.SpeakingPlan(tone=T.derive(slow.cache.explain, "en"))
    evs = run(events(fast.respond(plan, "offer")))
    assert evs[0].kind == "stale_refetch" and port.fetches == 2 and slow.cache.stamp == "s2"
    assert any("₹1,50,000" in t for t in texts(evs))


def test_slow_refetch_plays_filler_by_deadline():
    port = FakePort(DEMO["S4"], delay=0.0)
    slow = TT.SlowThinker(port)
    slow.mark_verified()
    run(slow.prefetch())
    port._stamp, port.delay = "s9", 0.15
    fast = TT.FastThinker(slow, filler_deadline_s=0.05)
    plan = T.SpeakingPlan(tone=T.derive(slow.cache.explain, "en"))
    evs = run(events(fast.respond(plan, "explain")))
    assert evs[0].kind == "filler" and evs[1].kind == "stale_refetch"


def test_cache_miss_filler_by_700ms_then_slow_path():
    port = FakePort(DEMO["S4"])
    slow = TT.SlowThinker(port)
    slow.mark_verified()
    run(slow.prefetch())

    async def slow_path(intent, text, plan):
        await asyncio.sleep(0.12)
        yield "A person can explain the processing fee <laugh> from the policy."
        yield "You were almost there."                       # must be dropped (never-say)

    fast = TT.FastThinker(slow, slow_path, filler_deadline_s=0.05)
    plan = T.SpeakingPlan(tone=T.derive(slow.cache.explain, "en"))

    async def timed():
        t0 = asyncio.get_running_loop().time()
        out = []
        async for e in fast.respond(plan, None, "what is the processing fee?"):
            out.append((e, asyncio.get_running_loop().time() - t0))
        return out

    out = run(timed())
    assert out[0][0].kind == "filler" and out[0][1] < 0.1
    spoken = [e.text for e, _ in out if e.kind == "sentence"]
    assert spoken == ["A person can explain the processing fee from the policy."]


def test_real_deadline_constant_is_700ms():
    assert TT.FILLER_DEADLINE_S == 0.7


def test_fast_slow_path_without_filler_when_quick():
    port = FakePort(DEMO["S4"])
    slow = TT.SlowThinker(port)
    slow.mark_verified()
    run(slow.prefetch())

    async def slow_path(intent, text, plan):
        yield "Quick answer."

    fast = TT.FastThinker(slow, slow_path)
    evs = run(events(fast.respond(T.SpeakingPlan(tone=T.derive(slow.cache.explain, "en")), None, "hmm")))
    assert [e.kind for e in evs] == ["sentence"]


# --------------------------------------------------------------------------------------------- invariance
NUM = re.compile(r"₹[\d,]*\d(?:\.\d\d)?|\b\d+\b")


@pytest.mark.parametrize("name", list(DEMO))
def test_tone_never_changes_facts(name):
    rec = DEMO[name]
    ex = EP.resolve(rec)
    for loc in ("en", "hi", "te"):
        card = C.fact_card(rec, ex, loc)
        allowed = C.card_numbers(card)
        for tier, intent in itertools.product(T.TIERS, C.INTENTS):
            for s in C.lead(intent, card, tier, loc):
                for n in NUM.findall(s):
                    assert n in allowed, (name, loc, tier, intent, s)
        # the card is identical whatever the tier: tone chooses sentences, never facts
        assert card == C.fact_card(rec, ex, loc)
    cats = {tuple(C.fact_card(rec, ex, loc)["categories"]) for loc in ("en", "hi", "te")}
    assert len(cats) == 1


def test_offer_numbers_present_in_every_tier_offer_lead():
    rec = DEMO["S6+S8"]
    card = C.fact_card(rec, EP.resolve(rec), "en")
    for tier in T.TIERS:
        joined = " ".join(C.lead("offer", card, tier, "en"))
        assert "₹1,50,000" in joined and "₹5,200" in joined and "36" in joined


@pytest.mark.parametrize("name", list(DEMO))
def test_no_never_say_hits_in_any_lead_or_line(name):
    rec = DEMO[name]
    ex = EP.resolve(rec)
    for loc in ("en", "hi", "te"):
        card = C.fact_card(rec, ex, loc)
        for tier, intent in itertools.product(T.TIERS, C.INTENTS):
            for s in C.lead(intent, card, tier, loc):
                assert not C.never_say_hits(s, ex.primary), (name, loc, tier, intent, s)
    for key in C.LINES:
        for loc in ("en", "hi", "te"):
            assert not C.never_say_hits(C.line(key, loc))


def test_s7_generic_message_identical_across_tiers():
    card = C.fact_card(DEMO["S7"], EP.resolve(DEMO["S7"]), "en")
    for tier in T.TIERS:
        assert C.S7_GENERIC["en"] in C.lead("explain", card, tier, "en")


def test_closed_set_lines_are_short_and_complete():
    for key, row in C.LINES.items():
        assert set(row) == {"en", "hi", "te"}
        assert len(row["en"].split()) <= 14, key


def test_safety_line_is_unapproved_placeholder_never_spoken():
    assert C.SAFETY_LINE_APPROVED is False and C.line("safety", "en") is None


def test_kfs_facts_pass_through_only_when_stored():
    rec = record(["REQUEST_OUTSIDE_OFFER"], offer=COUNTER, kfs={"apr": "14.2% (synthetic)", "invented": 1})
    card = C.fact_card(rec, EP.resolve(rec), "en")
    assert card["kfs"] == {"apr": "14.2% (synthetic)"}
    assert C.fact_card(DEMO["S4"], EP.resolve(DEMO["S4"]), "en")["kfs"] == {}


# --------------------------------------------------------------------------------------------- hook (demo scripts)
def test_hook_opening_requires_verification_and_asks_consent():
    hook, port = make_hook(DEMO["S4"], "hi")
    evs = run(hook.on_verified())
    assert port.fetches == 1 and [e.key for e in evs] == ["verified", "consent_check"]
    assert hook.base.tier == "guided"


def test_demo1_barge_ins_go_brief_same_numbers():
    hook, _ = make_hook(DEMO["S6+S8"], "en", channel="voice")
    run(hook.on_verified())
    p1 = hook.on_user_final("why was my loan rejected", "en-IN")
    first = texts(run(events(hook.respond(p1, text="why was my loan rejected"))))
    for _ in range(2):
        hook.on_barge_in()
        hook.on_agent_turn_end()
    p2 = hook.on_user_final("haan haan kitna milega", "hi-IN")
    assert p2.tone.tier == "brief" and hook.tts_options(p2)["speed"] == 1.05
    brief = texts(run(events(hook.respond(p2, text="kitna milega"))))
    assert any("₹1,50,000" in s and "₹5,200" in s and "36" in s for s in brief)
    assert sum(len(s.split()) for s in brief) <= 25
    assert first


def test_demo2_confusion_simplifies_without_llm_then_income_question():
    called = []

    async def slow_path(intent, text, plan):
        called.append(intent)
        yield "x."

    hook, _ = make_hook(DEMO["S1"], "te", slow_path=slow_path)
    run(hook.on_verified())
    p = hook.on_user_final("naa loan enduku reject ayyindi", "te-IN")
    run(events(hook.respond(p, text="naa loan enduku reject ayyindi")))
    p2 = hook.on_user_final("artham kaledu", "te-IN")
    evs = run(events(hook.respond(p2, text="artham kaledu")))
    assert evs[0].key == "ack_confusion" and p2.tone.speed == 0.9 and not called
    p3 = hook.on_user_final("what can I do", "en-IN")
    evs3 = run(events(hook.respond(p3, text="what can I do")))
    assert evs3[-1].key == "income_question"
    hook.on_user_final("Swiggy delivery", "en-IN")
    assert hook.explain.income == "GIG_PLATFORM"
    assert "పేఅవుట్" in " ".join(hook.slow.cache.cards["te"]["checklist"])


def test_demo4_anger_s7_firm_generic_and_person_turn_one():
    hook, _ = make_hook(DEMO["S7"], "en")
    run(hook.on_verified())
    p = hook.on_user_final("Why can't you tell me? Are you calling me a fraud? This is nonsense.", "en-IN")
    evs = run(events(hook.respond(p, text="why")))
    said = " ".join(texts(evs))
    assert "frustrating" in said and C.S7_GENERIC["en"] in said and p.offer_human
    assert not C.never_say_hits(said.replace("<slow>", ""), "S7")


def test_demo5_distress_ack_first_slow_tag_person_same_turn_no_credit_types():
    hook, _ = make_hook(DEMO["S3"], "hi", product_id="PL_MEDICAL_EMERGENCY")
    run(hook.on_verified())
    p = hook.on_user_final("Papa ki surgery hai, main bahut pareshan hoon, kya karun?", "hi-IN")
    evs = run(events(hook.respond(p, text="koi aur loan?")))
    assert evs[0].kind == "line" and evs[0].text.startswith("<slow>") and evs[0].key == "ack_worry"
    assert p.tone.speed == 0.9 and p.offer_human and "offer_person" in p.lines
    tags = sum(len(re.findall(r"<[^>]+>", e.text or "")) for e in evs)
    assert tags <= 2
    assert not re.search(r"gold|credit card|सोना", " ".join(texts(evs)), re.I)


def test_demo6_language_switch_same_voice_same_numbers():
    hook, _ = make_hook(DEMO["S6+S8"], "hi", channel="text")
    run(hook.on_verified())
    p1 = hook.on_user_final("offer kitna hai", "hi-IN")
    hi = texts(run(events(hook.respond(p1, text="offer kitna hai"))))
    p2 = hook.on_user_final("Can you tell me in English?", "en-IN")
    en = texts(run(events(hook.respond(p2, intent="offer"))))
    assert hook.tts_options(p2)["language"] == "en-IN"
    nums = lambda xs: set(NUM.findall(" ".join(xs)))  # noqa: E731
    assert {"₹1,50,000", "₹5,200", "36"} <= nums(hi) and {"₹1,50,000", "₹5,200", "36"} <= nums(en)


def test_human_request_hands_over_same_turn():
    hook, _ = make_hook(DEMO["S4"], "en")
    run(hook.on_verified())
    p = hook.on_user_final("Can I talk to a person?", "en-IN")
    evs = run(events(hook.respond(p, text="Can I talk to a person?")))
    assert "handover" in [e.kind for e in evs] and p.handover


def test_self_harm_flags_and_never_speaks_placeholder():
    hook, _ = make_hook(DEMO["S4"], "en")
    run(hook.on_verified())
    p = hook.on_user_final("I want to end my life", "en-IN")
    evs = run(events(hook.respond(p, text="I want to end my life")))
    kinds = [e.kind for e in evs]
    assert "safety_flag" in kinds and "handover" in kinds and p.urgent_human
    assert not any("TODO" in (e.text or "") for e in evs)
    assert not any(e.kind == "sentence" for e in evs)      # no explanation this turn


def test_silence_and_close_through_hook():
    hook, _ = make_hook(DEMO["S6+S8"], "en", channel="text")
    run(hook.on_verified())
    assert hook.on_silence(8).lines == ("nudge_wait",)
    assert hook.on_silence(15).offer_human
    assert hook.on_close().lines == ("close",)            # person already offered at the 15 s nudge
    hook.on_call_end()
    assert hook.slow.cache.record is None and not hook.slow.verified


def test_null_hook_is_inert():
    h = H.NullToneHook("hi")
    assert run(h.on_verified()) == [] and run(events(h.respond(h.on_user_final("x")))) == []
    assert h.tts_options(h.on_user_final("x"))["language"] == "hi-IN"


@pytest.mark.parametrize("text,intent", [("why was it rejected", "explain"), ("kitna milega", "offer"),
                                         ("what documents do I need", "documents"), ("what can I do now", "improve"),
                                         ("tell me about fees", None)])
def test_guess_intent(text, intent):
    assert H.guess_intent(text) == intent
