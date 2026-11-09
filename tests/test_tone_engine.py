"""tone: base derivation, live overrides in their fixed order, speech rendering, budgets and exact numbers."""
import itertools
import re
from dataclasses import asdict, replace

import pytest

from loan_advisory.agent import explain_profile as EP
from loan_advisory.agent import tone as T
from loan_advisory.agent.live_signals import Signals, detect


def prof(cats, **kw):
    offer = kw.pop("offer", None)
    outcome = "COUNTER_OFFER" if offer else "DECLINE"
    r = {"outcome": outcome, "reason_categories": cats, "offer": offer, "rule_evaluations": [],
         "bureau": kw.pop("bureau", {"match_status": "HIT"})}
    return EP.resolve(r, **kw)


S4 = prof(["REPAYMENTS_TOO_HIGH"], employment_type="SALARIED")
S7 = prof(["DETAILS_NOT_VERIFIED"])
S8 = prof([], offer={"amount": 150000, "emi": 5200, "tenure_months": 36, "counter_offer": True}, channel="text")


# --------------------------------------------------------------------------------------------- derive
def test_word_budgets_and_speeds_per_tier():
    assert {k: v[:2] for k, v in T.TIERS.items()} == {"brief": (25, 2), "plain": (45, 3), "guided": (30, 2),
                                                      "distress": (35, 3)}
    for _, _, speed in T.TIERS.values():
        assert 0.9 <= speed <= 1.05


def test_base_per_assist_mode():
    assert T.derive(S4, "hi").tier == "guided" and T.derive(S4, "hi").speed == 0.95
    assert T.derive(S8, "en", channel="text").tier == "plain"
    s7 = T.derive(S7, "en")
    assert s7.handoff == "early_offer" and s7.firmness == "firm_minimal" and s7.max_numbers == 0
    assert T.derive(S8, "en").repeat_key_number and T.derive(S8, "en").max_numbers == 3


def test_distress_profile_overrides_everything():
    d = prof(["CREDIT_HISTORY"], intent="distress", product_id="PL_MEDICAL_EMERGENCY")
    t = T.derive(d, "hi")
    assert (t.tier, t.speed, t.handoff, t.checks, t.warmth) == ("distress", 0.9, "offer_now", "never", "ack_first")


def test_low_literacy_never_below_guided():
    p = prof(["EMPLOYMENT_STABILITY"], channel="text", literacy="low")
    assert T.derive(p, "en", literacy="low").tier == "guided"


def test_income_profile_changes_no_tone_dimension():
    tones = {asdict(T.derive(prof(["EMPLOYMENT_STABILITY"], slots={"stated_income_type": inc}), "en"))["tier"]
             for inc in ("SALARIED", "SELF_EMPLOYED", "GIG_PLATFORM", "FARM")}
    full = {tuple(sorted(asdict(T.derive(prof(["EMPLOYMENT_STABILITY"], slots={"stated_income_type": inc}), "en")).items()))
            for inc in ("SALARIED", "SELF_EMPLOYED", "GIG_PLATFORM", "FARM")}
    assert len(tones) == 1 and len(full) == 1


def test_locale_moves_only_language_fields():
    a, b = asdict(T.derive(S4, "en")), asdict(T.derive(S4, "te"))
    diff = {k for k in a if a[k] != b[k]}
    assert diff <= {"reply_locale", "mix"}


def test_speed_band_enforced():
    t = T.derive(S4, "en")
    with pytest.raises(ValueError):
        replace(t, speed=0.85)
    with pytest.raises(ValueError):
        replace(t, speed=1.2)
    assert T.tts_params(t) == {"speed": 0.95, "language": "en-IN"}
    assert T.tts_params(T.derive(S4, "te"))["language"] == "te-IN"


# --------------------------------------------------------------------------------------------- live overrides
def sig(**kw):
    return Signals(**{"words": 4, **kw})


def test_human_request_hands_over_now_even_in_brief():
    st = T.ToneState(barge_ins=[1, 1, 0])
    p = T.plan_user_turn(T.derive(S4, "en"), st, sig(human=True))
    assert p.handover and p.offer_human and not p.llm_allowed and "handover_ack" in p.lines
    assert p.tone.handoff == "offer_now"


def test_self_harm_sets_safety_flag_and_urgent_human():
    st = T.ToneState()
    p = T.plan_user_turn(T.derive(S4, "en"), st, detect("I want to end my life"))
    assert p.safety_flag and p.urgent_human and p.handover and p.no_explain and not p.llm_allowed
    assert p.lines[0] == "safety" and st.distress and st.self_harm


def test_human_and_self_harm_together_keep_both():
    p = T.plan_user_turn(T.derive(S4, "en"), T.ToneState(), sig(human=True, self_harm=True, distress=True))
    assert p.handover and p.safety_flag and p.urgent_human
    assert p.fired[:2] == ("P1:human_request", "P2:self_harm")


def test_distress_is_sticky_and_empathy_plays_once():
    base, st = T.derive(S4, "hi"), T.ToneState()
    p1 = T.plan_user_turn(base, st, detect("main bahut pareshan hoon"))
    assert p1.lines[0] == "ack_worry" and p1.tone.tier == "distress" and p1.offer_human
    p2 = T.plan_user_turn(base, st, detect("theek hai"))
    assert p2.tone.tier == "distress" and not any(x.startswith("ack_w") for x in p2.lines)
    p3 = T.plan_user_turn(base, st, detect("this is nonsense"))
    assert "ack_frustration" not in p3.lines and "ack_worry" not in p3.lines   # empathy once per call


def test_anger_gets_frustration_ack_and_second_hit_escalates():
    base, st = T.derive(S7, "en"), T.ToneState()
    p1 = T.plan_user_turn(base, st, detect("this is nonsense"))
    assert p1.lines[0] == "ack_frustration" and p1.offer_human
    p2 = T.plan_user_turn(base, st, detect("useless"))
    assert "P3:anger_x2" in p2.fired and p2.offer_human


def test_confusion_ladder_and_third_offers_person():
    base, st = T.derive(S4, "en"), T.ToneState()
    p1 = T.plan_user_turn(base, st, detect("I don't understand"))
    assert p1.simplify and p1.lines[0] == "ack_confusion" and not p1.llm_allowed and p1.tone.speed == 0.9
    p2 = T.plan_user_turn(base, st, detect("samajh nahi aaya"))
    assert st.guided_locked and p2.tone.tier == "guided"
    p3 = T.plan_user_turn(base, st, detect("artham kaledu"))
    assert "confusion_offer_person" in p3.lines and p3.offer_human and "P4:confusion_x3" in p3.fired


def test_understood_twice_resets_confusion():
    base, st = T.derive(S8, "en", channel="text"), T.ToneState()
    T.plan_user_turn(base, st, detect("I don't understand"))
    T.plan_user_turn(base, st, detect("got it"))
    T.plan_user_turn(base, st, detect("got it"))
    assert st.consecutive_confusion == 0


def test_explicit_language_switch_is_same_turn():
    base, st = T.derive(S8, "hi", channel="text"), T.ToneState()
    p = T.plan_user_turn(base, st, detect("Can you tell me in English?", "en-IN"))
    assert p.tone.reply_locale == "en" and p.tts["language"] == "en-IN"


def test_implicit_language_switch_needs_two_utterances_of_three_words():
    base, st = T.derive(S8, "hi", channel="text"), T.ToneState()
    assert T.plan_user_turn(base, st, detect("okay", "en-IN")).tone.reply_locale == "hi"
    assert T.plan_user_turn(base, st, detect("so what is the EMI", "en-IN")).tone.reply_locale == "hi"
    assert T.plan_user_turn(base, st, detect("and for how many months", "en-IN")).tone.reply_locale == "en"


def test_two_barge_ins_in_three_turns_go_brief():
    base, st = T.derive(S8, "hi", channel="text"), T.ToneState()
    for n in (1, 0, 1):
        for _ in range(n):
            T.note_barge_in(st)
        T.note_agent_turn_end(st)
    p = T.plan_user_turn(base, st, detect("haan haan kitna milega"))
    assert p.tone.tier == "brief" and p.tone.speed == 1.05 and not p.end_with_check


def test_false_interruptions_do_not_count():
    st = T.ToneState()
    for _ in range(3):
        T.note_barge_in(st, false_interruption=True)
        T.note_agent_turn_end(st)
    p = T.plan_user_turn(T.derive(S8, "en", channel="text"), st, detect("ok tell me"))
    assert p.tone.tier != "brief"


def test_barge_in_never_overrides_distress_or_low_literacy():
    base, st = T.derive(S4, "en"), T.ToneState(barge_ins=[1, 1, 1], distress=True)
    assert T.plan_user_turn(base, st, detect("ok")).tone.tier == "distress"
    st2 = T.ToneState(barge_ins=[1, 1, 1])
    assert T.plan_user_turn(base, st2, detect("ok"), literacy="low").tone.tier != "brief"


def test_silence_nudges_8_then_15_max_two():
    base, st = T.derive(S4, "en"), T.ToneState()
    assert T.plan_silence(base, st, 5) is None
    p1 = T.plan_silence(base, st, 8)
    assert p1.lines == ("nudge_wait",) and not p1.llm_allowed
    p2 = T.plan_silence(base, st, 15)
    assert p2.lines == ("nudge_away",) and p2.offer_human
    assert T.plan_silence(base, st, 30) is None


def test_silence_in_distress_includes_person():
    st = T.ToneState(distress=True)
    p = T.plan_silence(T.derive(S4, "en"), st, 8)
    assert "offer_person" in p.lines and p.offer_human


def test_handover_floor_at_close():
    base, st = T.derive(S8, "en", channel="text"), T.ToneState()
    assert "offer_person" in T.plan_close(base, st).lines
    st2 = T.ToneState(human_offered=True)
    assert T.plan_close(base, st2).lines == ("close",)


def test_early_offer_for_s7_on_turn_one():
    p = T.plan_user_turn(T.derive(S7, "en"), T.ToneState(), detect("why can't you tell me"))
    assert p.offer_human and "offer_person" in p.lines


def test_ratchet_live_rules_never_speed_up_beyond_base_except_brief():
    base = T.derive(S4, "en")
    for s in (detect("I don't understand"), detect("main pareshan hoon"), detect("artham kaledu")):
        st = T.ToneState()
        assert T.plan_user_turn(base, st, s).tone.speed <= base.speed


def test_tone_version_bumps_every_plan():
    base, st = T.derive(S4, "en"), T.ToneState()
    v = [T.plan_user_turn(base, st, detect("ok")).tone.version for _ in range(3)]
    assert v == sorted(set(v))


# --------------------------------------------------------------------------------------------- rendering
def test_render_strips_llm_tags_and_caps_code_tags():
    tone = T.derive(S4, "en")  # guided, before_numbers
    out = T.render_for_speech(["Your EMIs are high <laugh>.", "We can offer ₹1,50,000.", "EMI ₹5,200.", "Okay ₹3."], tone)
    assert "<laugh>" not in " ".join(out) and T.count_tags(out) <= 2
    assert all(t in T.ALLOWED_TAGS for s in out for t in re.findall(r"<[^>]+>", s))


def test_render_distress_slow_on_ack_only():
    d = T.derive(prof(["CREDIT_HISTORY"], distress=True), "en")
    out = T.render_for_speech(["I can hear this is worrying.", "The area was your credit history.", "Want a person?"],
                              d, ack_first=True)
    assert out[0].startswith("<slow>") and "<slow>" not in " ".join(out[1:]) and T.count_tags(out) == 2


def test_brief_has_no_tags():
    tone = replace(T.derive(S8, "en"), **T._tier_fields("brief"))
    assert T.count_tags(T.render_for_speech(["a.", "₹5 b.", "c."], tone)) == 0


def test_fit_budget_defers_whole_sentences_never_cuts():
    tone = replace(T.derive(S8, "en"), **T._tier_fields("guided"))
    sents = ["We can't offer the full amount, but we can offer ₹1,50,000.",
             "The EMI would be ₹5,200 a month for 36 months.", "This is not a final sanction yet."]
    kept, deferred = T.fit_budget(sents, tone)
    assert kept + deferred == sents and sum(T.word_count(s) for s in kept) <= tone.max_words
    assert len(kept) <= tone.max_sentences


# --------------------------------------------------------------------------------------------- numbers
@pytest.mark.parametrize("v,s", [(150000, "₹1,50,000"), (5200, "₹5,200"), (12345678, "₹1,23,45,678"),
                                 (999, "₹999"), (5200.5, "₹5,200.50"), (100000000, "₹10,00,00,000")])
def test_speak_amount_exact_indian_grouping(v, s):
    assert T.speak_amount(v) == s


def test_speak_amount_refuses_to_round():
    with pytest.raises(ValueError):
        T.speak_amount(5200.555)
    with pytest.raises(TypeError):
        T.speak_amount(True)


@pytest.mark.parametrize("v", [150000, 5200, 12345678, 1, 100000, 10000000, 249999, 36])
@pytest.mark.parametrize("loc", ["en", "hi", "te"])
def test_amount_words_round_trip_exact(v, loc):
    words = T.amount_words(v, loc)
    assert T.amount_value_from_words(words) == v


def test_amount_words_lakh_crore():
    assert T.amount_words(150000) == "1 lakh 50 thousand rupees"
    assert T.amount_words(12345678) == "1 crore 23 lakh 45 thousand 6 hundred 78 rupees"
    assert T.amount_words(150000, "hi") == "1 लाख 50 हज़ार रुपये"


# --------------------------------------------------------------------------------------------- prompt / proxies
def test_prompt_fragment_is_closed_table_and_bans_thresholds():
    f = T.prompt_fragment(T.derive(S4, "en"))
    assert "at most 30 words" in f and "Never round" in f and "tags" in f


def test_tone_inputs_are_not_protected_attributes():
    import inspect
    for fn in (T.derive, T.plan_user_turn, T.plan_silence, T.plan_close):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"name", "gender", "age", "dob", "city", "pincode", "marital_status", "voice_pitch"}


def test_all_tiers_x_locales_valid_speed():
    for tier, loc in itertools.product(T.TIERS, ("en", "hi", "te")):
        t = replace(T.derive(S4, loc), **T._tier_fields(tier))
        assert 0.85 <= T.tts_params(t)["speed"] <= 1.15
