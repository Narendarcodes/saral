"""Saral tone engine: ToneProfile -> per-turn speaking plan (research/tone-segmentation-v3.md §2, §4).

Tone decides HOW the voice agent speaks: how many words, how fast, where a pause goes, whether an
acknowledgement plays first, when a person is offered. It never decides WHAT is true. Every fact (outcome, reason
category, offer amount, EMI, loan period) comes from the ZEN record through facts.py, unchanged in every tone; a tone
may only DEFER a fact to the next turn or the written summary (see `fit_budget`), never alter, round or add one.

Pieces:
- `derive()`        ExplainProfile (+ locale, literacy) -> base ToneProfile, at call start.
- `ToneState`       live per-call counters owned by the voice worker.
- `plan_user_turn()` live overrides on each final transcript, in this fixed order:
      1 human request      -> hand over now
      2 self-harm cue      -> safety flag + urgent human (the safety wording is an unapproved TODO placeholder)
      3 distress / anger   -> distress override (sticky), acknowledgement at most once per call, person offered
      4 confused 3x        -> offer a person (confused 1-2x -> simpler, slower, no LLM)
      5 language switch    -> explicit request now, implicit after 2 consecutive utterances
      6 2 barge-ins in 3   -> brief
  `plan_silence()`  8 s -> "take your time"; 15 s -> summary-or-person offer (max 2 nudges per call).
  `plan_close()`    the hand-over floor: a person is offered at least once in every call.
- `render_for_speech()` strips any tag from LLM text and inserts at most 2 code-inserted tags (`<pauses>`, and
  `<slow>` only on a distress acknowledgement).
- `speak_amount()` / `amount_words()` exact Indian-grouped rupees and lakh/crore words, never rounded.

Ratchet: live rules only add care (slower, simpler, earlier human) or shorten on barge-ins; nothing a caller says can
clear distress, remove the human offer, or change a fact. Inputs never include name, gender, age, DOB, area or
pincode, marital status, ZEN threshold groups or voice features. Language moves only `reply_locale` / `mix` / the
Timbre language.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Literal

from loan_advisory.agent.explain_profile import ExplainProfile
from loan_advisory.agent.live_signals import Signals
from loan_advisory.agent.settings import LOCALE_TTS, TTS_SPEED_MAX, TTS_SPEED_MIN, check_tts_speed

Tier = Literal["brief", "plain", "guided", "distress"]

# tier -> (max words, max sentences, Timbre speed). Check-ins (guided) are allowed on top of max_sentences.
TIERS: dict[str, tuple[int, int, float]] = {
    "brief": (25, 2, 1.05),
    "plain": (45, 3, 1.0),
    "guided": (30, 2, 0.95),
    "distress": (35, 3, 0.9),
}
SPEED_FLOOR, SPEED_CEIL = 0.9, 1.05          # tone engine's own band, inside Timbre's 0.85-1.15
assert TTS_SPEED_MIN <= SPEED_FLOOR <= SPEED_CEIL <= TTS_SPEED_MAX
MAX_CODE_TAGS = 2                             # Gnani: "two or three tags per paragraph"; we use at most 2
ALLOWED_TAGS = ("<pauses>", "<slow>")
SILENCE_NUDGE_S, SILENCE_AWAY_S = 8.0, 15.0
MAX_SILENCE_NUDGES = 2
CONFUSION_HANDOVER = 3
BARGE_WINDOW, BARGE_BRIEF = 3, 2
LANG_STREAK, LANG_MIN_WORDS = 2, 3


@dataclass(frozen=True)
class ToneProfile:
    tier: Tier
    warmth: Literal["neutral", "warm", "ack_first"]
    directness: Literal["answer_first", "ack_then_answer"]
    max_words: int
    max_sentences: int
    speed: float
    pauses: Literal["none", "between_ideas", "before_numbers"]
    jargon: Literal["none", "glossed"]
    max_numbers: int
    repeat_key_number: bool
    checks: Literal["every_turn", "every_2nd", "end_only", "never"]
    empathy: Literal["none", "light", "ack_once"]
    firmness: Literal["gentle_firm", "firm_minimal"]
    handoff: Literal["end_offer", "early_offer", "offer_now"]
    reply_locale: Literal["en", "hi", "te"]
    mix: Literal["native", "light_mix", "mirror_mix"]
    ack_kind: Literal["none", "worry", "frustration", "confusion"] = "none"
    segment: str | None = None
    version: int = 0

    def __post_init__(self):
        check_tts_speed(self.speed)
        if not (SPEED_FLOOR <= self.speed <= SPEED_CEIL):
            raise ValueError(f"tone speed {self.speed} outside {SPEED_FLOOR}-{SPEED_CEIL}")


@dataclass
class ToneState:
    """Live counters for one call. Owned by the voice worker; reset only by a new call."""
    consecutive_confusion: int = 0
    guided_locked: bool = False
    barge_ins: list[int] = field(default_factory=list)   # one entry per agent turn (excl. false interruptions)
    current_barge: int = 0
    anger_hits: int = 0
    distress: bool = False                                # sticky
    ack_kind: str = "none"
    self_harm: bool = False                               # sticky
    empathy_used: bool = False                            # one acknowledgement per call
    human_offered: bool = False
    handed_over: bool = False
    reply_locale: str | None = None                       # set from the base tone on the first turn
    lang_streak: tuple[str, int] = ("", 0)
    latin_ratio_ema: float = 0.0
    silence_nudges: int = 0
    understood_streak: int = 0
    turns: int = 0
    version: int = 0


@dataclass(frozen=True)
class SpeakingPlan:
    tone: ToneProfile
    lines: tuple[str, ...] = ()        # closed-set line keys (tone_cards.LINES) to play first, in order
    offer_human: bool = False
    handover: bool = False             # create the review ticket / route to a person this turn
    urgent_human: bool = False
    safety_flag: bool = False
    no_explain: bool = False           # say nothing about the decision this turn
    llm_allowed: bool = True           # False on template turns (no Groq call)
    end_with_check: bool = False
    simplify: bool = False             # replay the last point as the simplified lead
    fired: tuple[str, ...] = ()        # rule names that fired (turn log / judge view; never spoken)

    @property
    def tts(self) -> dict:
        return tts_params(self.tone)


# --------------------------------------------------------------------------------------------- derivation
def _tier_fields(tier: str) -> dict:
    words, sentences, speed = TIERS[tier]
    return {"tier": tier, "max_words": words, "max_sentences": sentences, "speed": speed}


def _default_mix(locale: str) -> str:
    return "native" if locale == "en" else "light_mix"


def _distress_override(t: ToneProfile, ack_kind: str, ack_available: bool) -> ToneProfile:
    return replace(t, **_tier_fields("distress"), warmth="ack_first" if ack_available else "warm",
                   directness="ack_then_answer" if ack_available else "answer_first", pauses="between_ideas",
                   checks="never", handoff="offer_now", empathy="ack_once" if ack_available else "none",
                   jargon="none", ack_kind=ack_kind if ack_available else "none")


def derive(explain: ExplainProfile | None, locale: str = "en", literacy: str = "medium",
           channel: str = "voice") -> ToneProfile:
    """Base tone at call start. Pure: same inputs, same profile."""
    locale = locale if locale in ("en", "hi", "te") else "en"
    assist = explain.assist if explain else ("GUIDED_VOICE" if channel == "voice" or literacy == "low" else "SELF_SERVE")
    if assist == "SELF_SERVE" and literacy != "low":
        t = ToneProfile(**_tier_fields("plain"), warmth="warm", directness="answer_first", pauses="none",
                        jargon="glossed", max_numbers=2, repeat_key_number=False, checks="end_only", empathy="light",
                        firmness="gentle_firm", handoff="end_offer", reply_locale=locale, mix=_default_mix(locale))
    else:
        t = ToneProfile(**_tier_fields("guided"), warmth="warm", directness="answer_first", pauses="between_ideas",
                        jargon="none", max_numbers=1, repeat_key_number=False, checks="every_turn", empathy="light",
                        firmness="gentle_firm", handoff="early_offer" if assist == "WANTS_PERSON" else "end_offer",
                        reply_locale=locale, mix=_default_mix(locale))
    if explain is None:
        return t
    seg = explain.primary
    if seg == "S1":
        t = replace(t, warmth="warm", max_numbers=min(t.max_numbers, 1))
    elif seg == "S3":
        t = replace(t, warmth="warm", max_numbers=0)
    elif seg == "S4":
        t = replace(t, warmth="warm", max_numbers=2, pauses="before_numbers")
    elif seg == "S5":
        t = replace(t, warmth="warm", checks="every_2nd" if t.checks == "end_only" else t.checks)
    elif seg == "S6":
        t = replace(t, warmth="neutral", firmness="gentle_firm")
    elif seg == "S7":
        t = replace(t, warmth="neutral", firmness="firm_minimal", handoff="early_offer", max_numbers=0)
    elif seg == "S8":
        t = replace(t, warmth="warm", max_numbers=3, pauses="before_numbers", repeat_key_number=True)
    t = replace(t, segment=seg)
    if explain.distress:
        t = _distress_override(t, "worry", True)
    # Income profile (Axis B) changes vocabulary and examples only: it is deliberately not read here.
    return t


def tts_params(tone: ToneProfile) -> dict:
    """Timbre options for this turn. The voice is fixed per call (never a collections voice); not set here."""
    return {"speed": check_tts_speed(tone.speed), "language": LOCALE_TTS.get(tone.reply_locale, LOCALE_TTS["en"])}


# --------------------------------------------------------------------------------------------- live rules
def note_barge_in(state: ToneState, false_interruption: bool = False) -> None:
    """LiveKit: the caller started speaking while the agent spoke. Backchannel (false interruption) is not counted."""
    if not false_interruption:
        state.current_barge += 1


def note_agent_turn_end(state: ToneState) -> None:
    state.barge_ins = (state.barge_ins + [state.current_barge])[-BARGE_WINDOW:]
    state.current_barge = 0


def _bump(state: ToneState) -> int:
    state.version += 1
    return state.version


def _apply_live_tone(base: ToneProfile, state: ToneState, literacy: str) -> ToneProfile:
    """The standing (sticky) part of the live state applied to the base tone."""
    state.reply_locale = state.reply_locale or base.reply_locale
    t = replace(base, reply_locale=state.reply_locale, mix=_mix(state, base))
    if state.guided_locked and t.tier in ("plain", "brief"):
        t = replace(t, **_tier_fields("guided"), checks="every_turn", jargon="none", pauses="between_ideas")
    if state.distress:
        t = _distress_override(t, state.ack_kind if state.ack_kind != "none" else "worry", False)
    return t


def _mix(state: ToneState, base: ToneProfile) -> str:
    if state.reply_locale == "en":
        return "native"
    e = state.latin_ratio_ema
    return "mirror_mix" if e > 0.3 else ("light_mix" if e >= 0.1 else "native")


def plan_user_turn(base: ToneProfile, state: ToneState, sig: Signals, literacy: str = "medium") -> SpeakingPlan:
    """Update the live state from one final transcript and return this turn's speaking plan (same turn, no LLM)."""
    state.turns += 1
    fired: list[str] = []
    lines: list[str] = []
    offer_human = handover = urgent = safety = no_explain = simplify = False
    llm_allowed = True
    state.reply_locale = state.reply_locale or base.reply_locale

    # code-mix level (EMA, alpha 0.5) and language streak, tracked every turn
    state.latin_ratio_ema = 0.5 * state.latin_ratio_ema + 0.5 * sig.latin_ratio

    # 1 human request
    if sig.human:
        handover = offer_human = True
        llm_allowed = False
        state.handed_over = True
        fired.append("P1:human_request")
    # 2 self-harm cue (flags accumulate even when a human was also asked for)
    if sig.self_harm:
        state.self_harm = state.distress = True
        safety = urgent = handover = offer_human = no_explain = True
        llm_allowed = False
        state.handed_over = True
        lines.append("safety")
        fired.append("P2:self_harm")
    # 3 distress / anger (sticky)
    if sig.distress or sig.anger:
        newly = not state.distress
        state.distress = True
        if sig.anger:
            state.anger_hits += 1
        kind = "frustration" if sig.anger and not sig.distress else "worry"
        if newly or state.ack_kind == "none":
            state.ack_kind = kind
        if not state.empathy_used and not safety:
            lines.append("ack_" + kind)
            state.empathy_used = True
        offer_human = True
        fired.append("P3:distress" if sig.distress else "P3:anger")
        if state.anger_hits >= 2:
            fired.append("P3:anger_x2")
    # 4 confusion
    if sig.confusion:
        state.consecutive_confusion += 1
        state.understood_streak = 0
    elif sig.understood:
        state.understood_streak += 1
        if state.understood_streak >= 2:
            state.consecutive_confusion = 0
            if literacy != "low":
                state.guided_locked = False
    if sig.confusion and not (handover or safety):
        n = state.consecutive_confusion
        if n >= CONFUSION_HANDOVER:
            lines.append("confusion_offer_person")
            offer_human = True
            fired.append("P4:confusion_x3")
        else:
            if n >= 2:
                state.guided_locked = True
            lines.append("ack_confusion")
            simplify = True
            fired.append(f"P7:confusion_x{n}")
        llm_allowed = False
    # 5 language switch
    switched = None
    if sig.lang_request and sig.lang_request != state.reply_locale:
        switched = sig.lang_request
        fired.append(f"P5:language_request->{switched}")
    elif sig.lang and sig.words >= LANG_MIN_WORDS:
        prev, count = state.lang_streak
        count = count + 1 if prev == sig.lang else 1
        state.lang_streak = (sig.lang, count)
        if sig.lang != state.reply_locale and count >= LANG_STREAK:
            switched = sig.lang
            fired.append(f"P6:language_streak->{switched}")
    if switched:
        state.reply_locale = switched
        state.lang_streak = (switched, 0)

    tone = _apply_live_tone(base, state, literacy)
    # ack line this turn -> the tone opens with it
    ack_now = next((x for x in lines if x.startswith("ack_") and x != "ack_confusion"), None)
    if ack_now:
        tone = _distress_override(tone, ack_now[4:], True)
    if sig.confusion and not (handover or safety) and not state.distress:
        speed = max(SPEED_FLOOR, round(min(tone.speed, TIERS["guided"][2]) - 0.05, 2))
        tone = replace(tone, **{**_tier_fields("guided"), "speed": speed}, checks="every_turn", jargon="none",
                       pauses="between_ideas", ack_kind="confusion")
    # 6 barge-ins: 2 in the last 3 agent turns -> brief (never over distress, confusion or a low-literacy floor)
    recent = sum(1 for b in state.barge_ins[-BARGE_WINDOW:] if b > 0)
    if recent >= BARGE_BRIEF and not state.distress and not sig.confusion and not sig.anger \
            and literacy != "low" and not state.guided_locked:
        tone = replace(tone, **_tier_fields("brief"), checks="never", pauses="none")
        fired.append("P8:barge_in_brief")
    elif state.barge_ins and state.barge_ins[-1] > 0 and tone.tier != "distress":
        tone = replace(tone, max_words=max(TIERS["brief"][0], int(tone.max_words * 0.7)))
        fired.append("P8:barge_in_shorter")

    if handover:
        if "handover_ack" not in lines:
            lines.append("handover_ack")
        tone = replace(tone, handoff="offer_now")
    if offer_human:
        tone = replace(tone, handoff="offer_now")
        state.human_offered = True
        if not handover and "confusion_offer_person" not in lines:
            lines.append("offer_person")
    elif tone.handoff == "early_offer" and not state.human_offered and state.turns <= 2:
        offer_human = True
        state.human_offered = True
        lines.append("offer_person")
        fired.append("handoff:early_offer")

    end_check = tone.checks == "every_turn" or (tone.checks == "every_2nd" and state.turns % 2 == 0)
    if offer_human or handover or tone.tier in ("brief", "distress"):
        end_check = False
    tone = replace(tone, version=_bump(state))
    return SpeakingPlan(tone=tone, lines=tuple(dict.fromkeys(lines)), offer_human=offer_human, handover=handover,
                        urgent_human=urgent, safety_flag=safety, no_explain=no_explain, llm_allowed=llm_allowed,
                        end_with_check=end_check, simplify=simplify, fired=tuple(fired))


def plan_silence(base: ToneProfile, state: ToneState, seconds: float, literacy: str = "medium") -> SpeakingPlan | None:
    """Caller silent after a question: 8 s -> gentle nudge; 15 s (LiveKit "away") -> summary-or-person. Max 2."""
    if seconds < SILENCE_NUDGE_S or state.silence_nudges >= MAX_SILENCE_NUDGES:
        return None
    tone = replace(_apply_live_tone(base, state, literacy), version=_bump(state))
    if seconds >= SILENCE_AWAY_S or state.silence_nudges >= 1:
        state.silence_nudges = MAX_SILENCE_NUDGES
        state.human_offered = True
        return SpeakingPlan(tone=replace(tone, handoff="offer_now"), lines=("nudge_away",), offer_human=True,
                            llm_allowed=False, fired=("P9:silence_15s",))
    state.silence_nudges += 1
    lines = ("nudge_wait", "offer_person") if state.distress else ("nudge_wait",)
    if state.distress:
        state.human_offered = True
    return SpeakingPlan(tone=tone, lines=lines, offer_human=state.distress, llm_allowed=False,
                        fired=("P9:silence_8s",))


def plan_close(base: ToneProfile, state: ToneState, literacy: str = "medium") -> SpeakingPlan:
    """End of call. Hand-over floor: a person is offered at least once in every call, at the latest here."""
    tone = replace(_apply_live_tone(base, state, literacy), version=_bump(state))
    need_offer = not state.human_offered or state.distress
    state.human_offered = True
    lines = ("offer_person", "close") if need_offer else ("close",)
    return SpeakingPlan(tone=tone, lines=lines, offer_human=need_offer, llm_allowed=False, fired=("close",))


# --------------------------------------------------------------------------------------------- speech rendering
_TAG = re.compile(r"<[^<>]{0,40}>")
_AMOUNT = re.compile(r"₹\s?\d")


def strip_tags(text: str) -> str:
    """Remove every <...> from model text before TTS (audio tags are code-inserted only)."""
    return re.sub(r"\s{2,}", " ", _TAG.sub("", text or "")).strip()


def render_for_speech(sentences: list[str], tone: ToneProfile, ack_first: bool = False) -> list[str]:
    """Each sentence is synthesised on its own. Adds at most MAX_CODE_TAGS tags: `<slow>` on a distress
    acknowledgement, `<pauses>` between ideas or before the sentence carrying a ₹ amount. Brief turns get none."""
    out = [strip_tags(s) for s in sentences if strip_tags(s)]
    budget = 0 if tone.tier == "brief" else MAX_CODE_TAGS
    if not out or budget == 0:
        return out
    if ack_first and tone.tier == "distress":
        out[0] = "<slow> " + out[0]
        budget -= 1
    for i in range(1, len(out)):
        if budget == 0:
            break
        if tone.pauses == "between_ideas" or (tone.pauses == "before_numbers" and _AMOUNT.search(out[i])):
            out[i] = "<pauses> " + out[i]
            budget -= 1
    return out


def count_tags(sentences: list[str]) -> int:
    return sum(len(_TAG.findall(s)) for s in sentences)


_WORDS = re.compile(r"\S+")


def word_count(text: str) -> int:
    return len(_WORDS.findall(strip_tags(text)))


def fit_budget(sentences: list[str], tone: ToneProfile, checks: int = 0) -> tuple[list[str], list[str]]:
    """Whole sentences that fit the tier's word/sentence budget, and the rest DEFERRED (next turn or written summary).

    Sentences are never cut or rewritten, so a number is either said exactly or deferred, never altered. The first
    sentence is always kept. `checks` trailing check-in questions are allowed on top of max_sentences."""
    kept, deferred, words = [], [], 0
    limit = tone.max_sentences
    for i, s in enumerate(sentences):
        n = word_count(s)
        is_check = i >= len(sentences) - checks
        if not kept or (words + n <= tone.max_words and (len(kept) < limit or is_check) and not deferred):
            kept.append(s)
            words += n
        else:
            deferred.append(s)
    return kept, deferred


# --------------------------------------------------------------------------------------------- numbers
def _exact(value) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        raise TypeError(f"not an amount: {value!r}")
    d = Decimal(str(value))
    if not d.is_finite():
        raise ValueError(f"not a finite amount: {value!r}")
    return d


def _group_indian(n: int) -> str:
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups + [tail])


def speak_amount(value) -> str:
    """₹ with Indian grouping, exactly as recorded: ₹1,50,000 / ₹5,200.50. Never rounded (paise are kept)."""
    d = _exact(value)
    sign = "-" if d < 0 else ""
    d = abs(d)
    rupees = int(d)
    paise = d - rupees
    text = sign + "₹" + _group_indian(rupees)
    if paise:
        p = (paise * 100)
        if p != p.to_integral_value():
            raise ValueError(f"amount has sub-paise precision, refusing to round: {value!r}")
        text += "." + f"{int(p):02d}"
    return text


UNIT_WORDS = {
    "en": {"crore": "crore", "lakh": "lakh", "thousand": "thousand", "hundred": "hundred", "rupees": "rupees",
           "paise": "paise", "and": "and"},
    "hi": {"crore": "करोड़", "lakh": "लाख", "thousand": "हज़ार", "hundred": "सौ", "rupees": "रुपये", "paise": "पैसे",
           "and": "और"},
    "te": {"crore": "కోట్ల", "lakh": "లక్షల", "thousand": "వేల", "hundred": "వందల", "rupees": "రూపాయలు",
           "paise": "పైసలు", "and": "మరియు"},
}


def amount_words(value, locale: str = "en") -> str:
    """Lakh/crore reading with digits per group, exact: 150000 -> "1 lakh 50 thousand rupees". A fallback for
    pre-rendering when Timbre's own normalisation is not trusted for a language (v3 Q3); never rounded."""
    w = UNIT_WORDS.get(locale, UNIT_WORDS["en"])
    d = _exact(value)
    if d < 0:
        raise ValueError("negative amounts are not spoken")
    rupees = int(d)
    paise_d = (d - rupees) * 100
    if paise_d != paise_d.to_integral_value():
        raise ValueError(f"amount has sub-paise precision, refusing to round: {value!r}")
    paise = int(paise_d)
    parts = []
    crore, rest = divmod(rupees, 10_000_000)
    lakh, rest = divmod(rest, 100_000)
    thousand, rest = divmod(rest, 1000)
    hundred, units = divmod(rest, 100)
    for n, unit in ((crore, "crore"), (lakh, "lakh"), (thousand, "thousand"), (hundred, "hundred")):
        if n:
            parts.append(f"{_group_indian(n)} {w[unit]}")
    if units or not parts:
        parts.append(str(units))
    text = " ".join(parts) + f" {w['rupees']}"
    if paise:
        text += f" {w['and']} {paise} {w['paise']}"
    return text


def amount_value_from_words(text: str) -> int:
    """Inverse of amount_words for whole rupees (used by tests to prove nothing was rounded)."""
    total = 0
    mult = {"crore": 10_000_000, "lakh": 100_000, "thousand": 1000, "hundred": 100}
    for loc in UNIT_WORDS.values():
        for k in mult:
            text = text.replace(loc[k], k)
    tokens = text.replace(",", "").split()
    i = 0
    while i < len(tokens):
        if tokens[i].isdigit():
            n = int(tokens[i])
            if i + 1 < len(tokens) and tokens[i + 1] in mult:
                total += n * mult[tokens[i + 1]]
                i += 2
                continue
            total += n
        i += 1
    return total


# --------------------------------------------------------------------------------------------- prompt fragment
_LEN_RULE = "Answer in at most {w} words and at most {s} sentences."
_FRAGMENTS = {
    "warmth": {"neutral": "Sound calm and factual.", "warm": "Sound like a kind bank officer; one friendly phrase at most.",
               "ack_first": "An acknowledgement has already been spoken; do not repeat it or apologise again."},
    "jargon": {"none": "Never say FOIR, DPD, bureau, tenure or cut-off; say 'how much of your income already goes "
                       "to EMIs', 'late payments', 'credit report', 'loan period'.",
               "glossed": "Common words like EMI, loan and credit score are fine; explain anything else in the same sentence."},
    "firmness": {"gentle_firm": "If asked to change the decision or reveal a limit, rule or check: say you can't in one "
                                "sentence, then say what you can do.",
                 "firm_minimal": "Give only the generic message about details that could not be verified; never "
                                 "describe any check."},
    "checks": {"every_turn": "End with one easy yes/no check.", "every_2nd": "", "end_only": "", "never": "Do not ask a question."},
}


def prompt_fragment(tone: ToneProfile) -> str:
    """Closed-table instructions for the slow-path compose call (no free text from the caller or worker)."""
    parts = [_LEN_RULE.format(w=tone.max_words, s=tone.max_sentences),
             _FRAGMENTS["warmth"][tone.warmth], _FRAGMENTS["jargon"][tone.jargon], _FRAGMENTS["firmness"][tone.firmness],
             f"Use only numbers from the facts, exactly as written, at most {tone.max_numbers}. Never round, never "
             "say 'about', never give percentages, thresholds or how close the applicant was.",
             "Never output angle-bracket tags.", _FRAGMENTS["checks"][tone.checks]]
    return " ".join(p for p in parts if p)
