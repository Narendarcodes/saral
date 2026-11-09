"""Explanation-side profile of a declined or counter-offered applicant (research/customer-segmentation-v2.md).

ZEN decides; Saral explains. This module picks HOW Saral explains and WHICH next steps it offers. It never changes
the outcome, amount, rate or reasons, never feeds back into ZEN and never invents a reason ZEN did not fire.

Three axes, all pure rules (no clustering, no model):
- reason segment (Axis A) from the closed borrower reason categories stored on the ZEN record:
  S1 thin file, S3 repayment track, S4 affordability, S5 income/employment/documents, S6 policy fit,
  S7 couldn't verify, S8 counter-offer. Precedence S7 > S6 > S3 > S4 > S5 > S1, then S8.
- income profile (Axis B): SALARIED / SELF_EMPLOYED / GIG_PLATFORM / FARM / UNKNOWN, from the recorded
  employment type or the caller's own one-line answer. Asked for only when the primary segment is S1 or S5.
- assist mode (Axis C): SELF_SERVE / GUIDED_VOICE / WANTS_PERSON, from channel, applicant-chosen literacy and
  behaviour (asked for a person, clarify turns), never from who the caller is.

Never read: age, DOB, gender, marital status, name, residence city, pincode, metro flag, ZEN's decision-side
`SegmentAssignment` / threshold groups, voice features or device data. `resolve_for_applicant` copies only
`employment_type` out of an applicant mapping, so nothing else can leak in. `fixable_soon` (the v1 near-miss idea)
only orders next steps; nothing here produces "you were close", a gap or a threshold.

Synthetic prototype only: the records this reads are produced by the synthetic catalogue-v2 policy.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

from loan_advisory.agent import facts as F

SEGMENT_ORDER = ("S7", "S6", "S3", "S4", "S5", "S1", "S8")
INCOME_PROFILES = ("SALARIED", "SELF_EMPLOYED", "GIG_PLATFORM", "FARM", "UNKNOWN")
ASSIST_MODES = ("SELF_SERVE", "GUIDED_VOICE", "WANTS_PERSON")

S5_CATEGORIES = frozenset({"INCOME_NOT_VERIFIED_OR_INSUFFICIENT", "EMPLOYMENT_STABILITY", "DOCUMENTS_INCOMPLETE_OR_UNCLEAR"})
S4_CATEGORIES = frozenset({"REPAYMENTS_TOO_HIGH", "RECENT_CREDIT_APPLICATIONS"})
THIN_FILE_DQ = frozenset({"DQ_BUREAU_NO_HIT", "DQ_BUREAU_INSUFFICIENT_HISTORY"})
# Categories a borrower can usually act on before reapplying (documents, EMIs, applying less often). Used for
# `fixable_soon` ordering only.
CURABLE_CATEGORIES = frozenset({"DOCUMENTS_INCOMPLETE_OR_UNCLEAR", "INCOME_NOT_VERIFIED_OR_INSUFFICIENT",
                                "REPAYMENTS_TOO_HIGH", "RECENT_CREDIT_APPLICATIONS"})
# Catalogue-v2 rule groups whose failures are eligibility / product-fit policy (S6 when the category is OTHER).
ELIGIBILITY_GROUPS = frozenset({"A_APPLICANT_ELIGIBILITY", "H_LOAN_REQUEST_AND_PRODUCT_PARAMETERS"})
_REACHABLE_WHAT_IF = ("APPROVED", "ELIGIBILITY_CONDITIONS_MET", "COUNTER_OFFER")
_CATALOGUE = Path(__file__).resolve().parents[5] / "docs" / "saral" / "rule_catalogue.v2.json"

# The only applicant fields this module may read. Everything else on an applicant is ignored by construction.
ALLOWED_APPLICANT_FIELDS = ("employment_type",)


@dataclass(frozen=True)
class ExplainProfile:
    primary: str                         # S1 | S3 | S4 | S5 | S6 | S7 | S8
    secondary: tuple[str, ...]           # at most one, spoken only on request
    segments: tuple[str, ...]            # every segment that fired, in precedence order
    categories: tuple[str, ...]          # the record's borrower reason category keys (all go in the written summary)
    fixable_soon: bool                   # ordering hint only, never spoken as "close"
    income: str                          # INCOME_PROFILES
    ask_income: bool                     # ask "how do you mostly earn?" once
    assist: str                          # ASSIST_MODES
    distress: bool
    allow_credit_types: bool             # may describe other credit TYPES if the caller asks (never S3/S4/distress)
    has_offer: bool
    reasons: tuple[str, ...] = field(default=())   # which rules fired (debug / judge view; never spoken)


@lru_cache(maxsize=1)
def _rule_groups() -> dict[str, str]:
    try:
        cat = json.loads(_CATALOGUE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {r["rule_id"]: r.get("rule_group") or "" for r in cat.get("rules", [])}


def _failed_rule_ids(record: Mapping) -> list[str]:
    out = []
    for e in record.get("rule_evaluations") or []:
        if e.get("result") == "FAIL" and e.get("effect") not in ("NONE", "NO_EFFECT"):
            out.append(str(e.get("rule_id")))
    return out


def eligibility_fail(record: Mapping, rule_groups: Mapping[str, str] | None = None) -> bool:
    """A failed rule from the eligibility / product-parameter groups (server-side only; never voiced)."""
    groups = _rule_groups() if rule_groups is None else rule_groups
    return any(groups.get(rid) in ELIGIBILITY_GROUPS for rid in _failed_rule_ids(record))


def is_thin_file(record: Mapping) -> bool:
    bureau = record.get("bureau") or {}
    score = bureau.get("score") or {}
    dq = bureau.get("data_quality") or record.get("review") or {}
    codes = {str(c) for c in dq.get("reason_codes") or []}
    return bool(bureau.get("new_to_credit") or bureau.get("match_status") == "NO_HIT"
                or score.get("source") == "NO_BUREAU_SCORE" or codes & THIN_FILE_DQ)


def _offer(record: Mapping) -> dict | None:
    return F.borrower_offer(dict(record))


def is_segmentable(record: Mapping | None) -> bool:
    """Only credit rejections and counter-offers are segmented; approvals, holds and errors keep their templates."""
    if not record:
        return False
    outcome = F.borrower_outcome(record.get("outcome") or record.get("decision"))
    offer = _offer(record)
    if outcome == "COUNTER" or (offer and offer.get("counter_offer")):
        return True
    return outcome == "REJECTED"


def _fixable_soon(record: Mapping, cats: Iterable[str]) -> bool:
    cats = set(cats)
    if not cats or not cats <= CURABLE_CATEGORIES:
        return False
    wi = record.get("what_if") or {}
    return bool(_offer(record)) or wi.get("outcome") in _REACHABLE_WHAT_IF or wi.get("result") in _REACHABLE_WHAT_IF


def resolve(record: Mapping | None, *, employment_type: str | None = None, channel: str = "voice",
            literacy: str = "medium", intent: str | None = None, distress: bool = False,
            product_id: str | None = None, slots: Mapping | None = None,
            rule_groups: Mapping[str, str] | None = None) -> ExplainProfile | None:
    """v2 §5.2 as one pure function. None for records that are not segmented (use the existing outcome templates)."""
    if not is_segmentable(record):
        return None
    slots = slots or {}
    cats = [c["key"] for c in F.reason_categories(dict(record))]
    cat_set = set(cats)
    offer = _offer(record)
    fired: list[str] = []
    segs: list[str] = []
    if "DETAILS_NOT_VERIFIED" in cat_set:
        segs.append("S7")
        fired.append("DETAILS_NOT_VERIFIED->S7")
    if "REQUEST_OUTSIDE_OFFER" in cat_set or ("OTHER" in cat_set and eligibility_fail(record, rule_groups)):
        segs.append("S6")
        fired.append("policy-fit->S6")
    if "CREDIT_HISTORY" in cat_set:
        seg = "S1" if is_thin_file(record) else "S3"
        segs.append(seg)
        fired.append("CREDIT_HISTORY->" + seg)
    if cat_set & S4_CATEGORIES:
        segs.append("S4")
        fired.append("affordability->S4")
    if cat_set & S5_CATEGORIES:
        segs.append("S5")
        fired.append("income/docs->S5")
    if offer and offer.get("counter_offer"):
        segs.append("S8")
        fired.append("counter-offer->S8")
    ordered = sorted(set(segs), key=SEGMENT_ORDER.index)
    if not ordered:
        ordered = ["S5"]                                    # OTHER-only: procedural default (v2 §5.2)
        fired.append("OTHER-only->S5")
    primary, secondary = ordered[0], tuple(ordered[1:2])

    stated = slots.get("stated_income_type")
    stated = stated.upper() if isinstance(stated, str) else None
    recorded = {"SALARIED": "SALARIED", "SELF_EMPLOYED": "SELF_EMPLOYED"}.get(str(employment_type or "").upper())
    income = stated if stated in INCOME_PROFILES else (recorded or "UNKNOWN")
    ask_income = primary in ("S5", "S1") and not stated and (income == "UNKNOWN" or bool(slots.get("income_conflict")))

    distress = bool(distress or intent == "distress")
    if slots.get("stated_prefers_person") or intent == "human_review" or primary == "S7" \
            or (product_id == "PL_MEDICAL_EMERGENCY" and distress):
        assist = "WANTS_PERSON"
    elif channel == "voice" or literacy == "low" or int(slots.get("consecutive_clarify_count", 0) or 0) >= 2:
        assist = "GUIDED_VOICE"
    else:
        assist = "SELF_SERVE"

    return ExplainProfile(primary=primary, secondary=secondary, segments=tuple(ordered), categories=tuple(cats),
                          fixable_soon=_fixable_soon(record, cats), income=income, ask_income=ask_income,
                          assist=assist, distress=distress,
                          allow_credit_types=primary in ("S1", "S5", "S6") and not distress,
                          has_offer=bool(offer), reasons=tuple(fired))


def resolve_for_applicant(record: Mapping | None, applicant: Mapping | None, **kw) -> ExplainProfile | None:
    """`resolve` with an applicant mapping: only ALLOWED_APPLICANT_FIELDS are copied out of it."""
    emp = (applicant or {}).get("employment_type")
    emp = getattr(emp, "value", emp)
    return resolve(record, employment_type=emp, **kw)


INCOME_QUESTION = {
    "en": "How do you mostly earn: a salary, your own business, app or gig work, or farming?",
    "hi": "आप ज़्यादातर कैसे कमाते हैं: सैलरी, अपना कारोबार, ऐप या गिग का काम, या खेती?",
    "te": "మీరు ఎక్కువగా ఎలా సంపాదిస్తారు: జీతం, సొంత వ్యాపారం, యాప్ లేదా గిగ్ పని, లేదా వ్యవసాయం?",
}

_INCOME_WORDS = (
    ("GIG_PLATFORM", ("swiggy", "zomato", "uber", "ola", "rapido", "delivery", "gig", "app work", "app ka kaam",
                      "freelanc", "डिलीवरी", "డెలివరీ")),
    ("FARM", ("farm", "kheti", "kisan", "crop", "dairy", "fisher", "vyavasayam", "vyavasaya", "खेती", "किसान",
              "వ్యవసాయం", "పొలం")),
    ("SELF_EMPLOYED", ("shop", "kirana", "dukaan", "dukan", "business", "trader", "own work", "vyapar", "vyaparam",
                       "दुकान", "कारोबार", "व्यापार", "వ్యాపారం", "షాప్", "అంగడి")),
    ("SALARIED", ("salary", "salaried", "naukri", "job", "employee", "jeetham", "jeetam", "सैलरी", "नौकरी", "జీతం",
                  "ఉద్యోగం")),
)


def _has_word(text: str, word: str) -> bool:
    """Latin words match at a word start (so "ola" never matches "bola"); native-script words match as substrings."""
    if word.isascii():
        return re.search(r"(?<![a-z])" + re.escape(word), text) is not None
    return word in text


def parse_income_answer(text: str) -> str | None:
    """The caller's one-line answer to INCOME_QUESTION -> an income profile, or None if unclear.

    Only the words the caller said are used; it is never inferred from voice, accent, language, name or area."""
    t = (text or "").lower()
    for profile, words in _INCOME_WORDS:
        if any(_has_word(t, w) for w in words):
            return profile
    return None
