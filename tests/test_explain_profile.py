"""explain_profile: v2 segmentation (reason segment, income profile, assist mode), rules only, synthetic records."""
import itertools
import re
from dataclasses import asdict

import pytest

from loan_advisory.agent import explain_profile as EP


def rec(cats, *, outcome="DECLINE", offer=None, bureau=None, evals=None, what_if=None):
    r = {"outcome": outcome, "decision": outcome, "reason_categories": list(cats), "offer": offer,
         "rule_evaluations": evals or [], "application_ref": "APP-SYN-1"}
    if bureau is not None:
        r["bureau"] = bureau
    if what_if is not None:
        r["what_if"] = what_if
    return r


COUNTER = {"amount": 150000, "emi": 5200, "tenure_months": 36, "counter_offer": True}
NO_HIT = {"match_status": "NO_HIT", "score": {"source": "NO_BUREAU_SCORE"}}


# ------------------------------------------------------------------------------------------- v2 §5.4 demo table
def test_caller1_affordability_salaried_voice():
    p = EP.resolve(rec(["REPAYMENTS_TOO_HIGH"]), employment_type="SALARIED", channel="voice")
    assert (p.primary, p.income, p.assist, p.distress) == ("S4", "SALARIED", "GUIDED_VOICE", False)
    assert p.allow_credit_types is False and p.ask_income is False


def test_caller2_thin_file_asks_income_then_gig():
    r = rec(["CREDIT_HISTORY"], bureau=NO_HIT)
    p = EP.resolve(r, employment_type=None, channel="voice")
    assert (p.primary, p.income, p.ask_income, p.assist) == ("S1", "UNKNOWN", True, "GUIDED_VOICE")
    stated = EP.parse_income_answer("Swiggy delivery chestanu")
    assert stated == "GIG_PLATFORM"
    p2 = EP.resolve(r, channel="voice", slots={"stated_income_type": stated})
    assert (p2.income, p2.ask_income) == ("GIG_PLATFORM", False)


def test_caller3_income_self_employed_text_self_serve():
    p = EP.resolve(rec(["INCOME_NOT_VERIFIED_OR_INSUFFICIENT"]), employment_type="SELF_EMPLOYED", channel="text",
                   literacy="high")
    assert (p.primary, p.income, p.assist, p.ask_income) == ("S5", "SELF_EMPLOYED", "SELF_SERVE", False)


def test_caller4_couldnt_verify_wants_person():
    p = EP.resolve(rec(["DETAILS_NOT_VERIFIED"]), channel="voice")
    assert (p.primary, p.assist) == ("S7", "WANTS_PERSON")


def test_caller5_repayment_medical_distress():
    p = EP.resolve(rec(["CREDIT_HISTORY"], bureau={"match_status": "HIT", "score": {"source": "SYNTHETIC_BUREAU"}}),
                   channel="voice", intent="distress", product_id="PL_MEDICAL_EMERGENCY")
    assert (p.primary, p.assist, p.distress, p.allow_credit_types) == ("S3", "WANTS_PERSON", True, False)


def test_caller6_counter_offer_then_policy():
    p = EP.resolve(rec(["REQUEST_OUTSIDE_OFFER"], outcome="COUNTER_OFFER", offer=COUNTER), channel="text")
    assert p.segments == ("S6", "S8") and p.primary == "S6" and p.secondary == ("S8",) and p.has_offer
    assert p.assist == "SELF_SERVE"


def test_counter_offer_alone_is_s8():
    p = EP.resolve(rec([], outcome="COUNTER_OFFER", offer=COUNTER))
    assert p.primary == "S8"


# ------------------------------------------------------------------------------------------- precedence
@pytest.mark.parametrize("cats,expected", [
    (["CREDIT_HISTORY", "DETAILS_NOT_VERIFIED"], "S7"),
    (["REPAYMENTS_TOO_HIGH", "REQUEST_OUTSIDE_OFFER"], "S6"),
    (["REPAYMENTS_TOO_HIGH", "CREDIT_HISTORY"], "S3"),
    (["DOCUMENTS_INCOMPLETE_OR_UNCLEAR", "RECENT_CREDIT_APPLICATIONS"], "S4"),
    (["EMPLOYMENT_STABILITY"], "S5"),
    (["OTHER"], "S5"),
])
def test_precedence(cats, expected):
    assert EP.resolve(rec(cats, bureau={"match_status": "HIT"})).primary == expected


def test_full_precedence_order():
    cats = ["DOCUMENTS_INCOMPLETE_OR_UNCLEAR", "REPAYMENTS_TOO_HIGH", "CREDIT_HISTORY", "REQUEST_OUTSIDE_OFFER",
            "DETAILS_NOT_VERIFIED"]
    p = EP.resolve(rec(cats, outcome="COUNTER_OFFER", offer=COUNTER, bureau={"match_status": "HIT"}))
    assert p.segments == ("S7", "S6", "S3", "S4", "S5", "S8")
    assert len(p.secondary) == 1


def test_thin_file_detected_from_dq_code():
    r = rec(["CREDIT_HISTORY"], bureau={"data_quality": {"reason_codes": ["DQ_BUREAU_INSUFFICIENT_HISTORY"]}})
    assert EP.resolve(r).primary == "S1"


def test_other_with_eligibility_rule_is_s6():
    evals = [{"rule_id": "PL-X01", "result": "FAIL", "effect": "DECLINE"}]
    p = EP.resolve(rec(["OTHER"], evals=evals), rule_groups={"PL-X01": "A_APPLICANT_ELIGIBILITY"})
    assert p.primary == "S6"
    p2 = EP.resolve(rec(["OTHER"], evals=evals), rule_groups={"PL-X01": "G_REPAYMENT_CAPACITY_AND_FOIR"})
    assert p2.primary == "S5"


def test_not_segmented_for_approvals_holds_errors():
    assert EP.resolve(rec([], outcome="ELIGIBILITY_CONDITIONS_MET",
                          offer={"amount": 300000, "emi": 9000, "tenure_months": 36, "counter_offer": False})) is None
    for o in ("MANUAL_REVIEW", "MORE_INFORMATION_REQUIRED", "ERROR"):
        assert EP.resolve(rec(["DETAILS_NOT_VERIFIED"], outcome=o)) is None
    assert EP.resolve(None) is None
    # v1 records say REJECTED
    assert EP.resolve(rec(["CREDIT_HISTORY"], outcome="REJECTED")).primary in ("S1", "S3")


# ------------------------------------------------------------------------------------------- fixable_soon
def test_fixable_soon_is_ordering_only_and_never_changes_segment():
    a = EP.resolve(rec(["REPAYMENTS_TOO_HIGH"], what_if={"outcome": "APPROVED"}))
    b = EP.resolve(rec(["REPAYMENTS_TOO_HIGH"]))
    assert a.fixable_soon and not b.fixable_soon
    da, db = asdict(a), asdict(b)
    for k in ("fixable_soon", "reasons"):
        da.pop(k); db.pop(k)
    assert da == db
    assert not EP.resolve(rec(["CREDIT_HISTORY"], what_if={"outcome": "APPROVED"})).fixable_soon


# ------------------------------------------------------------------------------------------- assist mode
def test_assist_mode_rules():
    r = rec(["EMPLOYMENT_STABILITY"])
    assert EP.resolve(r, channel="text", literacy="medium").assist == "SELF_SERVE"
    assert EP.resolve(r, channel="text", literacy="low").assist == "GUIDED_VOICE"
    assert EP.resolve(r, channel="text", slots={"consecutive_clarify_count": 2}).assist == "GUIDED_VOICE"
    assert EP.resolve(r, channel="text", slots={"stated_prefers_person": True}).assist == "WANTS_PERSON"
    assert EP.resolve(r, channel="text", intent="human_review").assist == "WANTS_PERSON"


def test_credit_types_never_for_s3_s4_or_distress():
    assert not EP.resolve(rec(["CREDIT_HISTORY"], bureau={"match_status": "HIT"})).allow_credit_types
    assert not EP.resolve(rec(["REPAYMENTS_TOO_HIGH"])).allow_credit_types
    assert EP.resolve(rec(["EMPLOYMENT_STABILITY"])).allow_credit_types
    assert not EP.resolve(rec(["EMPLOYMENT_STABILITY"]), distress=True).allow_credit_types


def test_income_only_asked_for_s1_s5_when_unknown_or_conflicting():
    assert not EP.resolve(rec(["REPAYMENTS_TOO_HIGH"])).ask_income
    assert not EP.resolve(rec(["EMPLOYMENT_STABILITY"]), employment_type="SALARIED").ask_income
    assert EP.resolve(rec(["EMPLOYMENT_STABILITY"]), employment_type="SALARIED",
                      slots={"income_conflict": True}).ask_income


@pytest.mark.parametrize("text,profile", [
    ("I run a kirana shop", "SELF_EMPLOYED"), ("meri dukaan hai", "SELF_EMPLOYED"), ("Zomato delivery", "GIG_PLATFORM"),
    ("kheti karta hoon", "FARM"), ("నేను వ్యవసాయం చేస్తాను", "FARM"), ("monthly salary", "SALARIED"),
    ("usne bola theek hai", None), ("", None),
])
def test_parse_income_answer(text, profile):
    assert EP.parse_income_answer(text) == profile


# ------------------------------------------------------------------------------------------- invariance + proxies
def test_same_record_same_categories_across_locale_channel_literacy():
    r = rec(["REQUEST_OUTSIDE_OFFER", "REPAYMENTS_TOO_HIGH"], outcome="COUNTER_OFFER", offer=COUNTER)
    seen = set()
    for ch, lit in itertools.product(("text", "voice"), ("low", "medium", "high")):
        p = EP.resolve(r, channel=ch, literacy=lit)
        seen.add((p.primary, p.segments, p.categories, p.has_offer))
    assert len(seen) == 1


BASE_APPLICANT = {"employment_type": "SALARIED", "name": "Synthetic Ravi", "dob": "1990-01-01", "gender": "M",
                  "marital_status": "SINGLE", "residence_city": "Hyderabad", "pincode": "500001", "metro": True,
                  "segment_assignment": "SALARIED_METRO", "age": 36}


@pytest.mark.parametrize("field,value", [("name", "Synthetic Lakshmi"), ("dob", "1960-05-05"), ("gender", "F"),
                                         ("marital_status", "MARRIED"), ("residence_city", "Mahabubnagar"),
                                         ("pincode", "509001"), ("metro", False), ("segment_assignment", "NEAR_PRIME"),
                                         ("age", 64)])
def test_protected_attributes_and_proxies_change_nothing(field, value):
    r = rec(["CREDIT_HISTORY", "REPAYMENTS_TOO_HIGH"], bureau={"match_status": "HIT"})
    a = EP.resolve_for_applicant(r, BASE_APPLICANT, channel="voice")
    b = EP.resolve_for_applicant(r, {**BASE_APPLICANT, field: value}, channel="voice")
    assert a == b


def test_resolver_signature_reads_no_protected_fields():
    import inspect
    params = set(inspect.signature(EP.resolve).parameters)
    for banned in ("age", "dob", "gender", "marital_status", "name", "city", "pincode", "metro", "segment"):
        assert banned not in params
    assert EP.ALLOWED_APPLICANT_FIELDS == ("employment_type",)


def test_no_near_miss_wording_anywhere_in_module():
    src = open(EP.__file__, encoding="utf-8").read()
    code = re.sub(r'""".*?"""', "", src, flags=re.S)  # docstrings explain the ban; code must not carry the words
    assert not re.search(r"you were close|almost|just short|barely", code, re.I)
