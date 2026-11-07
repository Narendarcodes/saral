"""Closed-set spoken lines, the fact card and deterministic lead templates for the voice tone engine.

Everything the fast path says comes from here, with no LLM:
- LINES: acknowledgements, fillers, silence nudges, hand-over, offer-a-person, consent check and close, in en/hi/te.
  Each is at most 14 words so it can be pre-rendered. The Hindi lines use feminine self-reference to match the fixed
  Timbre voice (Tarika) and address the caller with gender-neutral "aap" forms. Hindi/Telugu need one native-speaker
  read before a live demo (v3 Q10).
- SAFETY: the self-harm safety line is a TODO placeholder. It is NOT approved and NOT verified (no helpline was
  checked); `line()` returns None for it until the owner supplies approved wording, so it can never be spoken.
- fact_card(): the decision facts from the ZEN record via facts.py (outcome, reason categories, the offer as exact ₹
  strings, the document checklist, KFS facts that are stored on the record). Tone never edits a card.
- lead(): the sentences for one intent in one tier and locale, built only from the card.
- NEVER_SAY: a global no-threshold / no-"you were close" pattern plus a per-segment never-say list.
"""
from __future__ import annotations

import re
from typing import Mapping

from loan_advisory.agent import facts as F
from loan_advisory.agent.explain_profile import ExplainProfile
from loan_advisory.agent.tone import speak_amount

LOCALES = ("en", "hi", "te")

LINES: dict[str, dict[str, str]] = {
    "ack_worry": {"en": "I can hear this is worrying, and I'm sorry you're going through this.",
                  "hi": "मैं समझ सकती हूँ कि यह चिंता की बात है, मुझे खेद है।",
                  "te": "ఇది ఆందోళన కలిగించే విషయమని అర్థమవుతోంది, క్షమించండి."},
    "ack_frustration": {"en": "I understand this is frustrating.",
                        "hi": "मैं समझती हूँ कि यह परेशान करने वाला है।",
                        "te": "ఇది విసుగు కలిగిస్తుందని అర్థం చేసుకుంటాను."},
    "ack_confusion": {"en": "Let me say that more simply.",
                      "hi": "मैं इसे और आसान तरीके से कहती हूँ।",
                      "te": "దీన్ని ఇంకా సులభంగా చెబుతాను."},
    "filler_calm": {"en": "One moment, I'm checking that for you.",
                    "hi": "एक पल, मैं आपके लिए देख रही हूँ।",
                    "te": "ఒక్క క్షణం, మీ కోసం చూస్తున్నాను."},
    "filler_neutral": {"en": "Just a second, let me check.",
                       "hi": "एक सेकंड, मैं देखती हूँ।",
                       "te": "ఒక్క సెకను, చూస్తాను."},
    "filler_brief": {"en": "One second.", "hi": "एक सेकंड।", "te": "ఒక్క సెకను."},
    "nudge_wait": {"en": "Take your time, I'm here.",
                   "hi": "आराम से बोलिए, मैं यहीं हूँ।",
                   "te": "నెమ్మదిగా చెప్పండి, నేను ఇక్కడే ఉన్నాను."},
    "nudge_away": {"en": "Shall I put a summary in your portal, or have a person call you?",
                   "hi": "क्या मैं पोर्टल में सारांश डाल दूँ, या कोई व्यक्ति आपको कॉल करे?",
                   "te": "పోర్టల్‌లో సారాంశం పెట్టమంటారా, లేక ఒకరు మీకు కాల్ చేయాలా?"},
    "handover_ack": {"en": "I'll ask a person on our team to review it.",
                     "hi": "मैं हमारी टीम के एक व्यक्ति से इसे देखने को कहती हूँ।",
                     "te": "మా బృందంలో ఒకరిని దీన్ని చూడమని అడుగుతాను."},
    "offer_person": {"en": "Would you like a person on our team to go through it with you?",
                     "hi": "क्या आप चाहेंगे कि हमारी टीम का कोई व्यक्ति इसे आपके साथ देखे?",
                     "te": "మా బృందంలో ఒకరు మీతో కలిసి చూడాలనుకుంటున్నారా?"},
    "confusion_offer_person": {"en": "I may not be explaining this well. Shall someone from our team call you?",
                               "hi": "शायद मैं ठीक से नहीं समझा पा रही। क्या टीम का कोई व्यक्ति आपको कॉल करे?",
                               "te": "నేను సరిగ్గా వివరించలేకపోతున్నానేమో. మా బృందంలో ఒకరు మీకు కాల్ చేయాలా?"},
    "consent_check": {"en": "Is it okay to talk about your application right now?",
                      "hi": "क्या अभी आपके आवेदन के बारे में बात करना ठीक है?",
                      "te": "ఇప్పుడు మీ దరఖాస్తు గురించి మాట్లాడవచ్చా?"},
    "close": {"en": "Thank you for calling. A summary will be in your portal.",
              "hi": "कॉल करने के लिए धन्यवाद। सारांश आपके पोर्टल में होगा।",
              "te": "కాల్ చేసినందుకు ధన్యవాదాలు. సారాంశం మీ పోర్టల్‌లో ఉంటుంది."},
    "verified": {"en": "Thank you, you're verified.", "hi": "धन्यवाद, आपकी पहचान हो गई है।",
                 "te": "ధన్యవాదాలు, మీ ధృవీకరణ పూర్తయింది."},
}

# TODO(owner, v3 Q7): approved self-harm safety wording and a verified helpline. Placeholder only, never spoken.
SAFETY_LINE_APPROVED = False
SAFETY: dict[str, str] = {loc: "TODO: approved safety line (unverified placeholder)" for loc in LOCALES}


def line(key: str, locale: str = "en") -> str | None:
    """A closed-set line in the locale (English fallback). None for the unapproved safety placeholder."""
    if key == "safety":
        return SAFETY.get(locale, SAFETY["en"]) if SAFETY_LINE_APPROVED else None
    row = LINES.get(key)
    if row is None:
        raise KeyError(key)
    return row.get(locale) or row["en"]


def filler_key(tier: str, distress: bool) -> str:
    return "filler_calm" if distress or tier == "guided" else ("filler_brief" if tier == "brief" else "filler_neutral")


# --------------------------------------------------------------------------------------------- segment wording
SEGMENT_LINE: dict[str, dict[str, str]] = {
    "S1": {"en": "Having no loan history isn't a bad mark; the lender just has nothing to go on yet.",
           "hi": "लोन का इतिहास न होना कोई बुरी बात नहीं; बस अभी देखने को कोई रिकॉर्ड नहीं है।",
           "te": "లోన్ చరిత్ర లేకపోవడం చెడ్డ గుర్తు కాదు; చూడటానికి ఇంకా రికార్డు లేదు అంతే."},
    "S3": {"en": "This can improve with on-time payments, and you can dispute anything wrong on your report.",
           "hi": "समय पर भुगतान से यह सुधर सकता है, और रिपोर्ट में गलती हो तो आप शिकायत कर सकते हैं।",
           "te": "సమయానికి చెల్లింపులతో ఇది మెరుగవుతుంది; రిపోర్ట్‌లో తప్పు ఉంటే మీరు అభ్యంతరం చెప్పవచ్చు."},
    "S4": {"en": "Paying down a small loan first can help.",
           "hi": "पहले कोई छोटा लोन चुकाना मदद कर सकता है।",
           "te": "ముందుగా ఒక చిన్న లోన్ తీర్చడం సహాయపడుతుంది."},
    "S5": {"en": "Documents are usually the quickest thing to fix.",
           "hi": "दस्तावेज़ ठीक करना आमतौर पर सबसे जल्दी हो जाता है।",
           "te": "పత్రాలు సరిచేయడం సాధారణంగా చాలా త్వరగా అయ్యే పని."},
    "S6": {"en": "This is a rule for this loan, not a judgment of you.",
           "hi": "यह इस लोन का नियम है, आपके बारे में कोई राय नहीं।",
           "te": "ఇది ఈ లోన్ నియమం, మీ గురించి తీర్పు కాదు."},
}
S7_GENERIC = {"en": "We couldn't verify some details in your application. Our team will review it and may contact you.",
              "hi": "हम आपके आवेदन की कुछ जानकारी की पुष्टि नहीं कर सके। हमारी टीम इसे देखेगी और आपसे संपर्क कर सकती है।",
              "te": "మీ దరఖాస్తులోని కొన్ని వివరాలను మేము నిర్ధారించలేకపోయాము. మా బృందం దీన్ని పరిశీలించి మిమ్మల్ని సంప్రదించవచ్చు."}
S7_NOT_ACCUSATION = {"en": "This is not an accusation.", "hi": "यह कोई आरोप नहीं है।", "te": "ఇది ఆరోపణ కాదు."}
MAIN_REASON = {"en": "Mainly, {reason}.", "hi": "मुख्य कारण: {reason}।", "te": "ముఖ్య కారణం: {reason}."}
CHECK = {"en": "Shall I tell you what helps?", "hi": "क्या मैं बताऊँ कि क्या मदद करेगा?",
         "te": "ఏది సహాయపడుతుందో చెప్పమంటారా?"}
OFFER_FULL = {
    "en": ["We can't offer the full amount, but we can offer {amount}.", "The EMI would be {emi} a month for {months} months.",
           "This is not a final sanction yet.", "There's no need to decide on this call."],
    "hi": ["हम पूरी राशि नहीं दे सकते, पर {amount} दे सकते हैं।", "EMI {emi} महीना होगी, {months} महीनों के लिए।",
           "यह अभी अंतिम मंज़ूरी नहीं है।", "इस कॉल पर फ़ैसला करने की ज़रूरत नहीं है।"],
    "te": ["పూర్తి మొత్తం ఇవ్వలేము, కానీ {amount} ఇవ్వగలము.", "EMI నెలకు {emi}, {months} నెలల పాటు.",
           "ఇది ఇంకా తుది మంజూరు కాదు.", "ఈ కాల్‌లోనే నిర్ణయించాల్సిన అవసరం లేదు."],
}
OFFER_BRIEF = {"en": "We can offer {amount}, EMI {emi} a month, for {months} months.",
               "hi": "हम {amount} दे सकते हैं, EMI {emi} महीना, {months} महीने।",
               "te": "{amount} ఇవ్వగలము, EMI నెలకు {emi}, {months} నెలలు."}
OFFER_REPEAT = {"en": "That's {amount}.", "hi": "यानी {amount}।", "te": "అంటే {amount}."}
NO_OFFER = {"en": "There's no offer on this application right now.", "hi": "इस आवेदन पर अभी कोई ऑफ़र नहीं है।",
            "te": "ఈ దరఖాస్తుపై ప్రస్తుతం ఆఫర్ లేదు."}

CHECKLIST: dict[str, dict[str, list[str]]] = {
    "SALARIED": {"en": ["Recent salary slips.", "A bank statement showing your salary credits."],
                 "hi": ["हाल की सैलरी स्लिप।", "सैलरी जमा दिखाने वाला बैंक स्टेटमेंट।"],
                 "te": ["ఇటీవలి జీతం స్లిప్‌లు.", "జీతం జమ చూపించే బ్యాంక్ స్టేట్‌మెంట్."]},
    "SELF_EMPLOYED": {"en": ["Bank statements of the account where your sales money goes in.",
                             "Your GST returns, if you are registered."],
                      "hi": ["उस खाते के बैंक स्टेटमेंट जिसमें बिक्री का पैसा आता है।", "GST रिटर्न, अगर आप रजिस्टर्ड हैं।"],
                      "te": ["అమ్మకాల డబ్బు జమ అయ్యే ఖాతా బ్యాంక్ స్టేట్‌మెంట్లు.", "మీరు నమోదు అయి ఉంటే GST రిటర్న్‌లు."]},
    "GIG_PLATFORM": {"en": ["Bank statements showing your payouts.", "Your platform payout statements."],
                     "hi": ["पेआउट दिखाने वाले बैंक स्टेटमेंट।", "प्लेटफ़ॉर्म के पेआउट स्टेटमेंट।"],
                     "te": ["మీ పేఅవుట్‌లు చూపించే బ్యాంక్ స్టేట్‌మెంట్లు.", "ప్లాట్‌ఫాం పేఅవుట్ స్టేట్‌మెంట్లు."]},
    "FARM": {"en": ["Bank statements showing your income.", "Your land or tenancy record, if you have one."],
             "hi": ["आय दिखाने वाले बैंक स्टेटमेंट।", "ज़मीन या पट्टे का रिकॉर्ड, अगर हो।"],
             "te": ["ఆదాయం చూపించే బ్యాంక్ స్టేట్‌మెంట్లు.", "భూమి లేదా కౌలు రికార్డు, ఉంటే."]},
    "UNKNOWN": {"en": ["Proof of regular income, usually bank statements."],
                "hi": ["नियमित आय का सबूत, आमतौर पर बैंक स्टेटमेंट।"],
                "te": ["క్రమమైన ఆదాయానికి రుజువు, సాధారణంగా బ్యాంక్ స్టేట్‌మెంట్లు."]},
}

# --------------------------------------------------------------------------------------------- never-say
NEVER_SAY_ALL = re.compile(
    r"\b(you\s+were|you're|you\s+are)\s+(so\s+|very\s+)?close\b|\balmost\b|\bjust\s+short\b|\bbarely\b|"
    r"\bonly\s+a\s+little\b|\bcut-?off\b|\bthreshold\b|\bFOIR\b|\bDPD\b|\bPL-\d{3}\b|\b[A-Z]{2,4}-\d{3}\b|"
    r"\bdon'?t\s+worry\b|\bcalm\s+down\b|\d+(\.\d+)?\s?%|\babout\s+₹|\baround\s+₹|\broughly\b|"
    r"kareeb|thoda\s+sa\s+kam|konchem\s+takkuva|करीब|क़रीब|थोड़ा\s+सा\s+कम|కొంచెం\s+తక్కువ", re.I)
NEVER_SAY_SEGMENT: dict[str, re.Pattern] = {
    "S1": re.compile(r"score\s+is\s+low|bad\s+credit|credit\s+card|take\s+a\s+loan", re.I),
    "S3": re.compile(r"defaulter|should\s+have\s+paid|gold\s+loan|credit\s+card|new\s+loan|top-?up", re.I),
    "S4": re.compile(r"top-?up|gold\s+loan|another\s+loan|hide|income\s+needed", re.I),
    "S5": re.compile(r"worse|risky|form\s*16.*shop", re.I),
    "S6": re.compile(r"pincode|pin\s+code|your\s+area|improve\s+your\s+score", re.I),
    "S7": re.compile(r"fraud|flag|suspicious|blacklist|duplicate|mismatch|probably\s+nothing", re.I),
    "S8": re.compile(r"reject|should\s+take\s+it|limited\s+time|today\s+only|expires", re.I),
}


def never_say_hits(text: str, segment: str | None = None) -> list[str]:
    hits = [m.group(0) for m in NEVER_SAY_ALL.finditer(text or "")]
    pat = NEVER_SAY_SEGMENT.get(segment or "")
    if pat:
        hits += [m.group(0) for m in pat.finditer(text or "")]
    return hits


# --------------------------------------------------------------------------------------------- fact card
_KFS_KEYS = ("apr", "interest_rate", "processing_fee", "total_payable", "kfs_ref")


def fact_card(record: Mapping, explain: ExplainProfile | None, locale: str = "en") -> dict:
    """Decision facts for one locale, built only from the record (facts.py wording, exact ₹ strings)."""
    rec = dict(record or {})
    cats = [c["key"] for c in F.reason_categories(rec)]
    offer = F.borrower_offer(rec)
    off = None
    if offer and offer.get("amount"):
        off = {"amount": speak_amount(offer["amount"]),
               "emi": speak_amount(offer["emi"]) if offer.get("emi") else None,
               "months": int(offer["tenure_months"]) if offer.get("tenure_months") else None,
               "counter_offer": bool(offer.get("counter_offer")),
               "raw": {k: offer.get(k) for k in ("amount", "emi", "tenure_months")}}
    kfs_src = dict(rec.get("kfs") or {})
    kfs = {k: kfs_src[k] for k in _KFS_KEYS if k in kfs_src and kfs_src[k] is not None}  # stored values only
    income = explain.income if explain else "UNKNOWN"
    return {"locale": locale, "outcome": F.outcome_plain(rec, locale), "categories": cats,
            "primary_category": cats[0] if cats else None,
            "reasons": [F.reason_plain(k, locale) for k in cats],
            "guidance": [g["general_guidance"] for g in F.guidance_for(rec, locale, limit=2)],
            "segment": explain.primary if explain else None, "offer": off, "kfs": kfs,
            "checklist": list(CHECKLIST.get(income, CHECKLIST["UNKNOWN"]).get(locale)
                              or CHECKLIST.get(income, CHECKLIST["UNKNOWN"])["en"])}


def _l(table: Mapping[str, str], locale: str) -> str:
    return table.get(locale) or table["en"]


def card_numbers(card: Mapping) -> set[str]:
    """The exact amount strings and months a card allows (for invariance checks)."""
    o = card.get("offer") or {}
    return {x for x in (o.get("amount"), o.get("emi"), str(o["months"]) if o.get("months") else None) if x}


# --------------------------------------------------------------------------------------------- leads
INTENTS = ("explain", "offer", "improve", "documents", "human")


def lead(intent: str, card: Mapping, tier: str, locale: str = "en", simplify: bool = False) -> list[str]:
    """Deterministic sentences for one intent from the fact card. Tier decides how many facts go in this turn and
    in which words around them; the facts themselves are the card's, unchanged."""
    seg = card.get("segment")
    brief = tier == "brief"
    if intent == "human":
        return [line("handover_ack", locale)]
    if seg == "S7" and intent in ("explain", "improve", "offer"):
        out = [_l(S7_GENERIC, locale)]
        return out if brief else [_l(S7_NOT_ACCUSATION, locale)] + out
    if intent == "offer" or (intent == "explain" and seg == "S8"):
        o = card.get("offer")
        if not o:
            return [_l(NO_OFFER, locale)]
        if brief or simplify:
            return [_l(OFFER_BRIEF, locale).format(amount=o["amount"], emi=o["emi"], months=o["months"])]
        parts = [s.format(amount=o["amount"], emi=o["emi"], months=o["months"]) for s in OFFER_FULL[locale]
                 if "{emi}" not in s or o.get("emi")]
        if tier == "plain":
            parts.insert(2, _l(OFFER_REPEAT, locale).format(amount=o["amount"]))
        return parts
    if intent == "documents":
        items = card.get("checklist") or []
        head = [] if brief else [SEGMENT_LINE["S5"][locale]]
        return head + items[: (1 if brief or tier == "distress" else 2)]
    if intent == "improve":
        steps = list(card.get("guidance") or [])
        seg_line = SEGMENT_LINE.get(seg or "", {}).get(locale)
        if seg in ("S1", "S3", "S6") and seg_line and not brief:
            steps = [seg_line] + steps
        return steps[:1] if brief or simplify else steps[:2]
    # explain
    reason = (card.get("reasons") or [F.reason_plain("OTHER", locale)])[0]
    out = [_l(MAIN_REASON, locale).format(reason=reason)]
    seg_line = SEGMENT_LINE.get(seg or "", {}).get(locale)
    if seg_line and not brief and seg == "S6":
        out.insert(0, seg_line)                  # S6 says "a rule for this loan, not a judgment of you" first
    elif seg_line and not brief and seg != "S5":
        out.append(seg_line)
    if card.get("offer") and seg != "S8" and not brief:
        o = card["offer"]
        out.append(_l(OFFER_BRIEF, locale).format(amount=o["amount"], emi=o["emi"], months=o["months"]))
    return out[:1] if simplify else out


def check_line(locale: str = "en") -> str:
    return _l(CHECK, locale)
