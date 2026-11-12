"""Spoken numbers for Indian-language TTS (apps/voice/spoken_numbers.py) and its hook into the Timbre request body.

The verbaliser itself needs nothing but the repo. The hook tests need `livekit-plugins-gnani` and skip without it.
Telugu and Hindi expectations were written for native review; the other seven languages are marked in the module as
not yet native-reviewed, so their tests pin the current tables rather than vouch for them.
"""
import json
import sys
from pathlib import Path

import pytest

from loan_advisory.agent import settings as S

ROOT = Path(__file__).resolve().parents[1]
VOICE = ROOT / "apps" / "voice"
sys.path.insert(0, str(VOICE))

import spoken_numbers as sn  # noqa: E402
from spoken_numbers import verbalise  # noqa: E402


# ------------------------------------------------------------------------------------------------------- Telugu
TE = {
    0: "సున్నా", 7: "ఏడు", 10: "పది", 11: "పదకొండు", 19: "పందొమ్మిది", 20: "ఇరవై", 21: "ఇరవై ఒకటి",
    99: "తొంభై తొమ్మిది", 100: "వంద", 101: "నూట ఒకటి", 999: "తొమ్మిది వందల తొంభై తొమ్మిది", 1000: "వెయ్యి",
    1001: "వెయ్యి ఒకటి", 4100: "నాలుగు వేల వంద", 10000: "పది వేలు", 14400: "పద్నాలుగు వేల నాలుగు వందలు",
    18500: "పద్దెనిమిది వేల ఐదు వందలు", 32000: "ముప్పై రెండు వేలు",
    99999: "తొంభై తొమ్మిది వేల తొమ్మిది వందల తొంభై తొమ్మిది", 100000: "ఒక లక్ష", 300000: "మూడు లక్షలు",
    1234567: "పన్నెండు లక్షల ముప్పై నాలుగు వేల ఐదు వందల అరవై ఏడు", 10000000: "ఒక కోటి",
}

HI = {
    0: "शून्य", 7: "सात", 10: "दस", 11: "ग्यारह", 19: "उन्नीस", 20: "बीस", 21: "इक्कीस", 99: "निन्यानवे",
    100: "एक सौ", 101: "एक सौ एक", 999: "नौ सौ निन्यानवे", 1000: "एक हज़ार", 1001: "एक हज़ार एक",
    4100: "चार हज़ार एक सौ", 10000: "दस हज़ार", 14400: "चौदह हज़ार चार सौ", 18500: "अठारह हज़ार पाँच सौ",
    99999: "निन्यानवे हज़ार नौ सौ निन्यानवे", 100000: "एक लाख", 300000: "तीन लाख",
    1234567: "बारह लाख चौंतीस हज़ार पाँच सौ सड़सठ", 10000000: "एक करोड़",
}


@pytest.mark.parametrize("n,words", sorted(TE.items()))
def test_telugu_numbers(n, words):
    assert sn.telugu_number(n) == words


@pytest.mark.parametrize("n,words", sorted(HI.items()))
def test_hindi_numbers(n, words):
    assert sn.hindi_number(n) == words


def test_hindi_table_is_complete():
    assert len(sn.HI_0_99) == 100 and len(set(sn.HI_0_99)) == 100


@pytest.mark.parametrize("text,te,hi", [
    ("4,100", "నాలుగు వేల వంద", "चार हज़ार एक सौ"),
    ("1,00,000", "ఒక లక్ష", "एक लाख"),
    ("12,34,567", "పన్నెండు లక్షల ముప్పై నాలుగు వేల ఐదు వందల అరవై ఏడు", "बारह लाख चौंतीस हज़ार पाँच सौ सड़सठ"),
    ("1,00,00,000", "ఒక కోటి", "एक करोड़"),
    ("300,000", "మూడు లక్షలు", "तीन लाख"),                     # western commas
    ("58%", "యాభై ఎనిమిది శాతం", "अट्ठावन प्रतिशत"),
    ("10.5%", "పది పాయింట్ ఐదు శాతం", "दस दशमलव पाँच प्रतिशत"),
    ("3.14", "మూడు పాయింట్ ఒకటి నాలుగు", "तीन दशमलव एक चार"),
])
def test_plain_numbers_and_percent_in_text(text, te, hi):
    assert verbalise(text, "te-IN") == te and verbalise(text, "hi-IN") == hi


@pytest.mark.parametrize("text,te,hi", [
    ("₹4,100", "నాలుగు వేల వంద రూపాయలు", "चार हज़ार एक सौ रुपये"),
    ("Rs. 3,00,000", "మూడు లక్షల రూపాయలు", "तीन लाख रुपये"),
    ("Rs 18,500", "పద్దెనిమిది వేల ఐదు వందల రూపాయలు", "अठारह हज़ार पाँच सौ रुपये"),
    ("INR 300,000", "మూడు లక్షల రూపాయలు", "तीन लाख रुपये"),
    ("రూ.14,400", "పద్నాలుగు వేల నాలుగు వందల రూపాయలు", "चौदह हज़ार चार सौ रुपये"),
    ("₹4,100.50", "నాలుగు వేల వంద రూపాయల యాభై పైసలు", "चार हज़ार एक सौ रुपये पचास पैसे"),
    ("₹1", "ఒక రూపాయి", "एक रुपया"),
    ("₹0.50", "యాభై పైసలు", "पचास पैसे"),
    ("₹3.5 lakh", "మూడు లక్షల యాభై వేల రూపాయలు", "तीन लाख पचास हज़ार रुपये"),
    ("₹2 crore", "రెండు కోట్ల రూపాయలు", "दो करोड़ रुपये"),
])
def test_rupee_amounts(text, te, hi):
    assert verbalise(text, "te-IN") == te and verbalise(text, "hi-IN") == hi


def test_telugu_number_before_rupee_word_takes_the_oblique_form():
    assert verbalise("3,00,000 రూపాయలు", "te-IN") == "మూడు లక్షల రూపాయలు"
    assert verbalise("21,000", "te") == "ఇరవై ఒక్క వేలు"


@pytest.mark.parametrize("text", [
    "APP-TEST-002", "LN2024-0042", "A12", "id 12/05/2026", "at 10:30", "call 9876543210", "+91 98480 12345x",
    "OTP 0423", "v1.2.3", "user@42",
])
def test_ids_codes_and_phones_are_untouched(text):
    for lang in ("te-IN", "hi-IN", "ta-IN", "bn-IN"):
        assert verbalise(text, lang) == text


@pytest.mark.parametrize("lang", ["en-IN", "en", "auto", "hi-en", "", None, "fr-FR"])
def test_english_and_other_codes_pass_through(lang):
    text = "Your EMI is ₹18,500 at 10.5% for 36 months."
    assert verbalise(text, lang) == text


def test_mixed_sentence_changes_only_the_numbers():
    te = "మీ EMI ₹18,500, వడ్డీ 10.5%. Ref APP-TEST-002, 36 నెలలు."
    assert verbalise(te, "te-IN") == ("మీ EMI పద్దెనిమిది వేల ఐదు వందల రూపాయలు, వడ్డీ పది పాయింట్ ఐదు శాతం. "
                                      "Ref APP-TEST-002, ముప్పై ఆరు నెలలు.")
    loan_terms = "మీ loan application ను approve చేయలేము. EMI ₹18,500."
    assert verbalise(loan_terms, "te-IN") == "మీ loan application ను approve చేయలేము. EMI పద్దెనిమిది వేల ఐదు వందల రూపాయలు."
    hi = "आपकी EMI ₹18,500 है, ब्याज 58% (APP-TEST-002)।"
    assert verbalise(hi, "hi-IN") == "आपकी EMI अठारह हज़ार पाँच सौ रुपये है, ब्याज अट्ठावन प्रतिशत (APP-TEST-002)।"


def test_text_without_digits_is_returned_as_is():
    text = "నమస్కారం, మీ లోన్ గురించి చెప్పండి."
    assert verbalise(text, "te-IN") is text


# ------------------------------------------------------------------------- the other seven (not yet reviewed)
OTHERS = {
    "ta-IN": ("நான்காயிரத்து நூறு", "பதினெட்டாயிரத்து ஐந்நூறு", "மூன்று லட்சம்", "ஐம்பத்தெட்டு சதவீதம்",
              "நான்காயிரத்து நூறு ரூபாய்"),
    "kn-IN": ("ನಾಲ್ಕು ಸಾವಿರದ ನೂರು", "ಹದಿನೆಂಟು ಸಾವಿರದ ಐನೂರು", "ಮೂರು ಲಕ್ಷ", "ಶೇಕಡಾ ಐವತ್ತೆಂಟು",
              "ನಾಲ್ಕು ಸಾವಿರದ ನೂರು ರೂಪಾಯಿ"),
    "ml-IN": ("നാലായിരത്തി നൂറ്", "പതിനെട്ടായിരത്തി അഞ്ഞൂറ്", "മൂന്ന് ലക്ഷം", "അമ്പത്തിയെട്ട് ശതമാനം",
              "നാലായിരത്തി നൂറ് രൂപ"),
    "mr-IN": ("चार हजार शंभर", "अठरा हजार पाचशे", "तीन लाख", "अठ्ठावन्न टक्के", "चार हजार शंभर रुपये"),
    "bn-IN": ("চার হাজার একশো", "আঠারো হাজার পাঁচশো", "তিন লাখ", "আটান্ন শতাংশ", "চার হাজার একশো টাকা"),
    "gu-IN": ("ચાર હજાર એકસો", "અઢાર હજાર પાંચસો", "ત્રણ લાખ", "અઠ્ઠાવન ટકા", "ચાર હજાર એકસો રૂપિયા"),
    "pa-IN": ("ਚਾਰ ਹਜ਼ਾਰ ਇੱਕ ਸੌ", "ਅਠਾਰਾਂ ਹਜ਼ਾਰ ਪੰਜ ਸੌ", "ਤਿੰਨ ਲੱਖ", "ਅਠਵੰਜਾ ਪ੍ਰਤੀਸ਼ਤ", "ਚਾਰ ਹਜ਼ਾਰ ਇੱਕ ਸੌ ਰੁਪਏ"),
}


@pytest.mark.parametrize("lang", sorted(OTHERS))
def test_other_languages_core_amounts(lang):
    n4100, n18500, n300000, pct58, rs4100 = OTHERS[lang]
    assert verbalise("4,100", lang) == n4100
    assert verbalise("18,500", lang) == n18500
    assert verbalise("3,00,000", lang) == n300000
    assert verbalise("58%", lang) == pct58
    assert verbalise("₹4,100", lang) == rs4100


@pytest.mark.parametrize("lang", sorted(sn.LANGUAGES))
def test_every_language_covers_0_to_a_crore_without_gaps(lang):
    say = sn.LANGUAGES[lang].number
    seen = set()
    for n in list(range(0, 1000)) + [1000, 1001, 4100, 10000, 21000, 32000, 99999, 100000, 1234567, 10000000,
                                     12_34_56_789, 150_00_00_000]:
        words = say(n)
        assert words and words == words.strip() and "  " not in words and not any(c.isdigit() for c in words), (n, words)
        seen.add(words)
    assert len(seen) == 1012  # every number has its own words


def test_north_tables_are_complete():
    for table in (sn.HI_0_99, sn.MR_0_99, sn.GU_0_99, sn.PA_0_99, sn.BN_0_99):
        assert len(table) == 100 and len(set(table)) == 100


def test_module_marks_unreviewed_languages():
    doc = sn.__doc__
    assert "NOT YET NATIVE-REVIEWED" in doc
    for name in ("Tamil", "Kannada", "Malayalam", "Marathi", "Bengali", "Gujarati", "Punjabi"):
        assert name in doc


# ------------------------------------------------------------------------------------------------------ settings
def test_verbalise_flag_defaults_on_and_parses_off():
    assert S.GnaniSpeechSettings().tts_verbalise_numbers is True
    assert S.GnaniSpeechSettings.from_env({}).tts_verbalise_numbers is True
    assert S.GnaniSpeechSettings.from_env({"GNANI_TTS_VERBALISE_NUMBERS": "off"}).tts_verbalise_numbers is False
    assert S.GnaniSpeechSettings.from_env({"GNANI_TTS_VERBALISE_NUMBERS": "0"}).tts_verbalise_numbers is False
    with pytest.raises(ValueError):
        S.GnaniSpeechSettings.from_env({"GNANI_TTS_VERBALISE_NUMBERS": "maybe"})


# ------------------------------------------------------------------------------- Timbre request body (mocked)
@pytest.fixture
def gs(monkeypatch):
    pytest.importorskip("livekit.plugins.gnani")
    monkeypatch.setenv("GNANI_API_KEY", "test-dummy-gnani-key")
    monkeypatch.syspath_prepend(str(VOICE))
    sys.modules.pop("gnani_speech", None)
    import gnani_speech
    return gnani_speech


def test_tts_body_carries_words_for_telugu_and_hindi(gs):
    from livekit.plugins.gnani import tts as gt
    te = gs.build_tts(S.GnaniSpeechSettings(), "te")
    assert gt._build_payload(te._opts, "మీ EMI ₹4,100.")["text"] == "మీ EMI నాలుగు వేల వంద రూపాయలు."
    hi = gs.build_tts(S.GnaniSpeechSettings(), "hi")
    assert gt._build_payload(hi._opts, "EMI ₹4,100")["text"] == "EMI चार हज़ार एक सौ रुपये"


def test_tts_body_follows_a_language_switch_and_leaves_english(gs):
    from livekit.plugins.gnani import tts as gt
    t = gs.build_tts(S.GnaniSpeechSettings(), "en")
    assert gt._build_payload(t._opts, "EMI ₹4,100")["text"] == "EMI ₹4,100"
    t.update_options(language="ta-IN")
    assert gt._build_payload(t._opts, "₹4,100")["text"] == "நான்காயிரத்து நூறு ரூபாய்"


def test_tts_body_keeps_digits_when_the_flag_is_off(gs):
    from livekit.plugins.gnani import tts as gt
    t = gs.build_tts(S.GnaniSpeechSettings(tts_verbalise_numbers=False), "te")
    body = gt._build_payload(t._opts, "₹4,100")
    assert body["text"] == "₹4,100" and "verbalise_numbers" not in json.dumps(body)
    plain = gt.GnaniTTSOptions(api_key="x", model="timbre-v3.0", voice="Shivani", language="te-IN")
    assert gt._build_payload(plain, "₹4,100")["text"] == "₹4,100"  # non-Timbre options untouched
