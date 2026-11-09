"""Spoken numbers for Indian-language TTS: "₹4,100" -> "నాలుగు వేల వంద రూపాయలు" / "चार हज़ार एक सौ रुपये".

Timbre reads digits inside an Indian-language sentence unevenly (often in English, sometimes digit by digit), so
rupee amounts, percentages and plain numbers are written out as words before the text is synthesised. This is
deterministic: a lookup table for 0-99 and the Indian grouping (hundred, thousand, lakh, crore) on top. No LLM,
no dependencies.

Languages: the nine Indian languages Gnani Timbre speaks besides English. en-IN (and "auto", "hi-en") pass through
unchanged, because Timbre reads English numbers itself.

    te-IN Telugu, hi-IN Hindi          native-reviewed forms (Saral's evaluated languages)
    ta-IN Tamil, kn-IN Kannada, ml-IN Malayalam, mr-IN Marathi, bn-IN Bengali, gu-IN Gujarati, pa-IN Punjabi
                                       NOT YET NATIVE-REVIEWED: tables and joining rules written from standard
                                       references; a native speaker should check them before relying on them

It is applied to the TTS request body only (gnani_speech.timbre_payload), never to the chat transcript, the logs or
the LLM context.

Left alone on purpose:
* anything that looks like an ID or code: digits touching letters, '-', '_', '/', ':', '@' or '#' (APP-TEST-002,
  A12, 12/05/2026, 10:30);
* phone numbers: 10 or more digits, alone or grouped with spaces or hyphens (+91 98480 12345);
* numbers with a leading zero (007, an OTP such as 0423);
* every character that is not part of a number.

Telugu forms, for a native reviewer:
* plain numbers end in the plural: వందలు, వేలు, లక్షలు, కోట్లు (32,000 -> ముప్పై రెండు వేలు, 3,00,000 -> మూడు లక్షలు);
* a number followed by more of the number, or by a noun (రూపాయలు, శాతం), takes the oblique form: వందల, వేల, లక్షల,
  కోట్ల (14,400 -> పద్నాలుగు వేల నాలుగు వందలు; ₹3,00,000 -> మూడు లక్షల రూపాయలు);
* one hundred is వంద, and నూట when more follows (101 -> నూట ఒకటి); one thousand is వెయ్యి;
  one lakh / crore is ఒక లక్ష / ఒక కోటి;
* ఒకటి before a noun becomes ఒక (alone) or ఒక్క (after a tens word): 21,000 -> ఇరవై ఒక్క వేలు, ₹1 -> ఒక రూపాయి.

Joining in the not-yet-reviewed languages:
* Tamil / Malayalam: a vowel-initial unit fuses onto its tens or hundreds word (இருபத்து + ஒன்று -> இருபத்தொன்று;
  അമ്പത്തി + എട്ട് -> അമ്പത്തിയെട്ട്); thousands are fused (நான்காயிரம், നാലായിരം) and take a linking ending before
  more of the number (நான்காயிரத்து நூறு, നാലായിരത്തി നൂറ്), as does lakh (லட்சத்து, ലക്ഷത്തി) and Tamil crore (கோடியே).
* Kannada: units fuse onto the tens word (ಐವತ್ತೆಂಟು); ನೂರು, ಸಾವಿರ, ಲಕ್ಷ link as ನೂರ / ಸಾವಿರದ / ಲಕ್ಷದ; the
  percent word comes first (ಶೇಕಡಾ ಐವತ್ತೆಂಟು).
* Marathi, Bengali, Gujarati, Punjabi: full 0-99 tables; hundreds fuse (पाचशे, পাঁচশো, પાંચસો) except Punjabi (ਪੰਜ ਸੌ).
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

__all__ = ["verbalise", "number_words", "telugu_number", "hindi_number", "LANGUAGES"]

# --------------------------------------------------------------------------------------------------------- Telugu
TE_0_99 = (
    "సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది",          # 0-9
    "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పందొమ్మిది",  # 10-19
)
TE_TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై"}

# plain (end of the number) -> oblique (more follows, or a noun follows)
TE_OBLIQUE = {"వందలు": "వందల", "వేలు": "వేల", "లక్షలు": "లక్షల", "కోట్లు": "కోట్ల", "ఒకటి": "ఒక్క"}


def _te_below_100(n: int) -> str:
    if n < 20:
        return TE_0_99[n]
    tens, ones = divmod(n, 10)
    return TE_TENS[tens] if ones == 0 else f"{TE_TENS[tens]} {TE_0_99[ones]}"


def _te_attributive(n: int) -> str:
    """The number as it stands before a noun or a larger unit: ఒక, ఇరవై ఒక్క, రెండు వందల, మూడు లక్షల ..."""
    if n == 1:
        return "ఒక"
    words = telugu_number(n).split(" ")
    words[-1] = TE_OBLIQUE.get(words[-1], words[-1])
    return " ".join(words)


def _te_scaled(n: int, unit: int, one: str, plural: str, oblique: str) -> str:
    head, rest = divmod(n, unit)
    if head == 1:
        lead = one
    else:
        lead = f"{_te_attributive(head)} {plural if rest == 0 else oblique}"
    return lead if rest == 0 else f"{lead} {telugu_number(rest)}"


def telugu_number(n: int) -> str:
    """Telugu words for a non-negative integer, Indian grouping."""
    if n < 0:
        raise ValueError("negative numbers are not verbalised")
    if n < 100:
        return _te_below_100(n)
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        if hundreds == 1:
            return "వంద" if rest == 0 else f"నూట {_te_below_100(rest)}"
        return f"{TE_0_99[hundreds]} వందలు" if rest == 0 else f"{TE_0_99[hundreds]} వందల {_te_below_100(rest)}"
    if n < 1_00_000:
        return _te_scaled(n, 1000, "వెయ్యి", "వేలు", "వేల")
    if n < 1_00_00_000:
        return _te_scaled(n, 1_00_000, "ఒక లక్ష", "లక్షలు", "లక్షల")
    return _te_scaled(n, 1_00_00_000, "ఒక కోటి", "కోట్లు", "కోట్ల")


# ---------------------------------------------------------------------------------------------------------- Hindi
# Hindi 1-99 are irregular, so the whole table is written out.
HI_0_99 = (
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",                          # 0-9
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",          # 10-19
    "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",    # 20-29
    "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",  # 30-39
    "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",  # 40-49
    "पचास", "इक्यावन", "बावन", "तिरेपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",    # 50-59
    "साठ", "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",        # 60-69
    "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उन्यासी",  # 70-79
    "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी",  # 80-89
    "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",  # 90-99
)
assert len(HI_0_99) == 100


def hindi_number(n: int) -> str:
    """Hindi words for a non-negative integer, Indian grouping (सौ, हज़ार, लाख, करोड़)."""
    if n < 0:
        raise ValueError("negative numbers are not verbalised")
    if n < 100:
        return HI_0_99[n]
    for unit, word in ((1_00_00_000, "करोड़"), (1_00_000, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        if n >= unit:
            head, rest = divmod(n, unit)
            lead = f"{hindi_number(head)} {word}"
            return lead if rest == 0 else f"{lead} {hindi_number(rest)}"
    raise AssertionError("unreachable")



# ------------------------------------------------------------------------------- shared helpers for the rest
def _north(n: int, table: tuple, hundreds, units: tuple) -> str:
    """Hindi-style grouping: <head> <unit> <rest>, with a language's own hundreds rule."""
    if n < 100:
        return table[n]
    if n < 1000:
        return hundreds(n, False)
    for unit, word in units:
        if n >= unit:
            head, rest = divmod(n, unit)
            lead = f"{_north(head, table, hundreds, units)} {word}"
            if rest == 0:
                return lead
            tail = hundreds(rest, True) if 100 <= rest < 1000 else _north(rest, table, hundreds, units)
            return f"{lead} {tail}"
    raise AssertionError("unreachable")


def _fuse(stem: str, unit: str, signs: dict[str, str], *, glide: str = "") -> str:
    """Join a linking stem and a unit word. A vowel-initial unit loses its vowel letter, which becomes a vowel sign on
    the stem's last consonant (Tamil இருபத்த + ஒன்று -> இருபத்தொன்று), or on a glide consonant (Malayalam യ)."""
    first = unit[0]
    if first in signs:
        return stem + glide + signs[first] + unit[1:]
    return stem + unit


# ---------------------------------------------------------------------------------- Marathi (NOT YET REVIEWED)
MR_0_99 = (
    "शून्य", "एक", "दोन", "तीन", "चार", "पाच", "सहा", "सात", "आठ", "नऊ",                               # 0-9
    "दहा", "अकरा", "बारा", "तेरा", "चौदा", "पंधरा", "सोळा", "सतरा", "अठरा", "एकोणीस",                     # 10-19
    "वीस", "एकवीस", "बावीस", "तेवीस", "चोवीस", "पंचवीस", "सव्वीस", "सत्तावीस", "अठ्ठावीस", "एकोणतीस",      # 20-29
    "तीस", "एकतीस", "बत्तीस", "तेहेतीस", "चौतीस", "पस्तीस", "छत्तीस", "सदतीस", "अडतीस", "एकोणचाळीस",       # 30-39
    "चाळीस", "एक्केचाळीस", "बेचाळीस", "त्रेचाळीस", "चव्वेचाळीस", "पंचेचाळीस", "सेहेचाळीस", "सत्तेचाळीस",
    "अठ्ठेचाळीस", "एकोणपन्नास",                                                                          # 40-49
    "पन्नास", "एक्कावन्न", "बावन्न", "त्रेपन्न", "चोपन्न", "पंचावन्न", "छप्पन्न", "सत्तावन्न", "अठ्ठावन्न",
    "एकोणसाठ",                                                                                         # 50-59
    "साठ", "एकसष्ट", "बासष्ट", "त्रेसष्ट", "चौसष्ट", "पासष्ट", "सहासष्ट", "सदुसष्ट", "अडुसष्ट", "एकोणसत्तर",   # 60-69
    "सत्तर", "एकाहत्तर", "बाहत्तर", "त्र्याहत्तर", "चौऱ्याहत्तर", "पंच्याहत्तर", "शहात्तर", "सत्याहत्तर",
    "अठ्ठ्याहत्तर", "एकोणऐंशी",                                                                           # 70-79
    "ऐंशी", "एक्क्याऐंशी", "ब्याऐंशी", "त्र्याऐंशी", "चौऱ्याऐंशी", "पंच्याऐंशी", "शहाऐंशी", "सत्त्याऐंशी",
    "अठ्ठ्याऐंशी", "एकोणनव्वद",                                                                           # 80-89
    "नव्वद", "एक्क्याण्णव", "ब्याण्णव", "त्र्याण्णव", "चौऱ्याण्णव", "पंच्याण्णव", "शहाण्णव", "सत्त्याण्णव",
    "अठ्ठ्याण्णव", "नव्व्याण्णव",                                                                          # 90-99
)


def _mr_hundreds(n: int, inner: bool) -> str:
    h, rest = divmod(n, 100)
    lead = ("शंभर" if rest == 0 else "एकशे") if h == 1 else MR_0_99[h] + "शे"   # 100 शंभर, 101 एकशे एक, 500 पाचशे
    return lead if rest == 0 else f"{lead} {MR_0_99[rest]}"


def marathi_number(n: int) -> str:
    return _north(n, MR_0_99, _mr_hundreds, ((1_00_00_000, "कोटी"), (1_00_000, "लाख"), (1000, "हजार")))


# --------------------------------------------------------------------------------- Gujarati (NOT YET REVIEWED)
GU_0_99 = (
    "શૂન્ય", "એક", "બે", "ત્રણ", "ચાર", "પાંચ", "છ", "સાત", "આઠ", "નવ",                                   # 0-9
    "દસ", "અગિયાર", "બાર", "તેર", "ચૌદ", "પંદર", "સોળ", "સત્તર", "અઢાર", "ઓગણીસ",                          # 10-19
    "વીસ", "એકવીસ", "બાવીસ", "તેવીસ", "ચોવીસ", "પચ્ચીસ", "છવ્વીસ", "સત્તાવીસ", "અઠ્ઠાવીસ", "ઓગણત્રીસ",      # 20-29
    "ત્રીસ", "એકત્રીસ", "બત્રીસ", "તેત્રીસ", "ચોત્રીસ", "પાંત્રીસ", "છત્રીસ", "સાડત્રીસ", "આડત્રીસ", "ઓગણચાલીસ",  # 30-39
    "ચાલીસ", "એકતાલીસ", "બેતાલીસ", "તેતાલીસ", "ચુંમાલીસ", "પિસ્તાલીસ", "છેતાલીસ", "સુડતાલીસ", "અડતાલીસ",
    "ઓગણપચાસ",                                                                                        # 40-49
    "પચાસ", "એકાવન", "બાવન", "ત્રેપન", "ચોપન", "પંચાવન", "છપ્પન", "સત્તાવન", "અઠ્ઠાવન", "ઓગણસાઠ",          # 50-59
    "સાઠ", "એકસઠ", "બાસઠ", "ત્રેસઠ", "ચોસઠ", "પાંસઠ", "છાસઠ", "સડસઠ", "અડસઠ", "અગણોસિત્તેર",              # 60-69
    "સિત્તેર", "એકોતેર", "બોતેર", "તોતેર", "ચુમોતેર", "પંચોતેર", "છોતેર", "સિત્યોતેર", "ઇઠ્યોતેર", "ઓગણાએંસી",  # 70-79
    "એંસી", "એક્યાસી", "બ્યાસી", "ત્યાસી", "ચોર્યાસી", "પંચાસી", "છ્યાસી", "સિત્યાસી", "ઈઠ્યાસી", "નેવ્યાસી",    # 80-89
    "નેવું", "એકાણું", "બાણું", "ત્રાણું", "ચોરાણું", "પંચાણું", "છન્નું", "સત્તાણું", "અઠ્ઠાણું", "નવ્વાણું",       # 90-99
)


def _gu_hundreds(n: int, inner: bool) -> str:
    h, rest = divmod(n, 100)
    lead = ("સો" if (rest == 0 and not inner) else "એકસો") if h == 1 else GU_0_99[h] + "સો"   # 500 પાંચસો
    return lead if rest == 0 else f"{lead} {GU_0_99[rest]}"


def gujarati_number(n: int) -> str:
    return _north(n, GU_0_99, _gu_hundreds, ((1_00_00_000, "કરોડ"), (1_00_000, "લાખ"), (1000, "હજાર")))


# --------------------------------------------------------------------------------- Punjabi (NOT YET REVIEWED)
PA_0_99 = (
    "ਸਿਫ਼ਰ", "ਇੱਕ", "ਦੋ", "ਤਿੰਨ", "ਚਾਰ", "ਪੰਜ", "ਛੇ", "ਸੱਤ", "ਅੱਠ", "ਨੌਂ",                                # 0-9
    "ਦਸ", "ਗਿਆਰਾਂ", "ਬਾਰਾਂ", "ਤੇਰਾਂ", "ਚੌਦਾਂ", "ਪੰਦਰਾਂ", "ਸੋਲਾਂ", "ਸਤਾਰਾਂ", "ਅਠਾਰਾਂ", "ਉੱਨੀ",                  # 10-19
    "ਵੀਹ", "ਇੱਕੀ", "ਬਾਈ", "ਤੇਈ", "ਚੌਵੀ", "ਪੱਚੀ", "ਛੱਬੀ", "ਸਤਾਈ", "ਅਠਾਈ", "ਉਨੱਤੀ",                           # 20-29
    "ਤੀਹ", "ਇਕੱਤੀ", "ਬੱਤੀ", "ਤੇਤੀ", "ਚੌਂਤੀ", "ਪੈਂਤੀ", "ਛੱਤੀ", "ਸੈਂਤੀ", "ਅਠੱਤੀ", "ਉਨਤਾਲੀ",                     # 30-39
    "ਚਾਲੀ", "ਇਕਤਾਲੀ", "ਬਤਾਲੀ", "ਤਰਤਾਲੀ", "ਚੌਤਾਲੀ", "ਪੰਤਾਲੀ", "ਛਿਆਲੀ", "ਸੰਤਾਲੀ", "ਅਠਤਾਲੀ", "ਉਨੰਜਾ",         # 40-49
    "ਪੰਜਾਹ", "ਇਕਵੰਜਾ", "ਬਵੰਜਾ", "ਤਰਵੰਜਾ", "ਚਰਵੰਜਾ", "ਪਚਵੰਜਾ", "ਛਪੰਜਾ", "ਸਤਵੰਜਾ", "ਅਠਵੰਜਾ", "ਉਨਾਹਠ",       # 50-59
    "ਸੱਠ", "ਇਕਾਹਠ", "ਬਾਹਠ", "ਤਰੇਹਠ", "ਚੌਂਹਠ", "ਪੈਂਹਠ", "ਛਿਆਹਠ", "ਸਤਾਹਠ", "ਅਠਾਹਠ", "ਉਨੱਤਰ",               # 60-69
    "ਸੱਤਰ", "ਇਕਹੱਤਰ", "ਬਹੱਤਰ", "ਤਿਹੱਤਰ", "ਚੌਹੱਤਰ", "ਪਚੱਤਰ", "ਛਿਹੱਤਰ", "ਸਤੱਤਰ", "ਅਠੱਤਰ", "ਉਨਾਸੀ",          # 70-79
    "ਅੱਸੀ", "ਇਕਾਸੀ", "ਬਿਆਸੀ", "ਤਿਰਾਸੀ", "ਚੁਰਾਸੀ", "ਪਚਾਸੀ", "ਛਿਆਸੀ", "ਸਤਾਸੀ", "ਅਠਾਸੀ", "ਉਨੱਨਵੇਂ",           # 80-89
    "ਨੱਬੇ", "ਇਕਾਨਵੇਂ", "ਬਾਨਵੇਂ", "ਤਿਰਾਨਵੇਂ", "ਚੁਰਾਨਵੇਂ", "ਪਚਾਨਵੇਂ", "ਛਿਆਨਵੇਂ", "ਸਤਾਨਵੇਂ", "ਅਠਾਨਵੇਂ", "ਨੜਿੰਨਵੇਂ",  # 90-99
)


def _pa_hundreds(n: int, inner: bool) -> str:
    h, rest = divmod(n, 100)
    lead = f"{PA_0_99[h]} ਸੌ"
    return lead if rest == 0 else f"{lead} {PA_0_99[rest]}"


def punjabi_number(n: int) -> str:
    return _north(n, PA_0_99, _pa_hundreds, ((1_00_00_000, "ਕਰੋੜ"), (1_00_000, "ਲੱਖ"), (1000, "ਹਜ਼ਾਰ")))


# --------------------------------------------------------------------------------- Bengali (NOT YET REVIEWED)
BN_0_99 = (
    "শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়",                                   # 0-9
    "দশ", "এগারো", "বারো", "তেরো", "চোদ্দ", "পনেরো", "ষোলো", "সতেরো", "আঠারো", "উনিশ",                     # 10-19
    "কুড়ি", "একুশ", "বাইশ", "তেইশ", "চব্বিশ", "পঁচিশ", "ছাব্বিশ", "সাতাশ", "আটাশ", "ঊনত্রিশ",                # 20-29
    "ত্রিশ", "একত্রিশ", "বত্রিশ", "তেত্রিশ", "চৌত্রিশ", "পঁয়ত্রিশ", "ছত্রিশ", "সাঁইত্রিশ", "আটত্রিশ", "ঊনচল্লিশ",  # 30-39
    "চল্লিশ", "একচল্লিশ", "বিয়াল্লিশ", "তেতাল্লিশ", "চুয়াল্লিশ", "পঁয়তাল্লিশ", "ছেচল্লিশ", "সাতচল্লিশ", "আটচল্লিশ",
    "ঊনপঞ্চাশ",                                                                                         # 40-49
    "পঞ্চাশ", "একান্ন", "বাহান্ন", "তিপ্পান্ন", "চুয়ান্ন", "পঞ্চান্ন", "ছাপ্পান্ন", "সাতান্ন", "আটান্ন", "ঊনষাট",    # 50-59
    "ষাট", "একষট্টি", "বাষট্টি", "তেষট্টি", "চৌষট্টি", "পঁয়ষট্টি", "ছেষট্টি", "সাতষট্টি", "আটষট্টি", "ঊনসত্তর",     # 60-69
    "সত্তর", "একাত্তর", "বাহাত্তর", "তিয়াত্তর", "চুয়াত্তর", "পঁচাত্তর", "ছিয়াত্তর", "সাতাত্তর", "আটাত্তর", "ঊনআশি",  # 70-79
    "আশি", "একাশি", "বিরাশি", "তিরাশি", "চুরাশি", "পঁচাশি", "ছিয়াশি", "সাতাশি", "অষ্টআশি", "ঊননব্বই",            # 80-89
    "নব্বই", "একানব্বই", "বিরানব্বই", "তিরানব্বই", "চুরানব্বই", "পঁচানব্বই", "ছিয়ানব্বই", "সাতানব্বই", "আটানব্বই",
    "নিরানব্বই",                                                                                        # 90-99
)


def _bn_hundreds(n: int, inner: bool) -> str:
    h, rest = divmod(n, 100)
    lead = ("দু" if h == 2 else BN_0_99[h]) + "শো"   # একশো, দুশো, পাঁচশো
    return lead if rest == 0 else f"{lead} {BN_0_99[rest]}"


def bengali_number(n: int) -> str:
    return _north(n, BN_0_99, _bn_hundreds, ((1_00_00_000, "কোটি"), (1_00_000, "লাখ"), (1000, "হাজার")))


# ------------------------------------------------------------------------------------ Tamil (NOT YET REVIEWED)
TA_0_19 = ("பூஜ்ஜியம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது",
           "பத்து", "பதினொன்று", "பன்னிரண்டு", "பதின்மூன்று", "பதினான்கு", "பதினைந்து", "பதினாறு", "பதினேழு",
           "பதினெட்டு", "பத்தொன்பது")
TA_TENS = {2: "இருபது", 3: "முப்பது", 4: "நாற்பது", 5: "ஐம்பது", 6: "அறுபது", 7: "எழுபது", 8: "எண்பது", 9: "தொண்ணூறு"}
TA_TENS_LINK = {2: "இருபத்து", 3: "முப்பத்து", 4: "நாற்பத்து", 5: "ஐம்பத்து", 6: "அறுபத்து", 7: "எழுபத்து",
                8: "எண்பத்து", 9: "தொண்ணூற்று"}
TA_HUNDREDS = {1: "நூறு", 2: "இருநூறு", 3: "முந்நூறு", 4: "நானூறு", 5: "ஐந்நூறு", 6: "அறுநூறு", 7: "எழுநூறு",
               8: "எண்ணூறு", 9: "தொள்ளாயிரம்"}
# n thousand, fused (4,000 நான்காயிரம்); key 1 is the form inside a compound (21,000 இருபத்தோராயிரம்)
TA_THOUSANDS = {1: "ஓராயிரம்", 2: "இரண்டாயிரம்", 3: "மூவாயிரம்", 4: "நான்காயிரம்", 5: "ஐயாயிரம்", 6: "ஆறாயிரம்",
                7: "ஏழாயிரம்", 8: "எட்டாயிரம்", 9: "ஒன்பதாயிரம்", 10: "பத்தாயிரம்", 11: "பதினோராயிரம்",
                12: "பன்னிரண்டாயிரம்", 13: "பதின்மூன்றாயிரம்", 14: "பதினான்காயிரம்", 15: "பதினைந்தாயிரம்",
                16: "பதினாறாயிரம்", 17: "பதினேழாயிரம்", 18: "பதினெட்டாயிரம்", 19: "பத்தொன்பதாயிரம்",
                20: "இருபதாயிரம்", 30: "முப்பதாயிரம்", 40: "நாற்பதாயிரம்", 50: "ஐம்பதாயிரம்", 60: "அறுபதாயிரம்",
                70: "எழுபதாயிரம்", 80: "எண்பதாயிரம்", 90: "தொண்ணூறாயிரம்"}
TA_SIGNS = {"ஒ": "ொ", "ஓ": "ோ", "இ": "ி", "ஐ": "ை", "ஆ": "ா", "ஏ": "ே", "எ": "ெ"}


def _ta_join(link: str, unit: str) -> str:
    # இருபத்து + ஒன்று -> இருபத்தொன்று ; இருபத்து + மூன்று -> இருபத்து மூன்று
    return _fuse(link[:-1], unit, TA_SIGNS) if unit[0] in TA_SIGNS else f"{link} {unit}"


def _ta_below_1000(n: int) -> str:
    if n < 20:
        return TA_0_19[n]
    if n < 100:
        t, o = divmod(n, 10)
        return TA_TENS[t] if o == 0 else _ta_join(TA_TENS_LINK[t], TA_0_19[o])
    h, rest = divmod(n, 100)
    if rest == 0:
        return TA_HUNDREDS[h]
    link = TA_HUNDREDS[h][:-1] + "்று" if h < 9 else "தொள்ளாயிரத்து"   # நூறு -> நூற்று
    return _ta_join(link, _ta_below_1000(rest))


def tamil_number(n: int) -> str:
    if n < 1000:
        return _ta_below_1000(n)
    if n < 1_00_000:
        h, rest = divmod(n, 1000)
        if h == 1:
            lead = "ஆயிரம்"
        elif h < 20 or h % 10 == 0:
            lead = TA_THOUSANDS[h]
        else:
            lead = _ta_join(TA_TENS_LINK[h // 10], TA_THOUSANDS[h % 10])
        return lead if rest == 0 else f"{lead[:-2]}த்து {tamil_number(rest)}"   # ஆயிரம் -> ஆயிரத்து
    unit, word, link = (1_00_00_000, "கோடி", "கோடியே") if n >= 1_00_00_000 else (1_00_000, "லட்சம்", "லட்சத்து")
    h, rest = divmod(n, unit)
    head = "ஒரு" if h == 1 else tamil_number(h)
    return f"{head} {word}" if rest == 0 else f"{head} {link} {tamil_number(rest)}"


# --------------------------------------------------------------------------------- Kannada (NOT YET REVIEWED)
KN_0_19 = ("ಸೊನ್ನೆ", "ಒಂದು", "ಎರಡು", "ಮೂರು", "ನಾಲ್ಕು", "ಐದು", "ಆರು", "ಏಳು", "ಎಂಟು", "ಒಂಬತ್ತು",
           "ಹತ್ತು", "ಹನ್ನೊಂದು", "ಹನ್ನೆರಡು", "ಹದಿಮೂರು", "ಹದಿನಾಲ್ಕು", "ಹದಿನೈದು", "ಹದಿನಾರು", "ಹದಿನೇಳು",
           "ಹದಿನೆಂಟು", "ಹತ್ತೊಂಬತ್ತು")
KN_TENS = {2: "ಇಪ್ಪತ್ತು", 3: "ಮೂವತ್ತು", 4: "ನಲವತ್ತು", 5: "ಐವತ್ತು", 6: "ಅರವತ್ತು", 7: "ಎಪ್ಪತ್ತು", 8: "ಎಂಬತ್ತು",
           9: "ತೊಂಬತ್ತು"}
KN_HUNDREDS = {1: "ನೂರು", 2: "ಇನ್ನೂರು", 3: "ಮುನ್ನೂರು", 4: "ನಾನೂರು", 5: "ಐನೂರು", 6: "ಆರುನೂರು", 7: "ಏಳುನೂರು",
               8: "ಎಂಟುನೂರು", 9: "ಒಂಬೈನೂರು"}
KN_SIGNS = {"ಒ": "ೊ", "ಎ": "ೆ", "ಐ": "ೈ", "ಆ": "ಾ", "ಏ": "ೇ", "ಅ": ""}


def kannada_number(n: int) -> str:
    if n < 20:
        return KN_0_19[n]
    if n < 100:
        t, o = divmod(n, 10)
        # ಐವತ್ತು + ಎಂಟು -> ಐವತ್ತೆಂಟು ; ಇಪ್ಪತ್ತು + ಮೂರು -> ಇಪ್ಪತ್ತಮೂರು
        return KN_TENS[t] if o == 0 else _fuse(KN_TENS[t][:-1], KN_0_19[o], KN_SIGNS)
    if n < 1000:
        h, rest = divmod(n, 100)
        return KN_HUNDREDS[h] if rest == 0 else f"{KN_HUNDREDS[h][:-1]} {kannada_number(rest)}"   # ನೂರ ಒಂದು
    for unit, word, link in ((1_00_00_000, "ಕೋಟಿ", "ಕೋಟಿ"), (1_00_000, "ಲಕ್ಷ", "ಲಕ್ಷದ"), (1000, "ಸಾವಿರ", "ಸಾವಿರದ")):
        if n >= unit:
            h, rest = divmod(n, unit)
            return f"{kannada_number(h)} {word}" if rest == 0 else f"{kannada_number(h)} {link} {kannada_number(rest)}"
    raise AssertionError("unreachable")


# ------------------------------------------------------------------------------- Malayalam (NOT YET REVIEWED)
ML_0_19 = ("പൂജ്യം", "ഒന്ന്", "രണ്ട്", "മൂന്ന്", "നാല്", "അഞ്ച്", "ആറ്", "ഏഴ്", "എട്ട്", "ഒമ്പത്",
           "പത്ത്", "പതിനൊന്ന്", "പന്ത്രണ്ട്", "പതിമൂന്ന്", "പതിനാല്", "പതിനഞ്ച്", "പതിനാറ്", "പതിനേഴ്",
           "പതിനെട്ട്", "പത്തൊമ്പത്")
ML_TENS = {2: "ഇരുപത്", 3: "മുപ്പത്", 4: "നാൽപ്പത്", 5: "അമ്പത്", 6: "അറുപത്", 7: "എഴുപത്", 8: "എൺപത്", 9: "തൊണ്ണൂറ്"}
ML_TENS_LINK = {2: "ഇരുപത്തി", 3: "മുപ്പത്തി", 4: "നാൽപ്പത്തി", 5: "അമ്പത്തി", 6: "അറുപത്തി", 7: "എഴുപത്തി",
                8: "എൺപത്തി", 9: "തൊണ്ണൂറ്റി"}
ML_HUNDREDS = {1: "നൂറ്", 2: "ഇരുനൂറ്", 3: "മുന്നൂറ്", 4: "നാനൂറ്", 5: "അഞ്ഞൂറ്", 6: "അറുനൂറ്", 7: "എഴുനൂറ്",
               8: "എണ്ണൂറ്", 9: "തൊള്ളായിരം"}
ML_THOUSANDS = {1: "ഒന്നായിരം", 2: "രണ്ടായിരം", 3: "മൂവായിരം", 4: "നാലായിരം", 5: "അയ്യായിരം", 6: "ആറായിരം",
                7: "ഏഴായിരം", 8: "എട്ടായിരം", 9: "ഒമ്പതിനായിരം", 10: "പതിനായിരം", 11: "പതിനൊന്നായിരം",
                12: "പന്ത്രണ്ടായിരം", 13: "പതിമൂവായിരം", 14: "പതിനാലായിരം", 15: "പതിനയ്യായിരം", 16: "പതിനാറായിരം",
                17: "പതിനേഴായിരം", 18: "പതിനെട്ടായിരം", 19: "പത്തൊമ്പതിനായിരം", 20: "ഇരുപതിനായിരം",
                30: "മുപ്പതിനായിരം", 40: "നാൽപ്പതിനായിരം", 50: "അമ്പതിനായിരം", 60: "അറുപതിനായിരം",
                70: "എഴുപതിനായിരം", 80: "എൺപതിനായിരം", 90: "തൊണ്ണൂറായിരം"}
ML_SIGNS = {"ഒ": "ൊ", "എ": "െ", "ആ": "ാ", "ഏ": "േ", "ഇ": "ി", "അ": ""}


def _ml_join(link: str, unit: str) -> str:
    # അമ്പത്തി + എട്ട് -> അമ്പത്തിയെട്ട് ; ഇരുപത്തി + രണ്ട് -> ഇരുപത്തിരണ്ട്
    return _fuse(link, unit, ML_SIGNS, glide="യ")


def _ml_below_1000(n: int) -> str:
    if n < 20:
        return ML_0_19[n]
    if n < 100:
        t, o = divmod(n, 10)
        return ML_TENS[t] if o == 0 else _ml_join(ML_TENS_LINK[t], ML_0_19[o])
    h, rest = divmod(n, 100)
    if rest == 0:
        return ML_HUNDREDS[h]
    link = ML_HUNDREDS[h][:-2] + "റ്റി" if h < 9 else "തൊള്ളായിരത്തി"   # നൂറ് -> നൂറ്റി
    return _ml_join(link, _ml_below_1000(rest))


def malayalam_number(n: int) -> str:
    if n < 1000:
        return _ml_below_1000(n)
    if n < 1_00_000:
        h, rest = divmod(n, 1000)
        if h == 1:
            lead = "ആയിരം"
        elif h < 20 or h % 10 == 0:
            lead = ML_THOUSANDS[h]
        else:
            lead = _ml_join(ML_TENS_LINK[h // 10], ML_THOUSANDS[h % 10])
        return lead if rest == 0 else f"{lead[:-1]}ത്തി {malayalam_number(rest)}"   # ആയിരം -> ആയിരത്തി
    unit, word, link = (1_00_00_000, "കോടി", "കോടി") if n >= 1_00_00_000 else (1_00_000, "ലക്ഷം", "ലക്ഷത്തി")
    h, rest = divmod(n, unit)
    head = "ഒരു" if h == 1 else malayalam_number(h)
    return f"{head} {word}" if rest == 0 else f"{head} {link} {malayalam_number(rest)}"


# ------------------------------------------------------------------------------------------------- per language
class _Lang:
    """Words one language needs around a number: decimal point, percent, rupee and paise nouns."""

    def __init__(self, number, *, point, percent, rupee_one, rupees, paisa_one, paise, attributive=None,
                 rupees_with_paise=None, percent_first=False):
        self.number = number
        self.attributive = attributive or number          # the form before a noun (only Telugu differs)
        self.point, self.percent, self.percent_first = point, percent, percent_first
        self.rupee_one, self.rupees = rupee_one, rupees
        self.rupees_with_paise = rupees_with_paise or rupees
        self.paisa_one, self.paise = paisa_one, paise


LANGUAGES = {
    "te": _Lang(telugu_number, attributive=_te_attributive, point="పాయింట్", percent="శాతం", rupee_one="ఒక రూపాయి",
                rupees="రూపాయలు", rupees_with_paise="రూపాయల", paisa_one="ఒక పైసా", paise="పైసలు"),
    "hi": _Lang(hindi_number, point="दशमलव", percent="प्रतिशत", rupee_one="एक रुपया", rupees="रुपये",
                paisa_one="एक पैसा", paise="पैसे"),
    # --- not yet native-reviewed ---
    "ta": _Lang(tamil_number, point="புள்ளி", percent="சதவீதம்", rupee_one="ஒரு ரூபாய்", rupees="ரூபாய்",
                paisa_one="ஒரு பைசா", paise="பைசா"),
    "kn": _Lang(kannada_number, point="ಪಾಯಿಂಟ್", percent="ಶೇಕಡಾ", percent_first=True, rupee_one="ಒಂದು ರೂಪಾಯಿ",
                rupees="ರೂಪಾಯಿ", paisa_one="ಒಂದು ಪೈಸೆ", paise="ಪೈಸೆ"),
    "ml": _Lang(malayalam_number, point="പോയിന്റ്", percent="ശതമാനം", rupee_one="ഒരു രൂപ", rupees="രൂപ",
                paisa_one="ഒരു പൈസ", paise="പൈസ"),
    "mr": _Lang(marathi_number, point="दशांश", percent="टक्के", rupee_one="एक रुपया", rupees="रुपये",
                paisa_one="एक पैसा", paise="पैसे"),
    "bn": _Lang(bengali_number, point="দশমিক", percent="শতাংশ", rupee_one="এক টাকা", rupees="টাকা",
                paisa_one="এক পয়সা", paise="পয়সা"),
    "gu": _Lang(gujarati_number, point="દશાંશ", percent="ટકા", rupee_one="એક રૂપિયો", rupees="રૂપિયા",
                paisa_one="એક પૈસો", paise="પૈસા"),
    "pa": _Lang(punjabi_number, point="ਦਸ਼ਮਲਵ", percent="ਪ੍ਰਤੀਸ਼ਤ", rupee_one="ਇੱਕ ਰੁਪਈਆ", rupees="ਰੁਪਏ",
                paisa_one="ਇੱਕ ਪੈਸਾ", paise="ਪੈਸੇ"),
}
REVIEWED = ("te", "hi")

MAX_NUMBER = 10 ** 13  # beyond this, leave the digits alone


def _lang(code: str | None) -> _Lang | None:
    """te / te-IN, hi / hi-IN, ta-IN ...; en-IN, "hi-en" (Hinglish) and "auto" are not rewritten."""
    code = (code or "").strip().lower().replace("_", "-")
    if code.endswith("-in"):
        code = code[:-3]
    return LANGUAGES.get(code)


def number_words(n: int, lang: str) -> str:
    """Words for a non-negative integer in `lang` (a Gnani code such as ta-IN). Raises KeyError for en-IN."""
    L = _lang(lang)
    if L is None:
        raise KeyError(lang)
    return L.number(n)


# ------------------------------------------------------------------------------------------------------- patterns
NUM = r"\d+(?:,\d+)*"
_ID_BEFORE = r"(?<![A-Za-z0-9_\-/:#@.,+])"
# a phone number, also when grouped with spaces or hyphens ("+91 98480 12345"): 10 or more digits, left as written
PHONE = re.compile(r"\+?\d(?:[ \-]?\d){9,}")
_ID_AFTER = r"(?![A-Za-z0-9_\-/:#@]|[.,]\d)"
_CURRENCY = r"(?:₹|(?<![A-Za-z])(?:Rs\.?|INR)|రూ\.?|ரூ\.?|ರೂ\.?|രൂ\.?|रु\.?|रू\.?)"
_INDIC = r"\u0900-\u0DFF"
# a lakh / crore word after an amount ("₹3.5 lakh", "₹3 లక్షలు"): full words only, longest first
_SCALE_WORDS = {
    1_00_000: ("లక్షలు", "లక్షల", "లక్ష", "लाख", "लक्ष", "இலட்சம்", "லட்சம்", "ಲಕ್ಷ", "ലക്ഷം", "લાખ", "ਲੱਖ", "লাখ"),
    1_00_00_000: ("కోట్లు", "కోట్ల", "కోటి", "करोड़", "कोटी", "கோடி", "ಕೋಟಿ", "കോടി", "કરોડ", "ਕਰੋੜ", "কোটি"),
}
_SCALE_VALUE = {w: v for v, words in _SCALE_WORDS.items() for w in words}
_SCALE_EN = {"lakh": 1_00_000, "lakhs": 1_00_000, "lac": 1_00_000, "lacs": 1_00_000,
             "crore": 1_00_00_000, "crores": 1_00_00_000, "cr": 1_00_00_000}
_SCALE = (r"(?:(?P<scale_en>lakhs?|lacs?|crores?|cr)(?![A-Za-z])"
          rf"|(?P<scale_in>{'|'.join(sorted(_SCALE_VALUE, key=len, reverse=True))})(?P<scale_tail>[{_INDIC}]*))")
# a rupee noun right after a bare number: the number then takes its before-a-noun form (Telugu)
_RUPEE_WORD = r"(?=\s*(?:రూపాయ|रुपय|रुपए|ரூபாய்|ರೂಪಾಯಿ|രൂപ|રૂપિય|ਰੁਪਏ|টাকা|rupee))"

PATTERN = re.compile(
    rf"(?P<money>{_CURRENCY}\s?(?P<amount>{NUM}(?:\.\d+)?)(?![\d])(?:\s?{_SCALE})?)"
    rf"|{_ID_BEFORE}(?P<percent>{NUM}(?:\.\d+)?)\s?%"
    rf"|{_ID_BEFORE}(?P<plain>{NUM}(?:\.\d+)?){_ID_AFTER}(?P<before_rupee>{_RUPEE_WORD})?",
    re.IGNORECASE,
)


def _grouping_ok(digits: str) -> bool:
    """1,234,567 (western) or 12,34,567 (Indian); anything else is a list, not one number."""
    if "," not in digits:
        return True
    parts = digits.split(",")
    if not 1 <= len(parts[0]) <= 3 or len(parts[-1]) != 3:
        return False
    middle = parts[1:-1]
    western = all(len(p) == 3 for p in middle)
    indian = len(parts[0]) <= 2 and all(len(p) == 2 for p in middle)
    return western or indian


def _digits(s: str, L: _Lang) -> str:
    return " ".join(L.number(int(ch)) for ch in s)


def _decimal(text: str, L: _Lang, *, attributive: bool) -> str | None:
    whole, _, frac = text.partition(".")
    if not _grouping_ok(whole):
        return None
    n = int(whole.replace(",", ""))
    if n >= MAX_NUMBER:
        return None
    if frac:
        return f"{L.number(n)} {L.point} {_digits(frac, L)}"
    return L.attributive(n) if attributive else L.number(n)


def _money(amount: str, L: _Lang, scale: int = 1) -> str | None:
    whole, _, frac = amount.partition(".")
    if not _grouping_ok(whole):
        return None
    try:
        value = Decimal(whole.replace(",", "") + ("." + frac if frac else "")) * scale
    except InvalidOperation:
        return None
    if value >= MAX_NUMBER:
        return None
    if scale == 1 and len(frac) > 2:  # not a paise amount: say it as a decimal number of rupees
        return f"{_decimal(amount, L, attributive=False)} {L.rupees}"
    rupees = int(value)
    paise = int((value - rupees) * 100)
    if paise == 0:
        return L.rupee_one if rupees == 1 else f"{L.attributive(rupees)} {L.rupees}"
    paise_words = L.paisa_one if paise == 1 else f"{L.number(paise)} {L.paise}"
    if rupees == 0:
        return paise_words
    head = L.rupee_one if rupees == 1 else f"{L.attributive(rupees)} {L.rupees_with_paise}"
    return f"{head} {paise_words}"


def _plain(text: str, L: _Lang, *, attributive: bool) -> str | None:
    whole = text.partition(".")[0]
    bare = whole.replace(",", "")
    if len(bare) > 1 and bare.startswith("0"):
        return None  # a code or an OTP, not a quantity
    if "," not in whole and len(bare) >= 10:
        return None  # phone-like
    if not _grouping_ok(whole):
        # "1,2,3" is a list: say each item
        out = [_plain(p, L, attributive=False) for p in whole.split(",")]
        return None if any(o is None for o in out) or "." in text else ", ".join(out)
    return _decimal(text, L, attributive=attributive)


def _percent(text: str, L: _Lang) -> str | None:
    n = _decimal(text, L, attributive=True)
    if n is None:
        return None
    return f"{L.percent} {n}" if L.percent_first else f"{n} {L.percent}"


def verbalise(text: str, lang: str | None) -> str:
    """`text` with its numbers written out as words in `lang` (te, hi, ta, kn, ml, mr, bn, gu, pa, with or without
    -IN); English and any other language, and every non-number character, are returned unchanged."""
    L = _lang(lang)
    if L is None or not text or not any(ch.isdigit() for ch in text):
        return text

    phones = [m.span() for m in PHONE.finditer(text)]

    def sub(m: re.Match) -> str:
        if any(a <= m.start() < b for a, b in phones):
            return m.group(0)
        if m.group("money"):
            amount, en, ind, tail = m.group("amount"), m.group("scale_en"), m.group("scale_in"), m.group("scale_tail")
            if en:
                said = _money(amount, L, _SCALE_EN[en.lower()])
            elif ind and not tail:
                said = _money(amount, L, _SCALE_VALUE[ind])
            elif ind:  # an inflected unit word (లక్షలకు): say the number and keep the word as written
                n = _decimal(amount, L, attributive=True)
                said = None if n is None else f"{n} {ind}{tail}"
            else:
                said = _money(amount, L)
        elif m.group("percent"):
            said = _percent(m.group("percent"), L)
        else:
            said = _plain(m.group("plain"), L, attributive=m.group("before_rupee") is not None)
        return m.group(0) if said is None else said

    return PATTERN.sub(sub, text)
