"""live_signals: deterministic lexicon + keyword detection on one utterance (en / hi / te, script and romanised)."""
import pytest

from loan_advisory.agent import live_signals as LS


@pytest.mark.parametrize("text", ["I don't understand", "samajh nahi aaya", "समझ नहीं आया", "artham kaledu",
                                  "అర్థం కాలేదు", "malli cheppandi please", "kya matlab?"])
def test_confusion(text):
    s = LS.detect(text)
    assert s.confusion and not s.understood


@pytest.mark.parametrize("text", ["samajh gaya", "ardhamaindi", "got it, thanks", "అర్థమైంది"])
def test_understood_is_not_confusion(text):
    s = LS.detect(text)
    assert s.understood and not s.confusion


@pytest.mark.parametrize("text", ["this is nonsense", "bakwas service", "మోసం చేస్తున్నారు", "useless bank"])
def test_anger(text):
    assert LS.detect(text).anger


@pytest.mark.parametrize("text", ["main bahut pareshan hoon, kya karun", "Papa ki surgery hai", "em cheyalo teliyatledu",
                                  "I'll lose my house"])
def test_distress(text):
    assert LS.detect(text).distress


@pytest.mark.parametrize("text", ["I want to end my life", "aatmahatya kar lunga", "ఆత్మహత్య చేసుకుంటా"])
def test_self_harm_is_also_distress(text):
    s = LS.detect(text)
    assert s.self_harm and s.distress


def test_human_request_uses_router_pattern_and_skips_howto():
    assert LS.detect("Can I talk to a person?").human
    assert LS.detect("kisi insaan se baat karni hai").human
    assert not LS.detect("how do I complain to the grievance officer").human


@pytest.mark.parametrize("text,loc", [("Can you tell me in English?", "en"), ("Telugu lo cheppandi", "te"),
                                      ("हिंदी में बताइए", "hi")])
def test_explicit_language_request(text, loc):
    assert LS.detect(text).lang_request == loc


def test_language_code_and_latin_ratio():
    s = LS.detect("मेरा loan क्यों reject हुआ", "hi-IN")
    assert s.lang == "hi" and 0.2 < s.latin_ratio < 0.8 and s.words == 5
    assert LS.detect("hello", "xx-YY").lang is None
    assert LS.detect("", None).words == 0


def test_no_false_positive_on_plain_question():
    s = LS.detect("Why was my loan rejected?", "en-IN")
    assert not any([s.confusion, s.anger, s.distress, s.self_harm, s.human, s.lang_request])
