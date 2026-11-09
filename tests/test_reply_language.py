"""Call flow v2: the language a phone caller picked on the keypad reaches the graph as `reply_language` (voice turns
only). It never changes whose file is explained; it picks the answer language: en / hi / te map onto the graph's
template locale, the other seven keep en templates while the LLM is told to write in the chosen language."""
from types import SimpleNamespace

import pytest

from loan_advisory.agent import prompts as P
from loan_advisory.agent import server as srv
from loan_advisory.agent.contracts import REPLY_LANGUAGES

CLAIMS = {"application_ref": "APP-1", "applicant_ref": "CUS-1", "locale": "te", "literacy": "low"}


def req(body):
    return srv.App(None)._request(CLAIMS, {"request_id": "r", "conversation_id": "c", "message": "hi", **body})


def test_ten_reply_languages():
    assert set(REPLY_LANGUAGES) == {"en-IN", "hi-IN", "te-IN", "ta-IN", "kn-IN", "ml-IN", "mr-IN", "bn-IN", "gu-IN",
                                    "pa-IN"}


@pytest.mark.parametrize("code,locale", [("en-IN", "en"), ("hi-IN", "hi"), ("te-IN", "te"), ("ta-IN", "en"),
                                         ("pa-IN", "en")])
def test_voice_reply_language_sets_answer_language_and_template_locale(code, locale):
    r = req({"channel": "voice", "reply_language": code})
    assert r.reply_language == code and r.locale == locale
    assert P.answer_language(r) == REPLY_LANGUAGES[code]
    assert r.application_ref == "APP-1"  # identity still from the assertion


def test_reply_language_is_ignored_for_text_and_when_invalid():
    for body in ({"channel": "text", "reply_language": "ta-IN"}, {"channel": "voice", "reply_language": "fr-FR"},
                 {"channel": "voice", "reply_language": ["ta-IN"]}, {"channel": "voice"}):
        r = req(body)
        assert r.reply_language is None and r.locale == "te"  # the assertion's locale
        assert P.answer_language(r) == P.LANG["te"]


def test_answer_language_without_the_field():
    assert P.answer_language(SimpleNamespace(locale="hi")) == P.LANG["hi"]
    assert P.answer_language(SimpleNamespace(locale="xx", reply_language=None)) == "English"
