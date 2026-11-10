"""Keypad caller verification, call flow v2 (apps/voice/caller_verification.py): language from the directory record,
greeting with the press-9 language menu, ONE factor (last 4 of the loan application number), no read-back, salted-HMAC
match, lockout after 3 failures, keypress timeouts, the nudge at most once per prompt, robust caller-ID normalisation,
and the rule that keyed digits never reach logs or anything returned to the LLM. Pure asyncio; no LiveKit, no network."""
import asyncio
import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "voice"))
import caller_verification as cv  # noqa: E402
import prompt_texts as pt  # noqa: E402

KEY = b"test-only-hmac-key-0123456789abcdef"
PHONE = "+910000123456"
APP4 = "4729"
ENTRY = {"phone": PHONE, "applicant_id": "CUS-TEST-001", "application_id": "APP-TEST-001",
         "application_last4": APP4, "language": "te-IN"}
WAITS = {"greeting", "enter", "no_match", "cleared", cv.LANGUAGE_MENU}


def directory(entry=ENTRY):
    return cv.CallerDirectory.from_seed(KEY, cv.build_seed(KEY, [entry]))


class Caller:
    """Fake call: records the keys the agent speaks (and in which language), and keys the next scripted presses
    whenever a prompt waits for input. `None` in the script means "press nothing at this prompt"."""

    def __init__(self, script, entry=ENTRY, nudge_keys=None):
        self.script, self.nudge_keys = list(script), nudge_keys
        self.spoken, self.langs, self.changes = [], [], []
        self.v = cv.KeypadVerifier(directory(entry).lookup(entry["phone"]), KEY, speak=self.speak,
                                   on_language=self.changes.append, nudge_s=0.05, give_up_s=0.15, window_s=0.05,
                                   menu_wait_s=0.05)

    def _feed(self):
        if self.script:
            for k in self.script.pop(0) or "":
                self.v.press(k)

    async def speak(self, key):
        self.spoken.append(key)
        self.langs.append(self.v.language)
        if key in WAITS:
            self._feed()
        elif key == "nudge" and self.nudge_keys:
            for k in self.nudge_keys:
                self.v.press(k)
            self.nudge_keys = None


def run(caller):
    return asyncio.run(caller.v.run())


# ----------------------------------------------------------------------------------------------- seed + lookup
def test_seed_holds_hashes_and_language_only():
    seed = cv.build_seed(KEY, [ENTRY])
    blob = json.dumps(seed)
    for secret in (PHONE, PHONE[1:], "123456", APP4):
        assert secret not in blob
    rec = seed["records"][0]
    assert set(rec["factors"]) == {"application"} and len(rec["factors"]["application"]) == 64
    assert rec["language"] == "te-IN"


def test_language_defaults_to_english_and_is_validated():
    entry = {k: v for k, v in ENTRY.items() if k != "language"}
    assert directory(entry).lookup(PHONE).language == "en-IN"
    with pytest.raises(ValueError):
        cv.build_seed(KEY, [dict(ENTRY, language="fr-FR")])
    seed = cv.build_seed(KEY, [ENTRY])
    seed["records"][0]["language"] = "xx"  # a hand-edited directory falls back, never crashes
    assert cv.CallerDirectory.from_seed(KEY, seed).lookup(PHONE).language == "en-IN"


def test_v1_directory_with_bank_factor_still_loads_and_ignores_it():
    seed = cv.build_seed(KEY, [ENTRY])
    seed["records"][0]["factors"]["bank_account"] = "ab" * 32
    del seed["records"][0]["language"]
    rec = cv.CallerDirectory.from_seed(KEY, seed).lookup(PHONE)
    assert set(rec.factor_hmacs) == {"application"} and rec.language == "en-IN"


def test_known_and_unknown_caller_id():
    d = directory()
    assert d.lookup(PHONE).application_id == "APP-TEST-001"
    assert d.lookup("sip:+910000123456@trunk.example") is not None and d.lookup("tel:+91 0000 123456") is not None
    assert d.lookup("+910000999999") is None and d.lookup("") is None and d.lookup(None) is None
    assert d.lookup("anonymous") is None


@pytest.mark.parametrize("raw", ["+919876543210", "919876543210", "09876543210", "9876543210", "+91 98765-43210",
                                 "tel:+91-98765-43210", "sip:+919876543210@sip.example.com;user=phone",
                                 "sip:919876543210@10.0.0.1", "(+91) 98765 43210", "0091 98765 43210"])
def test_indian_number_forms_normalise_to_one_key(raw):
    assert cv.normalize_phone(raw) == "+919876543210"


def test_other_numbers_keep_their_country_code_and_junk_is_refused():
    assert cv.normalize_phone("+1 484 295 0295") == "+14842950295"
    assert cv.normalize_phone("14842950295") == "+14842950295"
    for bad in ("anonymous", "sip:anonymous@host", "12ab34", "+12", "", None):
        assert cv.normalize_phone(bad) is None


def test_caller_id_attribute_order_and_fallbacks():
    assert cv.caller_id({"sip.phoneNumber": "9876543210", "sip.from": "+911111111111"}) == "+919876543210"
    assert cv.caller_id({"sip.phoneNumber": "anonymous", "sip.from": "sip:+919876543210@host"}) == "+919876543210"
    assert cv.caller_id({"sip.fromUser": "09876543210"}) == "+919876543210"
    assert cv.caller_id({}, "sip_+919876543210") == "+919876543210"
    assert cv.caller_id({}, "web-user") is None and cv.caller_id(None) is None


def test_mask_phone_keeps_only_the_last_two_digits():
    assert cv.mask_phone("+919876543210") == "+**********10"
    assert "98765" not in cv.mask_phone("+919876543210") and cv.mask_phone(None) == "none"


def test_hash_match_and_mismatch_and_wrong_key():
    rec = directory().lookup(PHONE)
    assert rec.matches(KEY, "application", APP4)
    assert not rec.matches(KEY, "bank_account", APP4)  # the v1 second factor no longer exists
    assert not rec.matches(KEY, "application", "0000") and not rec.matches(KEY, "application", "472")
    assert not rec.matches(b"another-key-another-key-1234", "application", APP4)


def test_same_digits_hash_differently_per_record_salt():
    s = cv.build_seed(KEY, [ENTRY, dict(ENTRY, phone="+910000654321", applicant_id="CUS-TEST-002")])
    a, b = s["records"]
    assert a["factors"]["application"] != b["factors"]["application"]


def test_key_from_env_fails_closed_when_missing_or_short():
    assert cv.load_key({}) is None and cv.load_key({cv.KEY_ENV: "short"}) is None
    assert cv.load_key({cv.KEY_ENV: "x" * 32}) == b"x" * 32


# ----------------------------------------------------------------------------------------------- the flow
def test_greeting_in_directory_language_then_enter_then_verified():
    c = Caller([None, APP4])
    assert run(c) == cv.VERIFIED and c.v.verified and c.v.factor == "application"
    assert c.spoken == ["greeting", "enter", "verified"]
    assert set(c.langs) == {"te-IN"} and c.changes == []


def test_no_factor_menu_and_no_readback():
    c = Caller([None, APP4])
    run(c)
    assert not ({"menu", "menu_again", "you_entered", "confirm", "retype"} & set(c.spoken))
    assert not hasattr(cv, "readback") and "bank_account" not in cv.FACTORS


def test_press_9_opens_the_menu_and_the_choice_carries_on():
    c = Caller(["9", "4", APP4])  # Tamil
    assert run(c) == cv.VERIFIED
    assert c.spoken == ["greeting", cv.LANGUAGE_MENU, "enter", "verified"]
    assert c.langs[-2:] == ["ta-IN", "ta-IN"] and c.changes == ["ta-IN"] and c.v.language == "ta-IN"


@pytest.mark.parametrize("key,code", sorted(pt.LANGUAGE_BY_KEY.items()))
def test_menu_key_mapping(key, code):
    c = Caller(["9", key, APP4])
    assert run(c) == cv.VERIFIED and c.v.language == code


def test_menu_mapping_is_the_owner_design():
    assert pt.LANGUAGE_BY_KEY == {"1": "en-IN", "2": "hi-IN", "3": "te-IN", "4": "ta-IN", "5": "kn-IN",
                                  "6": "ml-IN", "7": "mr-IN", "8": "bn-IN", "9": "gu-IN", "*": "pa-IN"}
    assert cv.CONTINUE_KEY == "0" and cv.CONTINUE_KEY not in pt.LANGUAGE_BY_KEY


def test_menu_hash_is_ignored_and_no_key_keeps_the_language_without_replay():
    c = Caller(["9", "#", APP4])
    assert run(c) == cv.VERIFIED and c.v.language == "te-IN" and c.changes == []
    assert c.spoken == ["greeting", cv.LANGUAGE_MENU, "enter", "verified"]


def test_menu_no_key_keeps_the_current_language_without_replay():
    c = Caller(["9", None, APP4])
    assert run(c) == cv.VERIFIED and c.v.language == "te-IN" and c.changes == []
    assert c.spoken.count(cv.LANGUAGE_MENU) == 1 and c.spoken == ["greeting", cv.LANGUAGE_MENU, "enter", "verified"]


def test_menu_zero_keeps_the_current_language():
    c = Caller(["9", "0", APP4])
    assert run(c) == cv.VERIFIED and c.v.language == "te-IN" and c.changes == []
    assert c.spoken == ["greeting", cv.LANGUAGE_MENU, "enter", "verified"]


def test_menu_star_picks_punjabi():
    c = Caller(["9", "*", APP4])
    assert run(c) == cv.VERIFIED and c.v.language == "pa-IN" and c.changes == ["pa-IN"]


def test_menu_hash_then_choice_within_the_wait_still_switches():
    c = Caller(["9", "#2", APP4])
    assert run(c) == cv.VERIFIED and c.v.language == "hi-IN" and c.spoken.count(cv.LANGUAGE_MENU) == 1


def test_menu_waits_menu_wait_s_after_playing_then_moves_on():
    async def go():
        spoken = []

        async def speak(key):
            spoken.append(key)
            if key == "greeting":
                call.press("9")
        call = cv.KeypadCall(speak=speak, window_s=0.05, menu_wait_s=0.3)
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await call.run_unverifiable(cv.UNKNOWN_CALLER)
        return spoken, loop.time() - t0
    spoken, took = asyncio.run(go())
    assert spoken == ["greeting", cv.LANGUAGE_MENU, "unknown_caller"] and 0.3 <= took < 0.6
    assert cv.MENU_WAIT_S == 6.0


def test_zero_during_the_greeting_goes_straight_on_and_cuts_the_greeting():
    """0 while the greeting plays: the worker's interrupt is called and the enter prompt follows with no window."""
    async def go():
        spoken, cut = [], []
        playing = asyncio.Event()

        async def speak(key):
            spoken.append(key)
            if key == "greeting":
                call.press("0")
                try:  # a long greeting that only ends when interrupted
                    await asyncio.wait_for(playing.wait(), 5)
                except asyncio.TimeoutError:
                    pass
            elif key == "enter":
                for k in APP4:
                    call.press(k)

        def interrupt():
            cut.append(spoken[-1])
            playing.set()
        call = cv.KeypadVerifier(directory().lookup(PHONE), KEY, speak=speak, interrupt=interrupt, window_s=5,
                                 menu_wait_s=5, nudge_s=1, give_up_s=2)
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        out = await call.run()
        return out, spoken, cut, loop.time() - t0, call
    out, spoken, cut, took, call = asyncio.run(go())
    assert out == cv.VERIFIED and spoken == ["greeting", "enter", "verified"] and cut == ["greeting"]
    assert took < 1.0 and call.language == "te-IN"  # no 5 s window, no menu


def test_zero_in_the_greeting_window_goes_straight_on():
    async def go():
        spoken = []

        async def speak(key):
            spoken.append(key)
            if key == "enter":
                for k in APP4:
                    call.press(k)
        call = cv.KeypadVerifier(directory().lookup(PHONE), KEY, speak=speak, window_s=5, menu_wait_s=5,
                                 nudge_s=1, give_up_s=2)
        task = asyncio.ensure_future(call.run())
        await asyncio.sleep(0.05)
        call.press("0")
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        out = await task
        return out, spoken, loop.time() - t0
    out, spoken, took = asyncio.run(go())
    assert out == cv.VERIFIED and spoken == ["greeting", "enter", "verified"] and took < 1.0


def test_zero_during_the_menu_cuts_it_and_keeps_the_language():
    async def go():
        spoken, cut = [], []
        playing = asyncio.Event()

        async def speak(key):
            spoken.append(key)
            if key == "greeting":
                call.press("9")
            elif key == cv.LANGUAGE_MENU:
                call.press("0")
                try:
                    await asyncio.wait_for(playing.wait(), 5)
                except asyncio.TimeoutError:
                    pass
            elif key == "enter":
                for k in APP4:
                    call.press(k)

        def interrupt():
            cut.append(spoken[-1])
            playing.set()
        call = cv.KeypadVerifier(directory().lookup(PHONE), KEY, speak=speak, interrupt=interrupt, window_s=5,
                                 menu_wait_s=5, nudge_s=1, give_up_s=2)
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        out = await call.run()
        return out, spoken, cut, loop.time() - t0, call
    out, spoken, cut, took, call = asyncio.run(go())
    assert out == cv.VERIFIED and cut == [cv.LANGUAGE_MENU] and took < 1.0
    assert spoken == ["greeting", cv.LANGUAGE_MENU, "enter", "verified"] and call.language == "te-IN"


def test_a_menu_choice_during_playback_cuts_the_menu_and_switches():
    async def go():
        cut = []
        playing = asyncio.Event()

        async def speak(key):
            if key == "greeting":
                call.press("9")
            elif key == cv.LANGUAGE_MENU:
                call.press("3")
                try:
                    await asyncio.wait_for(playing.wait(), 5)
                except asyncio.TimeoutError:
                    pass

        def interrupt():
            cut.append(1)
            playing.set()
        call = cv.KeypadCall(speak=speak, language="en-IN", interrupt=interrupt, window_s=5, menu_wait_s=5)
        await call.run_unverifiable(cv.UNKNOWN_CALLER)
        return cut, call.language
    assert asyncio.run(go()) == ([1], "te-IN")


def test_other_keys_never_cut_the_greeting():
    async def go():
        cut = []

        async def speak(key):
            if key == "greeting":
                call.press("1")
                call.press("*")
        call = cv.KeypadCall(speak=speak, interrupt=lambda: cut.append(1), window_s=0.05, menu_wait_s=0.05)
        await call.run_unverifiable(cv.UNKNOWN_CALLER)
        return cut
    assert asyncio.run(go()) == []


def test_9_after_the_enter_prompt_is_an_ordinary_digit():
    entry = dict(ENTRY, application_last4="9123")
    c = Caller([None, "9123"], entry=entry)
    assert run(c) == cv.VERIFIED and cv.LANGUAGE_MENU not in c.spoken


def test_other_keys_during_the_greeting_are_dropped():
    c = Caller(["1", APP4])
    assert run(c) == cv.VERIFIED and cv.LANGUAGE_MENU not in c.spoken


def test_mismatch_says_try_again_then_success():
    c = Caller([None, "0000", APP4])
    assert run(c) == cv.VERIFIED and c.v.failures == 1
    assert c.spoken == ["greeting", "enter", "no_match", "verified"]
    assert pt.TEXT["en-IN"]["no_match"] == "That didn't match. Please try again."


def test_lockout_after_three_failures_and_stays_locked_without_replaying():
    c = Caller([None, "0000", "1111", "2222"])
    assert run(c) == cv.LOCKED and c.v.locked and c.v.failures == 3 and not c.v.verified
    assert c.spoken[-1] == "locked" and c.spoken.count("locked") == 1
    c.script = [APP4]
    assert run(c) == cv.LOCKED and not c.v.verified and c.spoken.count("locked") == 1  # not spoken again


def test_hash_and_star_in_the_greeting_are_dropped():
    c = Caller(["#*", APP4])
    assert run(c) == cv.VERIFIED and c.v.language == "te-IN" and c.changes == []
    assert c.spoken == ["greeting", "enter", "verified"]


def test_zero_in_the_greeting_continues_and_is_then_an_ordinary_digit():
    entry = {**ENTRY, "application_last4": "0470"}
    c = Caller(["0", "0470"], entry=entry)
    assert run(c) == cv.VERIFIED and c.v.language == "te-IN" and c.changes == []
    assert c.spoken == ["greeting", "enter", "verified"]


def test_star_clears_and_hash_is_ignored_while_typing():
    c = Caller([None, "47*", "#4729"])
    assert run(c) == cv.VERIFIED and "cleared" in c.spoken


def test_no_keypress_nudges_once_and_ends_at_15s():
    c = Caller([None, None])
    assert run(c) == cv.TIMEOUT and not c.v.verified and c.v.failures == 0
    assert c.spoken == ["greeting", "enter", "nudge", "timeout"]


def test_key_after_nudge_continues():
    c = Caller([None, None], nudge_keys=APP4)
    assert run(c) == cv.VERIFIED


def test_nudge_is_allowed_once_per_prompt():
    c = Caller([None, None])
    c.v.step = 5
    assert c.v.may_nudge() and not c.v.may_nudge() and not c.v.may_nudge()
    c.v.step = 6
    assert c.v.may_nudge() and not c.v.may_nudge()


def test_greeting_plays_once_across_attempts():
    c = Caller([None, None])
    assert run(c) == cv.TIMEOUT
    c.script = [APP4]
    assert run(c) == cv.VERIFIED and c.spoken.count("greeting") == 1


def test_timer_defaults_match_the_design():
    assert (cv.NUDGE_S, cv.GIVE_UP_S, cv.MAX_ATTEMPTS, cv.DIGITS, cv.LANGUAGE_KEY) == (8.0, 15.0, 3, 4, "9")


def test_presses_outside_an_attempt_are_ignored():
    c = Caller([])
    for k in "14729":
        c.v.press(k)
    assert c.v._keys.empty()


def test_unknown_caller_gets_greeting_menu_and_one_refusal_in_the_chosen_language():
    spoken, langs = [], []
    script = ["9", "3"]

    async def speak(key):
        spoken.append(key)
        langs.append(call.language)
        if key in WAITS and script:
            for k in script.pop(0):
                call.press(k)
    call = cv.KeypadCall(speak=speak, window_s=0.05, menu_wait_s=0.05)
    assert asyncio.run(call.run_unverifiable(cv.UNKNOWN_CALLER)) == cv.UNKNOWN_CALLER
    assert spoken == ["greeting", cv.LANGUAGE_MENU, "unknown_caller"]
    assert langs[0] == "en-IN" and langs[-1] == "te-IN"


# ----------------------------------------------------------------------------------------------- exposure
def test_digits_never_in_logs(caplog):
    caplog.set_level(logging.DEBUG)
    for script in ([None, "0000", "1111", "2222"], [None, APP4], ["9", "2", "0000", APP4]):
        run(Caller(script))
    text = caplog.text
    assert "****" in text
    for d in (APP4, "0000", "1111", "2222"):
        assert d not in text


def test_outcome_returned_to_the_llm_carries_no_digits():
    for script in ([None, APP4], [None, "0000", "1111", "2222"], [None, None]):
        out = run(Caller(script))
        assert out in (cv.VERIFIED, cv.LOCKED, cv.TIMEOUT) and not any(ch.isdigit() for ch in out)


def test_mask():
    assert cv.mask("4729") == "****" and cv.mask() == "****"


def test_pre_verification_lines_carry_no_digits_or_name_slots():
    for code, lines in pt.TEXT.items():
        assert set(lines) == set(pt.PROMPT_KEYS), code
        for key, text in lines.items():
            assert not any(ch.isdigit() for ch in text), (code, key)  # numbers are written as words
            assert "{" not in text, (code, key)
    assert "{name}" in pt.LIVE_GREETING["te-IN"][0] and "{name}" not in pt.LIVE_GREETING["te-IN"][1]


# ----------------------------------------------------------------------------------------------- the gate
def test_gate_refuses_the_agent_server_until_verified():
    calls = []

    async def exchange():
        calls.append(1)
        return {"assertion": "a"}

    c = Caller([None, APP4])
    gate = cv.VerifiedGate(c.v, exchange)
    with pytest.raises(cv.NotVerified):
        asyncio.run(gate.fresh())
    assert calls == []
    run(c)
    assert asyncio.run(gate.fresh()) == {"assertion": "a"} and calls == [1]


def test_gate_for_unknown_caller_never_opens():
    gate = cv.VerifiedGate(None, None)
    with pytest.raises(cv.NotVerified):
        asyncio.run(gate.fresh())
    assert not gate.verified


# ----------------------------------------------------------------------------------------------- seed script
def test_seed_script_writes_hashes_and_languages_and_round_trips(tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(ROOT / "scripts" / "voice"))
    import seed_caller_directory as sd
    monkeypatch.setenv(cv.KEY_ENV, KEY.decode())
    out = tmp_path / "dir.json"
    entries = sd.demo_entries()
    first = entries[0]["applicant_id"]
    assert sd.main(["--out", str(out), "--language", f"{first}=pa-IN"]) == 0
    blob, printed = out.read_text(), capsys.readouterr().out
    for e in entries:
        assert e["phone"].startswith("+910000") and e["phone"] not in blob and e["phone"][1:] not in blob
        assert e["phone"] not in printed  # demo values only with --show-demo-values
    d = cv.CallerDirectory.load(out, KEY)
    assert len(d) == len(entries)
    rec = d.lookup(entries[0]["phone"])
    assert rec.application_id == entries[0]["application_id"] and rec.language == "pa-IN"
    assert rec.matches(KEY, "application", entries[0]["application_last4"])
    assert d.lookup(entries[1]["phone"]).language == sd.DEMO_LANGUAGES[1]
    assert {e["language"] for e in entries} <= set(pt.LANGUAGES) and len({e["language"] for e in entries}) > 3


def test_seed_script_refuses_without_key_or_with_a_bad_language(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts" / "voice"))
    import seed_caller_directory as sd
    monkeypatch.delenv(cv.KEY_ENV, raising=False)
    assert sd.main(["--out", str(tmp_path / "x.json")]) == 2 and not (tmp_path / "x.json").exists()
    monkeypatch.setenv(cv.KEY_ENV, KEY.decode())
    assert sd.main(["--out", str(tmp_path / "y.json"), "--language", "CUS=fr-FR"]) == 2
    assert not (tmp_path / "y.json").exists()
