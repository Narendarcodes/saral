"""Keypad caller verification (apps/voice/caller_verification.py): menu, 4-digit DTMF collection, masked read-back,
confirm/retype, salted-HMAC match, lockout after 3 failures, keypress timeouts, unknown caller ID, and the rule that
keyed digits never reach logs, transcripts or anything returned to the LLM. Pure asyncio; no LiveKit, no network."""
import asyncio
import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "voice"))
import caller_verification as cv  # noqa: E402

KEY = b"test-only-hmac-key-0123456789abcdef"
PHONE = "+910000123456"
APP4, BANK4 = "4729", "8051"
ENTRY = {"phone": PHONE, "applicant_id": "CUS-TEST-001", "application_id": "APP-TEST-001",
         "application_last4": APP4, "bank_account_last4": BANK4}
EXPECTS_KEYS = {cv.PROMPT[k] for k in ("menu", "menu_again", "enter", "retype", "cleared", "confirm_again")}


def directory():
    return cv.CallerDirectory.from_seed(KEY, cv.build_seed(KEY, [ENTRY]))


class Caller:
    """Fake call: records what the agent says and keys the next scripted presses whenever a prompt waits for input."""

    def __init__(self, script, nudge_keys=None):
        self.script, self.nudge_keys = list(script), nudge_keys
        self.spoken, self.shown, self.private_spoken = [], [], []
        self.v = cv.KeypadVerifier(directory().lookup(PHONE), KEY, speak=self.speak, speak_private=self.speak_private,
                                   nudge_s=0.05, give_up_s=0.15)

    def _feed(self):
        if self.script:
            for k in self.script.pop(0):
                self.v.press(k)

    async def speak(self, text):
        self.spoken.append(text); self.shown.append(text)
        if text in EXPECTS_KEYS:
            self._feed()
        elif text == cv.PROMPT["nudge"] and self.nudge_keys:
            for k in self.nudge_keys:
                self.v.press(k)
            self.nudge_keys = None

    async def speak_private(self, spoken, shown):
        self.private_spoken.append(spoken); self.shown.append(shown)
        self._feed()


def run(caller):
    return asyncio.run(caller.v.run())


# ----------------------------------------------------------------------------------------------- seed + lookup
def test_seed_holds_hashes_only():
    seed = cv.build_seed(KEY, [ENTRY])
    blob = json.dumps(seed)
    for secret in (PHONE, PHONE[1:], "123456", APP4, BANK4):
        assert secret not in blob
    rec = seed["records"][0]
    assert set(rec["factors"]) == {"application", "bank_account"} and len(rec["factors"]["application"]) == 64


def test_known_and_unknown_caller_id():
    d = directory()
    assert d.lookup(PHONE).application_id == "APP-TEST-001"
    assert d.lookup("sip:+910000123456@trunk.example") is not None and d.lookup("tel:+91 0000 123456") is not None
    assert d.lookup("+910000999999") is None and d.lookup("") is None and d.lookup(None) is None
    assert d.lookup("anonymous") is None


def test_hash_match_and_mismatch_and_wrong_key():
    rec = directory().lookup(PHONE)
    assert rec.matches(KEY, "application", APP4) and rec.matches(KEY, "bank_account", BANK4)
    assert not rec.matches(KEY, "application", BANK4)  # the other factor's digits do not count
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
def test_menu_choice_1_collects_four_digits_and_verifies_application_factor():
    c = Caller(["1", APP4, "1"])
    assert run(c) == cv.VERIFIED and c.v.verified and c.v.factor == "application"
    assert c.spoken[0] == cv.PROMPT["menu"] and c.spoken[1] == cv.PROMPT["enter"]
    assert c.spoken[-1] == cv.PROMPT["verified"]


def test_menu_choice_2_uses_bank_account_only():
    c = Caller(["2", BANK4, "1"])
    assert run(c) == cv.VERIFIED and c.v.factor == "bank_account"
    c = Caller(["2", APP4, "1", APP4, "1", APP4, "1"])  # application digits under the bank factor never pass
    assert run(c) == cv.LOCKED


def test_invalid_menu_key_reprompts():
    c = Caller(["9", "1", APP4, "1"])
    assert run(c) == cv.VERIFIED and cv.PROMPT["menu_again"] in c.spoken


def test_readback_is_fixed_template_spoken_but_masked_when_shown():
    c = Caller(["1", APP4, "1"])
    run(c)
    assert c.private_spoken == ["You entered 4, 7, 2, 9. Press 1 to confirm or 2 to retype."]
    assert "You entered ****. Press 1 to confirm or 2 to retype." in c.shown
    assert not any(d in s for s in c.shown for d in (APP4, "4, 7, 2, 9"))


def test_retype_then_confirm():
    c = Caller(["1", "1111", "2", APP4, "1"])
    assert run(c) == cv.VERIFIED and c.v.failures == 0
    assert cv.PROMPT["retype"] in c.spoken and len(c.private_spoken) == 2


def test_star_clears_and_hash_is_ignored_while_typing():
    c = Caller(["1", "47*", "#4729", "1"])
    assert run(c) == cv.VERIFIED and cv.PROMPT["cleared"] in c.spoken


def test_extra_keys_after_four_digits_do_not_answer_the_readback():
    c = Caller(["1", APP4 + "1", "1"])  # the 5th key is drained before the read-back
    assert run(c) == cv.VERIFIED


def test_mismatch_then_success():
    c = Caller(["1", "0000", "1", APP4, "1"])
    assert run(c) == cv.VERIFIED and c.v.failures == 1 and cv.PROMPT["no_match"] in c.spoken


def test_lockout_after_three_failures_and_stays_locked_for_the_session():
    c = Caller(["1", "0000", "1", "1111", "1", "2222", "1"])
    assert run(c) == cv.LOCKED and c.v.locked and c.v.failures == 3 and not c.v.verified
    assert c.spoken[-1] == cv.PROMPT["locked"] and "team" in cv.PROMPT["locked"]
    c.script = ["1", APP4, "1"]  # even the right digits are not collected any more
    assert run(c) == cv.LOCKED and not c.v.verified


def test_no_keypress_nudges_at_8s_and_ends_at_15s():
    c = Caller([""])
    assert run(c) == cv.TIMEOUT and not c.v.verified and c.v.failures == 0
    assert c.spoken == [cv.PROMPT["menu"], cv.PROMPT["nudge"], cv.PROMPT["timeout"]]


def test_key_after_nudge_continues():
    c = Caller(["", APP4, "1"], nudge_keys="1")
    assert run(c) == cv.VERIFIED


def test_timer_defaults_match_the_design():
    assert (cv.NUDGE_S, cv.GIVE_UP_S, cv.MAX_ATTEMPTS, cv.DIGITS) == (8.0, 15.0, 3, 4)


def test_presses_outside_an_attempt_are_ignored():
    c = Caller(["", ""])
    for k in "14729":
        c.v.press(k)
    assert c.v._keys.empty()


# ----------------------------------------------------------------------------------------------- exposure
def test_digits_never_in_logs(caplog):
    caplog.set_level(logging.DEBUG)
    for script in (["1", "0000", "1", "1111", "1", "2222", "1"], ["1", APP4, "1"], ["2", BANK4, "2", BANK4, "1"]):
        run(Caller(script))
    text = caplog.text
    assert "****" in text
    for d in (APP4, BANK4, "0000", "1111", "2222", "4, 7, 2, 9"):
        assert d not in text


def test_outcome_returned_to_the_llm_carries_no_digits():
    for script in (["1", APP4, "1"], ["1", "0000", "1", "1111", "1", "2222", "1"], [""]):
        out = run(Caller(script))
        assert out in (cv.VERIFIED, cv.LOCKED, cv.TIMEOUT) and not any(ch.isdigit() for ch in out)


def test_mask_and_readback_validation():
    assert cv.mask("4729") == "****" and cv.mask() == "****"
    with pytest.raises(ValueError) as e:
        cv.readback("47a9")
    assert "47a9" not in str(e.value)


# ----------------------------------------------------------------------------------------------- the gate
def test_gate_refuses_the_agent_server_until_verified():
    calls = []

    async def exchange():
        calls.append(1)
        return {"assertion": "a"}

    c = Caller(["1", APP4, "1"])
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
def test_seed_script_writes_hashes_only_and_round_trips(tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(ROOT / "scripts" / "voice"))
    import seed_caller_directory as sd
    monkeypatch.setenv(cv.KEY_ENV, KEY.decode())
    out = tmp_path / "dir.json"
    assert sd.main(["--out", str(out)]) == 0
    entries = sd.demo_entries()
    blob, printed = out.read_text(), capsys.readouterr().out
    for e in entries:
        assert e["phone"].startswith("+910000") and e["phone"] not in blob and e["phone"][1:] not in blob
        assert e["phone"] not in printed  # demo values only with --show-demo-values
    d = cv.CallerDirectory.load(out, KEY)
    assert len(d) == len(entries)
    rec = d.lookup(entries[0]["phone"])
    assert rec.application_id == entries[0]["application_id"]
    assert rec.matches(KEY, "application", entries[0]["application_last4"])
    assert rec.matches(KEY, "bank_account", entries[0]["bank_account_last4"])


def test_seed_script_refuses_without_key(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts" / "voice"))
    import seed_caller_directory as sd
    monkeypatch.delenv(cv.KEY_ENV, raising=False)
    assert sd.main(["--out", str(tmp_path / "x.json")]) == 2 and not (tmp_path / "x.json").exists()
