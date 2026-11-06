"""Keypad (DTMF) verification for phone callers of the Saral voice agent.

A phone call arrives through LiveKit SIP. The caller ID (the SIP from-number, participant attribute
`sip.phoneNumber`) is looked up in a directory of synthetic applicant records. An unknown number gets no account
information and is offered a person. A known number must then prove one factor, the one the caller picks:

    "Press 1 to verify with your loan application number, or 2 with your bank account number."
    "Enter the last 4 digits."
    "You entered 4, 7, 2, 9. Press 1 to confirm or 2 to retype."

Digits come from the KEYPAD only (LiveKit `sip_dtmf_received` room events, fed in through `KeypadVerifier.press`),
never from speech. The read-back is built from a fixed template and sent straight to TTS by the worker; the
transcript and chat context only ever see the masked form, and nothing here logs a digit (`****`).

Each factor is stored as HMAC-SHA256(key, salt|factor|last4) with a per-record salt; the key comes from the
environment (CALLER_VERIFY_HMAC_KEY), so the directory file holds hashes only. Three confirmed mismatches lock
verification for the rest of the call. No keypress for 8 s plays a nudge; 15 s ends the attempt.

DEMO-GRADE: 4 digits is 10,000 combinations and the caller ID can be spoofed. This shows the flow; it is not a
production authentication scheme.

Kept free of LiveKit imports so the security boundary is unit-testable on its own.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Iterable

log = logging.getLogger("saral.voice.verify")

KEY_ENV = "CALLER_VERIFY_HMAC_KEY"
DIRECTORY_ENV = "CALLER_DIRECTORY"
MIN_KEY_BYTES = 16
DIGITS = 4
MAX_ATTEMPTS = 3
NUDGE_S = 8.0
GIVE_UP_S = 15.0
MASK = "*" * DIGITS

FACTOR_BY_KEY = {"1": "application", "2": "bank_account"}
FACTORS = tuple(FACTOR_BY_KEY.values())

VERIFIED, LOCKED, UNKNOWN_CALLER, TIMEOUT, UNAVAILABLE, NOT_REQUIRED = (
    "verified", "locked", "unknown_caller", "timeout", "unavailable", "not_required")

PROMPT = {
    "menu": "Press 1 to verify with your loan application number, or 2 with your bank account number.",
    "menu_again": "Please press 1 for your loan application number, or 2 for your bank account number.",
    "enter": "Enter the last 4 digits.",
    "cleared": "Cleared. Enter the last 4 digits.",
    "retype": "Okay. Enter the last 4 digits again.",
    "confirm_again": "Press 1 to confirm or 2 to retype.",
    "nudge": "Please use your phone keypad.",
    "no_match": "Those digits don't match our records. Please try again.",
    "verified": "Thank you, you're verified. How can I help with your loan?",
    "locked": ("I couldn't verify you, so I can't discuss account details on this call. "
               "I can connect you to a member of our team instead."),
    "timeout": ("I didn't receive any keypresses, so I've stopped verification. I can't share account details until "
                "you're verified. Say verify to try again, or I can connect you to a member of our team."),
    "unknown_caller": ("I'm sorry, I can't find an account linked to this phone number, so I can't share account "
                       "details on this call. I can connect you to a member of our team."),
    "unavailable": ("I'm sorry, I can't verify callers right now, so I can't share account details on this call. "
                    "I can connect you to a member of our team."),
    "needs_verification": "Before I can discuss your account, I need to verify you.",
}
READBACK = "You entered {spaced}. Press 1 to confirm or 2 to retype."


def mask(_digits: str | None = None) -> str:
    """Whatever was keyed, logs and transcripts only ever show this."""
    return MASK


def readback(digits: str) -> tuple[str, str]:
    """(spoken, shown): the spoken text goes to TTS only; the shown text is what transcripts and chat may hold."""
    if not re.fullmatch(r"\d{%d}" % DIGITS, digits or ""):
        raise ValueError("read-back needs exactly 4 keypad digits")
    return READBACK.format(spaced=", ".join(digits)), READBACK.format(spaced=MASK)


# ----------------------------------------------------------------------------------------------------- hashing
def load_key(env=None) -> bytes | None:
    """The HMAC key from the environment, or None when it is missing or too short (verification then fails closed)."""
    env = os.environ if env is None else env
    raw = (env.get(KEY_ENV) or "").encode()
    return raw if len(raw) >= MIN_KEY_BYTES else None


def normalize_phone(raw: str | None) -> str | None:
    """E.164 from a SIP caller ID such as '+910000123456', 'tel:+91...' or 'sip:+91...@host'; None if not a number."""
    if not raw:
        return None
    s = str(raw).strip()
    s = re.sub(r"^(sip:|tel:)", "", s, flags=re.I).split("@", 1)[0].split(";", 1)[0]
    s = re.sub(r"[\s\-().]", "", s)
    if s.startswith("00"):
        s = "+" + s[2:]
    if not s.startswith("+"):
        s = "+" + s
    return s if re.fullmatch(r"\+\d{8,15}", s) else None


def phone_hmac(key: bytes, phone: str) -> str:
    return hmac.new(key, f"phone|{phone}".encode(), hashlib.sha256).hexdigest()


def factor_hmac(key: bytes, salt: str, factor: str, last4: str) -> str:
    if factor not in FACTORS:
        raise ValueError("unknown factor")
    return hmac.new(key, f"{salt}|{factor}|{last4}".encode(), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class CallerRecord:
    applicant_id: str
    application_id: str
    salt: str
    factor_hmacs: dict  # factor -> hex HMAC; never a digit

    def matches(self, key: bytes, factor: str, last4: str) -> bool:
        stored = self.factor_hmacs.get(factor)
        if not stored or not re.fullmatch(r"\d{%d}" % DIGITS, last4 or ""):
            return False
        return hmac.compare_digest(stored, factor_hmac(key, self.salt, factor, last4))


class CallerDirectory:
    """Synthetic applicants keyed by the HMAC of their phone number. Holds hashes only."""

    def __init__(self, key: bytes, records: dict[str, CallerRecord]):
        self.key, self._records = key, dict(records)

    def __len__(self):
        return len(self._records)

    def lookup(self, caller_id: str | None) -> CallerRecord | None:
        phone = normalize_phone(caller_id)
        return self._records.get(phone_hmac(self.key, phone)) if phone else None

    @classmethod
    def from_seed(cls, key: bytes, seed: dict) -> "CallerDirectory":
        records = {}
        for row in seed.get("records", []):
            records[row["phone_hmac"]] = CallerRecord(row["applicant_id"], row["application_id"], row["salt"],
                                                      {f: row["factors"][f] for f in FACTORS if f in row["factors"]})
        return cls(key, records)

    @classmethod
    def load(cls, path: str | Path, key: bytes) -> "CallerDirectory":
        return cls.from_seed(key, json.loads(Path(path).read_text(encoding="utf-8")))


def build_seed(key: bytes, entries: Iterable[dict], *, salt_fn: Callable[[], str] = lambda: secrets.token_hex(16)) -> dict:
    """Hash synthetic plaintext entries {phone, applicant_id, application_id, application_last4, bank_account_last4}
    into a directory seed. The output carries no phone number and no digit."""
    records = []
    for e in entries:
        phone = normalize_phone(e["phone"])
        if phone is None:
            raise ValueError("seed entry has no usable phone number")
        salt = salt_fn()
        factors = {}
        for f in FACTORS:
            last4 = str(e[f"{f}_last4"])
            if not re.fullmatch(r"\d{%d}" % DIGITS, last4):
                raise ValueError("seed factor must be 4 digits")
            factors[f] = factor_hmac(key, salt, f, last4)
        records.append({"phone_hmac": phone_hmac(key, phone), "applicant_id": e["applicant_id"],
                        "application_id": e["application_id"], "salt": salt, "factors": factors})
    return {"_warning": "SYNTHETIC FIXTURES ONLY. Hashes only (HMAC-SHA256, key from CALLER_VERIFY_HMAC_KEY).",
            "algorithm": "hmac-sha256", "records": records}


# ----------------------------------------------------------------------------------------------------- the flow
class _NoKeypress(Exception):
    pass


Speak = Callable[[str], Awaitable[None]]
SpeakPrivate = Callable[[str, str], Awaitable[None]]  # (spoken text for TTS only, masked text for transcripts)


class KeypadVerifier:
    """One per call. `press()` is fed from the room's DTMF events; `run()` drives one verification attempt and
    returns only an outcome word (never digits), which is all the LLM or the logs ever get to see."""

    def __init__(self, record: CallerRecord, key: bytes, *, speak: Speak, speak_private: SpeakPrivate,
                 max_attempts: int = MAX_ATTEMPTS, nudge_s: float = NUDGE_S, give_up_s: float = GIVE_UP_S):
        self.record, self._key = record, key
        self._speak, self._speak_private = speak, speak_private
        self.max_attempts, self.nudge_s, self.give_up_s = max_attempts, nudge_s, give_up_s
        self._keys: asyncio.Queue[str] = asyncio.Queue()
        self._running = asyncio.Lock()
        self.verified = False
        self.locked = False
        self.failures = 0
        self.factor: str | None = None

    @property
    def busy(self) -> bool:
        return self._running.locked()

    def press(self, digit: str) -> None:
        """A keypad press. Ignored unless a verification attempt is waiting for keys."""
        if self.busy and isinstance(digit, str) and len(digit) == 1 and digit in "0123456789*#":
            self._keys.put_nowait(digit)

    def _drain(self):
        while not self._keys.empty():
            self._keys.get_nowait()

    async def _next_key(self) -> str:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.give_up_s
        try:
            return await asyncio.wait_for(self._keys.get(), self.nudge_s)
        except asyncio.TimeoutError:
            pass
        await self._speak(PROMPT["nudge"])
        try:
            return await asyncio.wait_for(self._keys.get(), max(0.0, deadline - loop.time()))
        except asyncio.TimeoutError:
            raise _NoKeypress() from None

    async def _choose_factor(self) -> str:
        self._drain()
        await self._speak(PROMPT["menu"])
        while True:
            key = await self._next_key()
            if key in FACTOR_BY_KEY:
                return FACTOR_BY_KEY[key]
            await self._speak(PROMPT["menu_again"])

    async def _collect(self, prompt: str) -> str:
        self._drain()
        await self._speak(PROMPT[prompt])
        buf: list[str] = []
        while len(buf) < DIGITS:
            key = await self._next_key()
            if key.isdigit():
                buf.append(key)
            elif key == "*":
                buf.clear()
                await self._speak(PROMPT["cleared"])
        return "".join(buf)

    async def _confirm(self, digits: str) -> bool:
        self._drain()  # a fifth key or a key typed during the read-back must not count as the answer
        spoken, shown = readback(digits)
        await self._speak_private(spoken, shown)
        while True:
            key = await self._next_key()
            if key == "1":
                return True
            if key == "2":
                return False
            await self._speak(PROMPT["confirm_again"])

    async def run(self) -> str:
        if self.verified:
            return VERIFIED
        if self.locked:
            await self._speak(PROMPT["locked"])
            return LOCKED
        if self._running.locked():
            return "in_progress"
        async with self._running:
            try:
                self.factor = self.factor or await self._choose_factor()
                prompt = "enter"
                while True:
                    digits = await self._collect(prompt)
                    if not await self._confirm(digits):
                        prompt = "retype"
                        continue
                    ok = self.record.matches(self._key, self.factor, digits)
                    digits = None  # noqa: F841  drop the plaintext as soon as it has been compared
                    if ok:
                        self.verified = True
                        log.info("caller verified factor=%s digits=%s", self.factor, MASK)
                        await self._speak(PROMPT["verified"])
                        return VERIFIED
                    self.failures += 1
                    log.warning("caller verification mismatch attempt=%d/%d factor=%s digits=%s",
                                self.failures, self.max_attempts, self.factor, MASK)
                    if self.failures >= self.max_attempts:
                        self.locked = True
                        log.warning("caller verification locked for this call")
                        await self._speak(PROMPT["locked"])
                        return LOCKED
                    await self._speak(PROMPT["no_match"])
                    prompt = "enter"
            except _NoKeypress:
                log.info("caller verification ended: no keypress")
                await self._speak(PROMPT["timeout"])
                return TIMEOUT
            finally:
                self._drain()


class NotVerified(RuntimeError):
    pass


class VerifiedGate:
    """Stands where the browser path's `Assertions` stands: `fresh()` raises until the caller is verified, so the
    worker cannot reach the advisory server (and so cannot reveal an application fact) for an unverified call."""

    def __init__(self, verifier: KeypadVerifier | None, exchange: Callable[[], Awaitable[dict]] | None):
        self.verifier, self._exchange = verifier, exchange
        self.session_id = secrets.token_hex(16)  # conversation id only; not an authorisation

    @property
    def verified(self) -> bool:
        return bool(self.verifier is not None and self.verifier.verified)

    async def fresh(self) -> dict:
        if not self.verified:
            raise NotVerified("caller is not verified")
        if self._exchange is None:
            raise NotVerified("no assertion exchange for phone callers")
        return await self._exchange()
