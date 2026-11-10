"""Keypad (DTMF) verification for phone callers of the Saral voice agent (call flow v2: language first, then 4 digits).

A phone call arrives through LiveKit SIP. The caller ID (the SIP from-number, participant attribute
`sip.phoneNumber`) is looked up in a directory of synthetic applicant records; each record names the caller's language
(one of the ten Gnani codes, default en-IN). Then:

    greeting, in that language, with no name or account data:  "Welcome to Saral. To change language, press nine."
    9 -> language menu, each option in its own language:        1 English ... 9 Gujarati, * Punjabi, then (in the
                                                                 call's language) "To continue in <it>, press zero."
    0 (greeting or menu) -> continue in the current language at once
    one factor only:                                            "Please enter the last four digits of your loan
                                                                 application number."
    wrong digits:                                               "That didn't match. Please try again."  (no read-back)
    three mismatches lock verification for the rest of the call.

`9` changes language only while the greeting plays and for LANGUAGE_WINDOW_S after it; once the enter-digits prompt
has started, 9 is an ordinary digit (a last-4 can start with 9). `0` during the greeting (or its window) or the menu
continues in the current language at once: it cuts the prompt that is playing (when the worker can interrupt it) and
goes straight on; once the enter-digits prompt has started, 0 is an ordinary digit too. In the menu `*` picks
Punjabi. The menu plays ONCE, then waits MENU_WAIT_S for a key; with no choice the call keeps its current language
and goes on (no replay). Keys that mean nothing at that point (`#` anywhere, `*` and other digits during the
greeting) are dropped.

An unknown number (or no directory) gets the same en-IN greeting with the press-9 option, then the unknown-caller line
in whatever language the caller picked; it is never told anything about an account.

Digits come from the KEYPAD only (LiveKit `sip_dtmf_received` room events, fed in through `press`), never from speech.
Every line before verification is a fixed key of prompt_texts.PROMPT_KEYS; the worker plays it as a pre-rendered
Timbre clip in the call's language (auth_prompts.py, apps/voice/prompts/<locale>/) and never calls the LLM or live TTS
for it. Nothing here logs a digit (`****`), and `run()` returns only an outcome word.

The factor is stored as HMAC-SHA256(key, salt|application|last4) with a per-record salt; the key comes from the
environment (CALLER_VERIFY_HMAC_KEY), so the directory file holds hashes (and the plaintext language code) only.
No keypress for 8 s plays a nudge; 15 s ends the attempt. Neither counts as a mismatch.

DEMO-GRADE, SYNTHETIC DATA ONLY: 4 digits is 10,000 combinations and the caller ID can be spoofed. This shows the flow;
it is not a production authentication scheme.

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

from prompt_texts import DEFAULT_LANGUAGE, LANGUAGE_BY_KEY, LANGUAGES, PROMPT_KEYS, TEXT

log = logging.getLogger("saral.voice.verify")

KEY_ENV = "CALLER_VERIFY_HMAC_KEY"
DIRECTORY_ENV = "CALLER_DIRECTORY"
MIN_KEY_BYTES = 16
DIGITS = 4
MAX_ATTEMPTS = 3
NUDGE_S = 8.0
GIVE_UP_S = 15.0
LANGUAGE_KEY = "9"
CONTINUE_KEY = "0"        # greeting / menu: carry on in the current language now (a plain digit after that)
LANGUAGE_WINDOW_S = 3.0   # after the greeting, how long a 9 still opens the language menu
MENU_WAIT_S = 6.0         # after the menu has played (once), how long it waits for a key before keeping the language
MASK = "*" * DIGITS

FACTOR = "application"    # the one factor: last 4 digits of the loan application number
FACTORS = (FACTOR,)

VERIFIED, LOCKED, UNKNOWN_CALLER, TIMEOUT, UNAVAILABLE, NOT_REQUIRED = (
    "verified", "locked", "unknown_caller", "timeout", "unavailable", "not_required")

# The English lines, for readers and tests; the worker always speaks a key in the call's language.
PROMPT = dict(TEXT[DEFAULT_LANGUAGE])
KEYS = PROMPT_KEYS
LANGUAGE_MENU = "language_menu"   # pseudo-key: the ten `option` clips in key order


def mask(_digits: str | None = None) -> str:
    """Whatever was keyed, logs and transcripts only ever show this."""
    return MASK


def valid_language(code) -> str:
    return code if code in LANGUAGES else DEFAULT_LANGUAGE


# ----------------------------------------------------------------------------------------------------- hashing
def load_key(env=None) -> bytes | None:
    """The HMAC key from the environment, or None when it is missing or too short (verification then fails closed)."""
    env = os.environ if env is None else env
    raw = (env.get(KEY_ENV) or "").encode()
    return raw if len(raw) >= MIN_KEY_BYTES else None


def normalize_phone(raw: str | None) -> str | None:
    """E.164 from a SIP caller ID, or None if it is not a number. Accepts 'tel:'/'sip:' prefixes, '@host' and ';params',
    spaces, dashes, dots and brackets, and the usual Indian forms as ONE key: '+91 98765 43210', '919876543210',
    '09876543210' and a bare '9876543210' all give '+919876543210'. '00' is an international prefix; any other number
    without '+' is taken as already carrying its country code ('14842950295' -> '+14842950295')."""
    if not raw:
        return None
    s = str(raw).strip()
    s = re.sub(r"^(sips?:|tel:)", "", s, flags=re.I).split("@", 1)[0].split(";", 1)[0]
    s = re.sub(r"[\s\-().]", "", s)
    if not re.fullmatch(r"\+?\d+", s):
        return None
    if s.startswith("+"):
        pass
    elif len(s) == 10:                          # bare Indian national number
        s = "+91" + s
    elif len(s) == 11 and s.startswith("0"):    # Indian trunk prefix 0 + national number
        s = "+91" + s[1:]
    elif len(s) == 12 and s.startswith("91"):   # 91 + national number, no '+'
        s = "+" + s
    elif s.startswith("00"):                    # international access prefix
        s = "+" + s[2:]
    else:
        s = "+" + s
    return s if re.fullmatch(r"\+\d{8,15}", s) else None


CALLER_ID_ATTRIBUTES = ("sip.phoneNumber", "sip.from", "sip.fromUser", "sip.callerId", "sip.callerNumber")


def caller_id(attributes: dict | None, identity: str | None = None) -> str | None:
    """The caller's E.164 number from LiveKit SIP participant attributes: `sip.phoneNumber` first, then the 'from'
    style fallbacks, then a `sip_<number>` participant identity. None when nothing parses."""
    attributes = attributes or {}
    for name in CALLER_ID_ATTRIBUTES:
        phone = normalize_phone(attributes.get(name))
        if phone:
            return phone
    if identity and str(identity).startswith("sip_"):
        return normalize_phone(str(identity)[4:])
    return None


def mask_phone(phone: str | None) -> str:
    """For logs: every digit but the last two masked ('+**********10'); never the number itself."""
    if not phone:
        return "none"
    digits = re.sub(r"\D", "", str(phone))
    if len(digits) <= 2:
        return "*" * len(digits)
    return ("+" if str(phone).startswith("+") else "") + "*" * (len(digits) - 2) + digits[-2:]


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
    language: str = DEFAULT_LANGUAGE  # Gnani code the call opens in; plaintext (not sensitive)

    def matches(self, key: bytes, factor: str, last4: str) -> bool:
        stored = self.factor_hmacs.get(factor)
        if not stored or not re.fullmatch(r"\d{%d}" % DIGITS, last4 or ""):
            return False
        return hmac.compare_digest(stored, factor_hmac(key, self.salt, factor, last4))


class CallerDirectory:
    """Synthetic applicants keyed by the HMAC of their phone number. Holds hashes (and a language code) only."""

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
            # A directory from the v1 flow may still carry a bank_account hash: it is ignored, never offered.
            records[row["phone_hmac"]] = CallerRecord(row["applicant_id"], row["application_id"], row["salt"],
                                                      {f: row["factors"][f] for f in FACTORS if f in row["factors"]},
                                                      valid_language(row.get("language")))
        return cls(key, records)

    @classmethod
    def load(cls, path: str | Path, key: bytes) -> "CallerDirectory":
        return cls.from_seed(key, json.loads(Path(path).read_text(encoding="utf-8")))


def build_seed(key: bytes, entries: Iterable[dict], *, salt_fn: Callable[[], str] = lambda: secrets.token_hex(16)) -> dict:
    """Hash synthetic plaintext entries {phone, applicant_id, application_id, application_last4, language?} into a
    directory seed. `language` is one of the ten Gnani codes (default en-IN). The output carries no phone number and
    no digit."""
    records = []
    for e in entries:
        phone = normalize_phone(e["phone"])
        if phone is None:
            raise ValueError("seed entry has no usable phone number")
        language = e.get("language") or DEFAULT_LANGUAGE
        if language not in LANGUAGES:
            raise ValueError(f"seed language must be one of {', '.join(LANGUAGES)}")
        salt = salt_fn()
        last4 = str(e["application_last4"])
        if not re.fullmatch(r"\d{%d}" % DIGITS, last4):
            raise ValueError("seed factor must be 4 digits")
        records.append({"phone_hmac": phone_hmac(key, phone), "applicant_id": e["applicant_id"],
                        "application_id": e["application_id"], "salt": salt, "language": language,
                        "factors": {FACTOR: factor_hmac(key, salt, FACTOR, last4)}})
    return {"_warning": "SYNTHETIC FIXTURES ONLY. Hashes only (HMAC-SHA256, key from CALLER_VERIFY_HMAC_KEY).",
            "algorithm": "hmac-sha256", "records": records}


# ----------------------------------------------------------------------------------------------------- the flow
class _NoKeypress(Exception):
    pass


Speak = Callable[[str], Awaitable[None]]  # a PROMPT_KEYS key (or LANGUAGE_MENU), played in the call's language
Interrupt = Callable[[], None]            # stop the prompt that is playing now (the worker's SpeechHandle.interrupt)


class KeypadCall:
    """The keypad side of one call: key queue, timeouts, the greeting and the press-9 language menu. Used alone for
    an unknown caller (nothing to verify) and as the base of `KeypadVerifier`."""

    def __init__(self, *, speak: Speak, language: str = DEFAULT_LANGUAGE,
                 on_language: Callable[[str], None] | None = None, nudge_s: float = NUDGE_S,
                 give_up_s: float = GIVE_UP_S, window_s: float = LANGUAGE_WINDOW_S, menu_wait_s: float = MENU_WAIT_S,
                 interrupt: Interrupt | None = None):
        self._speak, self._on_language, self._interrupt = speak, on_language, interrupt
        self.language = valid_language(language)
        self.nudge_s, self.give_up_s, self.window_s, self.menu_wait_s = nudge_s, give_up_s, window_s, menu_wait_s
        self._keys: asyncio.Queue[str] = asyncio.Queue()
        self._running = asyncio.Lock()
        self.greeted = False
        self.step = 0              # one per prompt that waits for keys (greeting, menu, enter, no_match, cleared)
        self._nudged_step = -1     # the step that already had its one "please use your keypad"

    async def _prompt(self, key: str) -> None:
        self.step += 1
        await self._speak(key)

    def may_nudge(self) -> bool:
        """True once per step: the keypad nudge (from the 8 s timer or from the caller talking) plays at most once
        per prompt, never on every utterance."""
        if self._nudged_step == self.step:
            return False
        self._nudged_step = self.step
        return True

    @property
    def busy(self) -> bool:
        return self._running.locked()

    def press(self, digit: str) -> None:
        """A keypad press. Ignored unless the call is waiting for keys."""
        if self.busy and isinstance(digit, str) and len(digit) == 1 and digit in "0123456789*#":
            self._keys.put_nowait(digit)

    def _drain(self):
        while not self._keys.empty():
            self._keys.get_nowait()

    async def _key_within(self, seconds: float) -> str | None:
        try:
            return await asyncio.wait_for(self._keys.get(), max(0.0, seconds))
        except asyncio.TimeoutError:
            return None

    async def _next_key(self) -> str:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.give_up_s
        key = await self._key_within(self.nudge_s)
        if key is not None:
            return key
        if self.may_nudge():
            await self._speak("nudge")
        key = await self._key_within(deadline - loop.time())
        if key is None:
            raise _NoKeypress()
        return key

    def _set_language(self, code: str) -> None:
        self.language = code
        log.info("call language set to %s", code)
        if self._on_language is not None:
            self._on_language(code)

    async def _speak_listening(self, key: str, wanted: Callable[[str], bool]) -> str | None:
        """Play prompt `key` while listening to the keypad. The first key for which `wanted` is true stops the prompt
        (through the worker's `interrupt`, when given) and is returned; any other key is dropped. None when the prompt
        played to its end without a wanted key."""
        self.step += 1
        speech = asyncio.ensure_future(self._speak(key))
        try:
            while True:
                getter = asyncio.ensure_future(self._keys.get())
                done, _ = await asyncio.wait({speech, getter}, return_when=asyncio.FIRST_COMPLETED)
                if getter in done:
                    pressed = getter.result()
                    if wanted(pressed):
                        if not speech.done() and self._interrupt is not None:
                            try:
                                self._interrupt()
                            except Exception:  # the prompt then just plays out
                                log.warning("could not cut the prompt short")
                        await speech
                        return pressed
                    if not speech.done():
                        continue
                else:
                    getter.cancel()
                await speech  # re-raises a failure of the prompt itself
                return None
        finally:
            if not speech.done():
                speech.cancel()

    async def _language_menu(self) -> None:
        """Ten options, each in its own language, then "to continue in <current language>, press zero", played ONCE.
        A language key (1-9, `*` for Punjabi; during it or within `menu_wait_s` after) switches; `0` or no key keeps
        the current language and goes on. `#` is dropped."""
        def wanted(k: str) -> bool:
            return k in LANGUAGE_BY_KEY or k == CONTINUE_KEY
        self._drain()
        key = await self._speak_listening(LANGUAGE_MENU, wanted)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.menu_wait_s
        while key is None or not wanted(key):
            key = await self._key_within(deadline - loop.time())
            if key is None:
                log.info("language menu: no choice; keeping %s", self.language)
                return
        if key == CONTINUE_KEY:
            log.info("language menu: 0 pressed; keeping %s", self.language)
            return
        self._set_language(LANGUAGE_BY_KEY[key])

    async def _greet(self) -> None:
        """The opening line, once per call. A 9 keyed during it or within `window_s` after it opens the menu; a `0`
        goes straight on in the current language (cutting the greeting short); any other early key is dropped (the
        enter-digits prompt has not been played yet)."""
        if self.greeted:
            return
        self.greeted = True
        self._drain()
        key = await self._speak_listening("greeting", lambda k: k in (LANGUAGE_KEY, CONTINUE_KEY))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.window_s
        while key not in (LANGUAGE_KEY, CONTINUE_KEY):
            key = await self._key_within(deadline - loop.time())
            if key is None:
                return
        if key == LANGUAGE_KEY:
            await self._language_menu()

    async def run_unverifiable(self, reason: str) -> str:
        """Unknown caller / no directory: greeting (with the press-9 option), then the `reason` line."""
        async with self._running:
            try:
                await self._greet()
                await self._speak(reason)
            finally:
                self._drain()
        return reason


class KeypadVerifier(KeypadCall):
    """One per call for a caller ID found in the directory. `press()` is fed from the room's DTMF events; `run()`
    greets once, offers the language menu, then drives one verification attempt and returns only an outcome word
    (never digits), which is all the LLM or the logs ever get to see."""

    def __init__(self, record: CallerRecord, key: bytes, *, speak: Speak,
                 on_language: Callable[[str], None] | None = None, max_attempts: int = MAX_ATTEMPTS,
                 nudge_s: float = NUDGE_S, give_up_s: float = GIVE_UP_S, window_s: float = LANGUAGE_WINDOW_S,
                 menu_wait_s: float = MENU_WAIT_S, interrupt: Interrupt | None = None):
        super().__init__(speak=speak, language=record.language, on_language=on_language, nudge_s=nudge_s,
                         give_up_s=give_up_s, window_s=window_s, menu_wait_s=menu_wait_s, interrupt=interrupt)
        self.record, self._key = record, key
        self.max_attempts = max_attempts
        self.verified = False
        self.locked = False
        self.failures = 0
        self.factor = FACTOR

    async def _collect(self, prompt: str) -> str:
        self._drain()
        await self._prompt(prompt)
        buf: list[str] = []
        while len(buf) < DIGITS:
            key = await self._next_key()
            if key.isdigit():
                buf.append(key)
            elif key == "*":
                buf.clear()
                await self._prompt("cleared")
        return "".join(buf)

    async def run(self) -> str:
        if self.verified:
            return VERIFIED
        if self.locked:
            return LOCKED  # the locked line was spoken once already; the worker ends the call after it
        if self._running.locked():
            return "in_progress"
        async with self._running:
            try:
                await self._greet()
                prompt = "enter"
                while True:
                    digits = await self._collect(prompt)
                    ok = self.record.matches(self._key, FACTOR, digits)
                    digits = None  # noqa: F841  drop the plaintext as soon as it has been compared
                    if ok:
                        self.verified = True
                        log.info("caller verified factor=%s digits=%s", FACTOR, MASK)
                        await self._speak("verified")
                        return VERIFIED
                    self.failures += 1
                    log.warning("caller verification mismatch attempt=%d/%d factor=%s digits=%s",
                                self.failures, self.max_attempts, FACTOR, MASK)
                    if self.failures >= self.max_attempts:
                        self.locked = True
                        log.warning("caller verification locked for this call")
                        await self._speak("locked")
                        return LOCKED
                    prompt = "no_match"
            except _NoKeypress:
                log.info("caller verification ended: no keypress")
                await self._speak("timeout")
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


def dtmf_handler(caller_identity: str, verifier: KeypadCall | None):
    """Listener for the LiveKit room event `sip_dtmf_received` (livekit.rtc.SipDTMF: .digit, .code, .participant).
    Only the caller's own presses count; nothing is logged and nothing reaches the chat context."""
    def _on(ev) -> None:
        p = getattr(ev, "participant", None)
        if verifier is None or p is None or getattr(p, "identity", None) != caller_identity:
            return
        digit = getattr(ev, "digit", "") or ""
        if not digit:  # RFC 4733 event codes: 0-9 digits, 10 '*', 11 '#'
            code = getattr(ev, "code", None)
            digit = "0123456789*#"[code] if isinstance(code, int) and 0 <= code <= 11 else ""
        verifier.press(digit)
    return _on


DTMF_TOPIC = "dtmf"
KEYPAD_KEYS = frozenset("0123456789*#")


def dtmf_data_handler(caller_identity: str, verifier: KeypadCall | None):
    """Fallback keypad path for the portal's WEB PHONE only: a LiveKit `data_received` listener for packets on topic
    "dtmf" carrying {"digit": "<one of 0-9 * #>"}, from the web-phone participant itself. Used when a site sets the
    browser to send keypad presses as data instead of `publishDtmf` (which already arrives as `sip_dtmf_received`).
    The worker registers it only for web-phone calls, never for SIP or voice mode. Anything else (another
    participant, another topic, a malformed body, more than one key) is dropped; nothing is logged."""
    def _on(packet) -> None:
        p = getattr(packet, "participant", None)
        if verifier is None or getattr(packet, "topic", None) != DTMF_TOPIC or p is None \
                or getattr(p, "identity", None) != caller_identity:
            return
        try:
            body = json.loads(bytes(getattr(packet, "data", b"") or b"")[:64].decode("utf-8"))
        except (ValueError, UnicodeDecodeError, TypeError):
            return
        digit = body.get("digit") if isinstance(body, dict) else None
        if isinstance(digit, str) and digit in KEYPAD_KEYS:
            verifier.press(digit)
    return _on
