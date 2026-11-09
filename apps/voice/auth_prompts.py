"""Pre-recorded audio for the phone authorization phase of the Saral voice agent.

Everything a phone caller hears before keypad verification succeeds is a fixed line (caller_verification.PROMPT and
the read-back template). Those lines are rendered ONCE with Gnani Timbre v3.0, voice Tarika, by
scripts/voice/render_auth_prompts.py and stored as small mono WAV clips:

    apps/voice/prompts/<locale>/<key>.wav      e.g. apps/voice/prompts/en-IN/menu.wav
    apps/voice/prompts/<locale>/manifest.json  key -> text, sha256 of the text, sample rate, duration, model, voice

During the auth phase the worker plays these clips through LiveKit and never calls the LLM or live TTS. The digit
read-back is stitched in order from `you_entered`, one `digit_<n>` clip per keyed digit and `confirm`, with a short
silence between parts. A clip that is missing, unreadable or rendered from different text (its manifest hash does
not match the current line) is treated as missing: the worker logs a warning and speaks the same fixed text with
live TTS, so a stale clip can never say something the code no longer says.

Only en-IN is rendered: the phone flow is English (the repo has no Hindi or Telugu text for these lines yet).

No LiveKit import, so lookup, stitching and fallback are unit-tested alone.
"""
from __future__ import annotations

import hashlib
import json
import logging
import wave
from dataclasses import dataclass
from pathlib import Path

import caller_verification as cv

log = logging.getLogger("saral.voice.prompts")

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_LOCALE = "en-IN"
LOCALE_DIRS = {"en": "en-IN", "hi": "hi-IN", "te": "te-IN"}
DIGIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
GAP_MS = 120          # silence between stitched read-back parts (the spoken commas)
SAMPLE_WIDTH = 2      # 16-bit PCM

# key -> text, per Timbre locale. Keys of PROMPT keep their names; the read-back adds its three kinds of part.
TEXT: dict[str, dict[str, str]] = {
    "en-IN": {**cv.PROMPT,
              "you_entered": cv.READBACK_LEAD,
              "confirm": cv.READBACK_TAIL,
              **{f"digit_{d}": DIGIT_WORDS[d].capitalize() for d in range(10)}},
}


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def readback_keys(digits: str) -> list[str]:
    """Clip keys of a read-back, in speaking order."""
    if len(digits or "") != cv.DIGITS or not digits.isdigit():
        raise ValueError("read-back needs exactly 4 keypad digits")
    return ["you_entered", *(f"digit_{d}" for d in digits), "confirm"]


@dataclass(frozen=True)
class Part:
    """One piece of a playback: a clip (pcm + rate), a silence gap (pcm of zeros), or live TTS of `text`."""
    kind: str               # clip | silence | tts
    key: str | None = None
    text: str = ""
    pcm: bytes = b""
    sample_rate: int = 0


class PromptStore:
    """Finds and loads the clips of one locale. Reads each WAV once."""

    def __init__(self, root: Path | str = PROMPTS_DIR, locale: str = DEFAULT_LOCALE):
        self.root, self.locale = Path(root), LOCALE_DIRS.get(locale, locale)
        self.texts = TEXT.get(self.locale, {})
        self._manifest: dict | None = None
        self._cache: dict[str, tuple[bytes, int] | None] = {}
        self._by_text = {}
        for k, t in self.texts.items():
            self._by_text.setdefault(t, k)

    @property
    def dir(self) -> Path:
        return self.root / self.locale

    def manifest(self) -> dict:
        if self._manifest is None:
            try:
                self._manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8")).get("clips", {})
            except (OSError, ValueError):
                self._manifest = {}
        return self._manifest

    def key_for(self, text: str) -> str | None:
        return self._by_text.get(text)

    def path(self, key: str) -> Path | None:
        """The clip file for `key`, or None when it is absent or was rendered from other text."""
        text = self.texts.get(key)
        p = self.dir / f"{key}.wav"
        if text is None or not p.is_file():
            return None
        entry = self.manifest().get(key)
        if not entry or entry.get("sha256") != text_sha(text):
            return None
        return p

    def load(self, key: str) -> tuple[bytes, int] | None:
        """(16-bit mono PCM, sample rate) of a clip, or None."""
        if key not in self._cache:
            self._cache[key] = None
            p = self.path(key)
            if p is not None:
                try:
                    with wave.open(str(p), "rb") as w:
                        if w.getnchannels() == 1 and w.getsampwidth() == SAMPLE_WIDTH:
                            self._cache[key] = (w.readframes(w.getnframes()), w.getframerate())
                except (OSError, EOFError, wave.Error):
                    pass
        return self._cache[key]

    # ------------------------------------------------------------------------------------------------ playback
    def _clips(self, keys: list[str], gap_ms: int) -> list[Part] | None:
        parts: list[Part] = []
        for i, key in enumerate(keys):
            clip = self.load(key)
            if clip is None:
                return None
            pcm, sr = clip
            if parts and parts[-1].sample_rate != sr:
                return None  # clips of one playback must share a rate
            if i and gap_ms:
                parts.append(Part("silence", sample_rate=sr, pcm=b"\x00" * (SAMPLE_WIDTH * sr * gap_ms // 1000)))
            parts.append(Part("clip", key, self.texts[key], pcm, sr))
        return parts

    def plan(self, text: str) -> list[Part]:
        """How to play one fixed line: its clip, or (warning) one live-TTS part with the same text."""
        key = self.key_for(text)
        parts = self._clips([key], 0) if key else None
        if parts:
            return parts
        log.warning("auth prompt clip missing for key=%s locale=%s; using live TTS", key or "unknown", self.locale)
        return [Part("tts", key, text)]

    def plan_readback(self, spoken: str) -> list[Part]:
        """The read-back stitched from clips in digit order, or (warning) the whole spoken line as live TTS.
        Neither the warning nor any part's key carries a digit beyond the clip name it plays."""
        digits = cv.readback_digits(spoken)
        parts = self._clips(readback_keys(digits), GAP_MS) if digits else None
        if parts:
            return parts
        log.warning("auth read-back clips missing locale=%s; using live TTS", self.locale)
        return [Part("tts", "readback", spoken)]

    def missing(self) -> list[str]:
        return [k for k in self.texts if self.load(k) is None]


__all__ = ["DEFAULT_LOCALE", "DIGIT_WORDS", "GAP_MS", "LOCALE_DIRS", "PROMPTS_DIR", "Part", "PromptStore", "TEXT",
           "readback_keys", "text_sha"]
