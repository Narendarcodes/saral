"""Pre-recorded audio for the phone authorization phase of the Saral voice agent, in ten languages.

Everything a phone caller hears before keypad verification succeeds is a fixed line (prompt_texts.PROMPT_KEYS, in the
call's language). Those lines are rendered ONCE with Gnani Timbre v3.0, voice Tarika, by
scripts/voice/render_auth_prompts.py and stored as small mono WAV clips:

    apps/voice/prompts/<locale>/<key>.wav      e.g. apps/voice/prompts/te-IN/enter.wav
    apps/voice/prompts/<locale>/manifest.json  key -> text, sha256 of the text, duration, model, voice, rate

The language menu is the ten `option` clips (one per locale, each in its own language) in key order, with a short
silence between them. During the auth phase the worker plays these clips through LiveKit and never calls the LLM or
live TTS. A clip that is missing, unreadable or rendered from different text (its manifest hash does not match the
current line) is treated as missing: the worker logs a warning and speaks the same fixed text with live TTS in that
clip's language, so a stale clip can never say something the code no longer says.

No LiveKit import, so lookup, stitching and fallback are unit-tested alone.
"""
from __future__ import annotations

import hashlib
import json
import logging
import wave
from dataclasses import dataclass
from pathlib import Path

import prompt_texts as pt

log = logging.getLogger("saral.voice.prompts")

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_LOCALE = pt.DEFAULT_LANGUAGE
LOCALES = pt.LANGUAGES
LOCALE_DIRS = {code.split("-")[0]: code for code in LOCALES}  # "te" -> "te-IN"
LANGUAGE_MENU = "language_menu"
GAP_MS = 250          # silence between language-menu options
SAMPLE_WIDTH = 2      # 16-bit PCM

TEXT: dict[str, dict[str, str]] = {code: dict(pt.TEXT[code]) for code in LOCALES}


def locale_dir(locale: str) -> str:
    return LOCALE_DIRS.get(locale, locale) if (locale in LOCALES or locale in LOCALE_DIRS) else DEFAULT_LOCALE


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Part:
    """One piece of a playback: a clip (pcm + rate), a silence gap (pcm of zeros), or live TTS of `text` in
    `language`."""
    kind: str               # clip | silence | tts
    key: str | None = None
    text: str = ""
    pcm: bytes = b""
    sample_rate: int = 0
    language: str = ""


class PromptStore:
    """Finds and loads the clips of every locale. Reads each manifest and WAV once. `locale` is only the default for
    calls that do not name one."""

    def __init__(self, root: Path | str = PROMPTS_DIR, locale: str = DEFAULT_LOCALE):
        self.root, self.locale = Path(root), locale_dir(locale)
        self._manifest: dict[str, dict] = {}
        self._cache: dict[tuple[str, str], tuple[bytes, int] | None] = {}

    def dir(self, locale: str | None = None) -> Path:
        return self.root / locale_dir(locale or self.locale)

    def manifest(self, locale: str | None = None) -> dict:
        loc = locale_dir(locale or self.locale)
        if loc not in self._manifest:
            try:
                self._manifest[loc] = json.loads((self.dir(loc) / "manifest.json").read_text(encoding="utf-8")).get(
                    "clips", {})
            except (OSError, ValueError):
                self._manifest[loc] = {}
        return self._manifest[loc]

    def text(self, key: str, locale: str | None = None) -> str | None:
        return TEXT.get(locale_dir(locale or self.locale), {}).get(key)

    def key_for(self, text: str, locale: str | None = None) -> str | None:
        """The key whose line in `locale` is exactly `text`."""
        for k, t in TEXT.get(locale_dir(locale or self.locale), {}).items():
            if t == text:
                return k
        return None

    def path(self, key: str, locale: str | None = None) -> Path | None:
        """The clip file for `key`, or None when it is absent or was rendered from other text."""
        loc = locale_dir(locale or self.locale)
        text = self.text(key, loc)
        p = self.dir(loc) / f"{key}.wav"
        if text is None or not p.is_file():
            return None
        entry = self.manifest(loc).get(key)
        if not entry or entry.get("sha256") != text_sha(text):
            return None
        return p

    def load(self, key: str, locale: str | None = None) -> tuple[bytes, int] | None:
        """(16-bit mono PCM, sample rate) of a clip, or None."""
        loc = locale_dir(locale or self.locale)
        if (loc, key) not in self._cache:
            self._cache[(loc, key)] = None
            p = self.path(key, loc)
            if p is not None:
                try:
                    with wave.open(str(p), "rb") as w:
                        if w.getnchannels() == 1 and w.getsampwidth() == SAMPLE_WIDTH:
                            self._cache[(loc, key)] = (w.readframes(w.getnframes()), w.getframerate())
                except (OSError, EOFError, wave.Error):
                    pass
        return self._cache[(loc, key)]

    # ------------------------------------------------------------------------------------------------ playback
    def _part(self, key: str, locale: str) -> Part:
        clip = self.load(key, locale)
        text = self.text(key, locale) or ""
        if clip is None:
            log.warning("auth prompt clip missing for key=%s locale=%s; using live TTS", key, locale)
            return Part("tts", key, text, language=locale)
        return Part("clip", key, text, clip[0], clip[1], language=locale)

    def plan_key(self, key: str, locale: str | None = None) -> list[Part]:
        """How to play one fixed line in `locale`: its clip, or (warning) live TTS of the same text."""
        loc = locale_dir(locale or self.locale)
        if key == LANGUAGE_MENU:
            return self.plan_menu()
        if self.text(key, loc) is None:
            raise KeyError(f"no auth prompt {key!r}")
        return [self._part(key, loc)]

    def plan(self, text: str, locale: str | None = None) -> list[Part]:
        """A fixed line given by its text (as the llm_node gate yields it). Unknown text is spoken live, never
        dropped."""
        loc = locale_dir(locale or self.locale)
        key = self.key_for(text, loc)
        if key is not None:
            return [self._part(key, loc)]
        log.warning("auth prompt clip missing for key=unknown locale=%s; using live TTS", loc)
        return [Part("tts", None, text, language=loc)]

    def plan_menu(self) -> list[Part]:
        """The language menu: every locale's `option` line, in key order, each in its own language, with a gap.
        A missing option clip becomes live TTS in that option's language; the others still play from clips."""
        parts: list[Part] = []
        rate = None
        for loc in LOCALES:
            part = self._part("option", loc)
            if part.kind == "clip":
                if rate is None:
                    rate = part.sample_rate
            if parts and GAP_MS:
                sr = rate or 16000
                parts.append(Part("silence", sample_rate=sr, pcm=b"\x00" * (SAMPLE_WIDTH * sr * GAP_MS // 1000)))
            parts.append(part)
        return parts

    def missing(self, locale: str | None = None) -> list[str]:
        """Keys without a usable clip, in one locale (or `locale/key` across all locales when None)."""
        if locale is not None:
            loc = locale_dir(locale)
            return [k for k in TEXT[loc] if self.load(k, loc) is None]
        return [f"{loc}/{k}" for loc in LOCALES for k in TEXT[loc] if self.load(k, loc) is None]


__all__ = ["DEFAULT_LOCALE", "GAP_MS", "LANGUAGE_MENU", "LOCALES", "LOCALE_DIRS", "PROMPTS_DIR", "Part", "PromptStore",
           "TEXT", "locale_dir", "text_sha"]
