"""Cheap, deterministic live signals from one caller utterance (research/tone-segmentation-v3.md §4.1).

Runs on the final STT transcript (Gnani Prisma) in well under a millisecond: lexicon lookups plus the existing
`_HUMAN` / `_DISTRESS` keyword patterns from intent_router, so the voice tone engine and the agent server agree on
what a human request or a distress cue is. Nothing here reads who the caller is (no name, gender, age, area, voice
pitch, accent or STT confidence); language is used only to answer in that language.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from loan_advisory.agent.intent_router import _DISTRESS, _HOWTO, _HUMAN

LEXICON_PATH = Path(__file__).resolve().parents[5] / "config" / "tone_lexicon.json"
LOCALES = ("en", "hi", "te")
_LATIN = re.compile(r"[A-Za-z]")
_LETTER = re.compile(r"[A-Za-z\u0900-\u097F\u0C00-\u0C7F]")
_WORD = re.compile(r"[\w\u0900-\u097F\u0C00-\u0C7F']+")


@lru_cache(maxsize=4)
def load_lexicon(path: str | None = None) -> dict:
    return json.loads(Path(path or LEXICON_PATH).read_text(encoding="utf-8"))


def _phrase_hit(text: str, phrases) -> bool:
    for p in phrases:
        p = p.lower()
        if p.isascii():
            if re.search(r"(?<![a-z])" + re.escape(p) + r"(?![a-z])", text):
                return True
        elif p in text:
            return True
    return False


def locale_of(lang_code: str | None) -> str | None:
    """Prisma `detected_language_code` ("hi-IN", "te-IN", "en-IN") -> "hi" / "te" / "en", else None."""
    if not lang_code:
        return None
    base = str(lang_code).split("-")[0].lower()
    return base if base in LOCALES else None


@dataclass(frozen=True)
class Signals:
    words: int = 0
    lang: str | None = None              # detected locale of this utterance (en/hi/te) or None
    confusion: bool = False
    understood: bool = False
    anger: bool = False
    distress: bool = False
    self_harm: bool = False
    human: bool = False
    lang_request: str | None = None      # explicit "speak in X" -> en/hi/te
    latin_ratio: float = 0.0             # share of Latin letters (code-mix level)


def detect(text: str, lang_code: str | None = None, lexicon: Mapping | None = None) -> Signals:
    lex = lexicon or load_lexicon()
    raw = text or ""
    t = raw.lower().strip()
    confusion = _phrase_hit(t, lex.get("confusion", ()))
    understood = (not confusion) and _phrase_hit(t, lex.get("understood", ()))
    self_harm = _phrase_hit(t, lex.get("self_harm", ()))
    anger = _phrase_hit(t, lex.get("anger", ()))
    distress = bool(_DISTRESS.search(raw)) or _phrase_hit(t, lex.get("distress_extra", ())) or self_harm
    human = bool(_HUMAN.search(raw)) and not _HOWTO.search(raw)
    lang_request = None
    for loc, phrases in (lex.get("language_request") or {}).items():
        if _phrase_hit(t, phrases):
            lang_request = loc
            break
    letters = _LETTER.findall(raw)
    latin = len(_LATIN.findall(raw))
    return Signals(words=len(_WORD.findall(raw)), lang=locale_of(lang_code), confusion=confusion,
                   understood=understood, anger=anger, distress=distress, self_harm=self_harm, human=human,
                   lang_request=lang_request, latin_ratio=(latin / len(letters)) if letters else 0.0)
