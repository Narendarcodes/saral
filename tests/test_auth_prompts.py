"""Pre-recorded auth-phase prompts in ten languages (apps/voice/auth_prompts.py + apps/voice/prompts/<locale>/): every
fixed line before keypad verification has a committed Timbre clip in every language, clips are looked up by key or
exact text per language, the language menu plays each option in its own language, and a missing or stale clip falls
back to live TTS of the same text in its language with a warning. No LiveKit, no network."""
import json
import logging
import shutil
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "voice"))
import auth_prompts as ap  # noqa: E402
import prompt_texts as pt  # noqa: E402

PROMPTS = ROOT / "apps" / "voice" / "prompts"


def store_copy(tmp_path, locale="en-IN", drop=(), restamp=None):
    """A copy of the committed clips, with some of one locale's removed or their manifest text changed."""
    dst = tmp_path / "prompts"
    shutil.copytree(PROMPTS, dst)
    for key in drop:
        (dst / locale / f"{key}.wav").unlink()
    if restamp:
        man_path = dst / locale / "manifest.json"
        man = json.loads(man_path.read_text())
        man["clips"][restamp]["sha256"] = ap.text_sha("some older wording")
        man_path.write_text(json.dumps(man))
    return ap.PromptStore(root=dst)


# ----------------------------------------------------------------------------------------------- committed clips
def test_ten_languages():
    assert ap.LOCALES == ("en-IN", "hi-IN", "te-IN", "ta-IN", "kn-IN", "ml-IN", "mr-IN", "bn-IN", "gu-IN", "pa-IN")
    assert sorted(p.name for p in PROMPTS.iterdir() if p.is_dir()) == sorted(ap.LOCALES)


@pytest.mark.parametrize("locale", ap.LOCALES)
def test_every_auth_line_has_a_committed_clip_matching_its_text(locale):
    store = ap.PromptStore()
    assert store.missing(locale) == []
    man = json.loads((PROMPTS / locale / "manifest.json").read_text())
    assert man["model"] == "timbre-v3.0" and man["voice"] == "Tarika" and man["language"] == locale
    assert set(man["clips"]) == set(pt.PROMPT_KEYS)
    for key, text in ap.TEXT[locale].items():
        assert man["clips"][key]["text"] == text and man["clips"][key]["sha256"] == ap.text_sha(text)


def test_v1_clips_are_gone():
    for locale in ap.LOCALES:
        names = {p.stem for p in (PROMPTS / locale).glob("*.wav")}
        assert names == set(pt.PROMPT_KEYS)
        assert not names & {"menu", "menu_again", "you_entered", "confirm", "retype", "confirm_again", "digit_0"}


def test_clips_are_mono_16k_wav():
    total = 0
    for p in PROMPTS.glob("*/*.wav"):
        with wave.open(str(p), "rb") as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
            assert 0.5 < w.getnframes() / w.getframerate() < 16
        total += p.stat().st_size
    assert total < 30_000_000


# ----------------------------------------------------------------------------------------------- lookup
def test_plan_key_in_each_language():
    store = ap.PromptStore()
    for locale in ap.LOCALES:
        parts = store.plan_key("enter", locale)
        assert [(p.kind, p.key, p.language) for p in parts] == [("clip", "enter", locale)]
        assert parts[0].sample_rate == 16000 and len(parts[0].pcm) > 16000
    with pytest.raises(KeyError):
        store.plan_key("menu", "en-IN")  # the v1 factor menu no longer exists


def test_lookup_by_exact_text_is_per_language():
    store = ap.PromptStore()
    te = pt.TEXT["te-IN"]["locked"]
    assert store.key_for(te, "te-IN") == "locked" and store.key_for(te, "en-IN") is None
    assert store.plan(te, "te-IN")[0].kind == "clip"
    assert store.key_for(pt.TEXT["en-IN"]["locked"] + " ", "en-IN") is None


def test_short_locales_map_to_dirs():
    assert ap.locale_dir("te") == "te-IN" and ap.locale_dir("en") == "en-IN" and ap.locale_dir("xx") == "en-IN"


def test_language_menu_plays_each_option_in_its_own_language_in_key_order():
    parts = ap.PromptStore().plan_menu()
    clips = [p for p in parts if p.kind == "clip"]
    assert [p.language for p in clips] == list(pt.LANGUAGE_BY_KEY.values())
    assert [p.text for p in clips] == [pt.TEXT[c]["option"] for c in pt.LANGUAGES]
    assert [p.kind for p in parts] == ["clip"] + ["silence", "clip"] * 9
    assert set(parts[1].pcm) == {0}
    assert ap.PromptStore().plan_key(ap.LANGUAGE_MENU, "ta-IN") == parts  # same menu whatever the call language


# ----------------------------------------------------------------------------------------------- fallback
def test_missing_clip_falls_back_to_live_tts_of_the_same_text_in_its_language(tmp_path, caplog):
    store = store_copy(tmp_path, "kn-IN", drop=["nudge"])
    with caplog.at_level(logging.WARNING, logger="saral.voice.prompts"):
        parts = store.plan_key("nudge", "kn-IN")
    assert [(p.kind, p.key, p.text, p.language) for p in parts] == [("tts", "nudge", pt.TEXT["kn-IN"]["nudge"],
                                                                     "kn-IN")]
    assert "clip missing" in caplog.text and "key=nudge" in caplog.text and "locale=kn-IN" in caplog.text
    assert store.plan_key("nudge", "en-IN")[0].kind == "clip"


def test_stale_clip_is_treated_as_missing(tmp_path, caplog):
    store = store_copy(tmp_path, "hi-IN", restamp="verified")
    with caplog.at_level(logging.WARNING, logger="saral.voice.prompts"):
        assert store.plan_key("verified", "hi-IN")[0].kind == "tts"
    assert "key=verified" in caplog.text


def test_missing_menu_option_is_spoken_live_in_that_options_language(tmp_path):
    store = store_copy(tmp_path, "pa-IN", drop=["option"])
    parts = [p for p in store.plan_menu() if p.kind != "silence"]
    assert [(p.kind, p.language) for p in parts][-1] == ("tts", "pa-IN")
    assert all(p.kind == "clip" for p in parts[:-1])


def test_no_clip_directory_at_all_still_speaks_every_line(tmp_path):
    store = ap.PromptStore(root=tmp_path)
    for locale in ap.LOCALES:
        for key in pt.PROMPT_KEYS:
            assert [p.kind for p in store.plan_key(key, locale)] == ["tts"]
    assert len(store.missing()) == len(ap.LOCALES) * len(pt.PROMPT_KEYS)


def test_unknown_text_is_spoken_live_never_dropped():
    parts = ap.PromptStore().plan("Something not in the auth set.", "en-IN")
    assert [(p.kind, p.text) for p in parts] == [("tts", "Something not in the auth set.")]
