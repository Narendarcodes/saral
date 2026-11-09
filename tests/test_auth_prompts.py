"""Pre-recorded auth-phase prompts (apps/voice/auth_prompts.py + apps/voice/prompts/): every fixed line before keypad
verification has a committed Timbre clip, clips are looked up by their exact text, the read-back is stitched in digit
order, and a missing or stale clip falls back to live TTS of the same text with a warning. No LiveKit, no network."""
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
import caller_verification as cv  # noqa: E402

CLIPS = ROOT / "apps" / "voice" / "prompts" / "en-IN"


def store_copy(tmp_path, drop=(), restamp=None):
    """A copy of the committed en-IN clips, with some removed or their manifest text changed."""
    dst = tmp_path / "en-IN"
    shutil.copytree(CLIPS, dst)
    for key in drop:
        (dst / f"{key}.wav").unlink()
    if restamp:
        man = json.loads((dst / "manifest.json").read_text())
        man["clips"][restamp]["sha256"] = ap.text_sha("some older wording")
        (dst / "manifest.json").write_text(json.dumps(man))
    return ap.PromptStore(root=tmp_path)


# ----------------------------------------------------------------------------------------------- committed clips
def test_every_auth_line_has_a_committed_clip_matching_its_text():
    store = ap.PromptStore()
    assert store.missing() == []
    man = json.loads((CLIPS / "manifest.json").read_text())
    assert man["model"] == "timbre-v3.0" and man["voice"] == "Tarika" and man["language"] == "en-IN"
    for key, text in ap.TEXT["en-IN"].items():
        assert man["clips"][key]["text"] == text and man["clips"][key]["sha256"] == ap.text_sha(text)


def test_auth_set_covers_the_whole_pre_verification_flow():
    keys = set(ap.TEXT["en-IN"])
    assert set(cv.PROMPT) <= keys
    assert {"greeting", "menu", "enter", "you_entered", "confirm", "no_match", "retype", "locked", "timeout",
            "verified"} <= keys
    assert {f"digit_{d}" for d in range(10)} <= keys
    assert "Connecting you to Saral" in cv.PROMPT["verified"]
    assert cv.PROMPT["menu"] == "Press 1 for your loan application, or 2 for your bank account."


def test_clips_are_small_mono_16k_wav():
    total = 0
    for p in CLIPS.glob("*.wav"):
        with wave.open(str(p), "rb") as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
            assert 0.2 < w.getnframes() / w.getframerate() < 12
        total += p.stat().st_size
    assert total < 3_000_000


# ----------------------------------------------------------------------------------------------- lookup
def test_lookup_by_exact_text():
    store = ap.PromptStore()
    assert store.key_for(cv.PROMPT["menu"]) == "menu"
    assert store.key_for(cv.PROMPT["menu"] + " ") is None and store.key_for("Your EMI is 12,000.") is None
    parts = store.plan(cv.PROMPT["locked"])
    assert [(p.kind, p.key) for p in parts] == [("clip", "locked")] and parts[0].sample_rate == 16000
    assert len(parts[0].pcm) > 16000  # more than half a second of 16-bit audio


def test_locales_map_to_timbre_dirs_and_only_en_is_rendered():
    assert ap.PromptStore(locale="en").locale == "en-IN" and ap.PromptStore(locale="te").locale == "te-IN"
    assert set(ap.TEXT) == {"en-IN"}


# ----------------------------------------------------------------------------------------------- stitching
def test_readback_is_stitched_in_digit_order():
    store = ap.PromptStore()
    spoken, shown = cv.readback("4729")
    parts = store.plan_readback(spoken)
    assert [p.key for p in parts if p.kind == "clip"] == ["you_entered", "digit_4", "digit_7", "digit_2", "digit_9",
                                                         "confirm"]
    assert [p.kind for p in parts] == ["clip"] + ["silence", "clip"] * 5
    gap = parts[1]
    assert set(gap.pcm) == {0} and len(gap.pcm) == 2 * 16000 * ap.GAP_MS // 1000
    assert ap.readback_keys("0099") == ["you_entered", "digit_0", "digit_0", "digit_9", "digit_9", "confirm"]
    assert shown == "You entered ****. Press 1 to confirm or 2 to retype."


def test_readback_digits_parses_only_the_fixed_template():
    spoken, shown = cv.readback("8051")
    assert cv.readback_digits(spoken) == "8051"
    assert cv.readback_digits(shown) is None and cv.readback_digits("You entered 1, 2, 3. Press 1") is None
    assert cv.readback_digits(None) is None
    with pytest.raises(ValueError):
        ap.readback_keys("12a4")


# ----------------------------------------------------------------------------------------------- fallback
def test_missing_clip_falls_back_to_live_tts_of_the_same_text(tmp_path, caplog):
    store = store_copy(tmp_path, drop=["nudge"])
    with caplog.at_level(logging.WARNING, logger="saral.voice.prompts"):
        parts = store.plan(cv.PROMPT["nudge"])
    assert [(p.kind, p.key, p.text) for p in parts] == [("tts", "nudge", cv.PROMPT["nudge"])]
    assert "clip missing" in caplog.text and "key=nudge" in caplog.text
    assert store.plan(cv.PROMPT["enter"])[0].kind == "clip"  # the others still play from clips


def test_stale_clip_is_treated_as_missing(tmp_path, caplog):
    store = store_copy(tmp_path, restamp="verified")
    with caplog.at_level(logging.WARNING, logger="saral.voice.prompts"):
        assert store.plan(cv.PROMPT["verified"])[0].kind == "tts"
    assert "key=verified" in caplog.text


def test_missing_digit_clip_reads_the_whole_line_live_without_logging_digits(tmp_path, caplog):
    store = store_copy(tmp_path, drop=["digit_7"])
    spoken, _ = cv.readback("4729")
    with caplog.at_level(logging.WARNING, logger="saral.voice.prompts"):
        parts = store.plan_readback(spoken)
    assert [(p.kind, p.text) for p in parts] == [("tts", spoken)]
    assert "read-back clips missing" in caplog.text and "4729" not in caplog.text and "4, 7" not in caplog.text
    assert store.plan_readback(cv.readback("4429")[0])[0].kind == "clip"


def test_no_clip_directory_at_all_still_speaks_every_line(tmp_path):
    store = ap.PromptStore(root=tmp_path)
    for text in cv.PROMPT.values():
        assert [p.kind for p in store.plan(text)] == ["tts"]
    assert len(store.missing()) == len(ap.TEXT["en-IN"])


def test_unknown_text_is_spoken_live_never_dropped():
    parts = ap.PromptStore().plan("Something not in the auth set.")
    assert [(p.kind, p.text) for p in parts] == [("tts", "Something not in the auth set.")]
