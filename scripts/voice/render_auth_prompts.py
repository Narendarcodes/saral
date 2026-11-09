"""Render the phone authorization prompts once with Gnani Timbre (voice Tarika) into apps/voice/prompts/<locale>/.

    GNANI_API_KEY=... python scripts/voice/render_auth_prompts.py [--locale en-IN] [--force] [--only menu,digit_4]

Uses the worker's own TTS builder (apps/voice/gnani_speech.py, livekit-plugins-gnani 1.8.11) and GNANI_* settings, with
two fixed overrides so every clip matches: model timbre-v3.0, voice Tarika, sample rate 16 kHz mono, speed 1.0.
One Timbre request per clip (about 30 for en-IN). A clip whose text is unchanged since its last render is skipped
unless --force. Leading and trailing silence is trimmed to 40 ms so stitched digits sound like one sentence, and a
digit clip keeps only its first spoken word (Timbre sometimes repeats a lone word after a pause).
Writes <key>.wav and manifest.json (key -> text, sha256 of the text, duration, model, voice, rate). The worker plays
a clip only when its manifest hash matches the current text (apps/voice/auth_prompts.py). The key is never printed.
Without GNANI_API_KEY it prints a skip line and exits 0.
"""
import argparse
import array
import asyncio
import json
import os
import sys
import time
import wave
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "apps" / "voice"), str(ROOT / "apps" / "api" / "src")]

import auth_prompts as ap  # noqa: E402

MODEL, VOICE, RATE, SPEED = "timbre-v3.0", "Tarika", 16000, 1.0
TRIM_THRESHOLD, PAD_MS = 300, 40
WORD_GAP_MS, WIN_MS = 200, 10   # a single-word clip ends at its first pause this long (Timbre can repeat a lone word)


def trim(pcm: bytes, rate: int) -> bytes:
    s = array.array("h")
    s.frombytes(pcm[: len(pcm) // 2 * 2])
    loud = [i for i, v in enumerate(s) if abs(v) > TRIM_THRESHOLD]
    if not loud:
        return pcm
    pad = rate * PAD_MS // 1000
    return s[max(0, loud[0] - pad): min(len(s), loud[-1] + pad)].tobytes()


def first_utterance(pcm: bytes, rate: int) -> bytes:
    """For one-word clips (the digits): keep only the first voiced stretch, so a repeated word or a trailing click
    after a pause is dropped."""
    s = array.array("h")
    s.frombytes(pcm[: len(pcm) // 2 * 2])
    win, need = rate * WIN_MS // 1000, WORD_GAP_MS // WIN_MS
    voiced = [max((abs(v) for v in s[i:i + win]), default=0) > TRIM_THRESHOLD for i in range(0, len(s), win)]
    if True not in voiced:
        return pcm
    start, quiet = voiced.index(True), 0
    for j in range(start, len(voiced)):
        quiet = 0 if voiced[j] else quiet + 1
        if quiet >= need:
            return s[: (j - quiet + 1) * win].tobytes()
    return pcm


async def synth(tts, text: str) -> bytes:
    pcm = bytearray()
    async with tts.synthesize(text) as stream:
        async for ev in stream:
            pcm += ev.frame.data.tobytes()
    return bytes(pcm)


async def main(args) -> int:
    if not os.environ.get("GNANI_API_KEY"):
        print("skip: GNANI_API_KEY is not set")
        return 0
    import aiohttp
    from gnani_speech import build_tts
    from loan_advisory.agent.settings import GnaniSpeechSettings

    texts = ap.TEXT.get(args.locale)
    if not texts:
        print(f"no prompt text for locale {args.locale}; known: {', '.join(ap.TEXT)}")
        return 2
    out = Path(args.out) / args.locale
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    clips = manifest.get("clips", {})
    cfg = replace(GnaniSpeechSettings.from_env(), tts_model=MODEL, tts_voice=VOICE, tts_language=args.locale,
                  tts_sample_rate=RATE, tts_speed=SPEED)
    tts = build_tts(cfg, "en")
    only = set(filter(None, (args.only or "").split(",")))
    done, skipped, failed = [], [], []
    async with aiohttp.ClientSession(trust_env=True) as http:  # honours HTTPS_PROXY where one is set
        tts._session = http  # outside a LiveKit job there is no shared http context
        for key, text in texts.items():
            if only and key not in only:
                continue
            sha = ap.text_sha(text)
            wav = out / f"{key}.wav"
            if not args.force and wav.is_file() and clips.get(key, {}).get("sha256") == sha:
                skipped.append(key)
                continue
            t0 = time.perf_counter()
            try:
                pcm = await synth(tts, text)
                if key.startswith("digit_"):
                    pcm = first_utterance(pcm, RATE)
                pcm = trim(pcm, RATE)
            except Exception as e:  # report and continue; the worker falls back to live TTS for a missing clip
                failed.append(f"{key}: {type(e).__name__}")
                continue
            with wave.open(str(wav), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(pcm)
            clips[key] = {"text": text, "sha256": sha, "duration_s": round(len(pcm) / 2 / RATE, 3)}
            done.append(f"{key} ({clips[key]['duration_s']} s, {round((time.perf_counter() - t0) * 1000)} ms)")
    manifest = {"model": MODEL, "voice": VOICE, "language": args.locale, "sample_rate": RATE, "channels": 1,
                "speed": SPEED, "clips": dict(sorted(clips.items()))}
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rendered": done, "skipped": skipped, "failed": failed}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--locale", default=ap.DEFAULT_LOCALE)
    p.add_argument("--out", default=str(ap.PROMPTS_DIR))
    p.add_argument("--force", action="store_true")
    p.add_argument("--only", help="comma list of keys")
    sys.exit(asyncio.run(main(p.parse_args())))
