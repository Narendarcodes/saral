"""Render the phone authorization prompts once with Gnani Timbre (voice Tarika) into apps/voice/prompts/<locale>/.

    GNANI_API_KEY=... python scripts/voice/render_auth_prompts.py [--locale all|te-IN] [--force] [--only enter,option]

Uses the worker's own TTS builder (apps/voice/gnani_speech.py, livekit-plugins-gnani 1.8.11) and GNANI_* settings, with
fixed overrides so every clip matches: model timbre-v3.0, voice Tarika, the clip's own language, sample rate 16 kHz
mono, speed 1.0. One Timbre request per clip (12 per language, 120 for all ten). A clip whose text is unchanged since
its last render is skipped unless --force. Leading and trailing silence is trimmed to 40 ms. A clip no longer in the
text set (e.g. the v1 factor menu, read-back and digit clips) is deleted with --prune.
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


def trim(pcm: bytes, rate: int) -> bytes:
    s = array.array("h")
    s.frombytes(pcm[: len(pcm) // 2 * 2])
    loud = [i for i, v in enumerate(s) if abs(v) > TRIM_THRESHOLD]
    if not loud:
        return pcm
    pad = rate * PAD_MS // 1000
    return s[max(0, loud[0] - pad): min(len(s), loud[-1] + pad)].tobytes()


async def synth(tts, text: str) -> bytes:
    pcm = bytearray()
    async with tts.synthesize(text) as stream:
        async for ev in stream:
            pcm += ev.frame.data.tobytes()
    return bytes(pcm)


async def render_locale(tts, locale: str, root: Path, args) -> dict:
    texts = ap.TEXT[locale]
    out = root / locale
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    clips = {k: v for k, v in manifest.get("clips", {}).items() if k in texts}
    only = set(filter(None, (args.only or "").split(",")))
    done, skipped, failed, pruned = [], [], [], []
    tts.update_options(language=locale)
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
            pcm = trim(await synth(tts, text), RATE)
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
    if args.prune:
        for wav in sorted(out.glob("*.wav")):
            if wav.stem not in texts:
                wav.unlink()
                pruned.append(wav.stem)
    manifest = {"model": MODEL, "voice": VOICE, "language": locale, "sample_rate": RATE, "channels": 1,
                "speed": SPEED, "clips": dict(sorted(clips.items()))}
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"rendered": done, "skipped": skipped, "failed": failed, "pruned": pruned}


async def main(args) -> int:
    if not os.environ.get("GNANI_API_KEY"):
        print("skip: GNANI_API_KEY is not set")
        return 0
    import aiohttp
    from gnani_speech import build_tts
    from loan_advisory.agent.settings import GnaniSpeechSettings

    locales = list(ap.LOCALES) if args.locale == "all" else [args.locale]
    unknown = [loc for loc in locales if loc not in ap.TEXT]
    if unknown:
        print(f"no prompt text for locale {', '.join(unknown)}; known: {', '.join(ap.TEXT)}")
        return 2
    cfg = replace(GnaniSpeechSettings.from_env(), tts_model=MODEL, tts_voice=VOICE, tts_language=locales[0],
                  tts_sample_rate=RATE, tts_speed=SPEED)
    tts = build_tts(cfg, "en")
    report = {}
    async with aiohttp.ClientSession(trust_env=True) as http:  # honours HTTPS_PROXY where one is set
        tts._session = http  # outside a LiveKit job there is no shared http context
        for loc in locales:
            report[loc] = await render_locale(tts, loc, Path(args.out), args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if any(r["failed"] for r in report.values()) else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--locale", default="all", help="a Gnani code such as te-IN, or all (default)")
    p.add_argument("--out", default=str(ap.PROMPTS_DIR))
    p.add_argument("--force", action="store_true")
    p.add_argument("--only", help="comma list of keys")
    p.add_argument("--prune", action="store_true", help="delete clips whose key is no longer in the text set")
    sys.exit(asyncio.run(main(p.parse_args())))
