"""Gnani speech round trip: one Timbre TTS line, then one Prisma streaming transcription of that audio.

    GNANI_API_KEY=... python scripts/voice/gnani_roundtrip.py --out /tmp/gnani_rt [--locale en|hi|te] [--text "..."]

Uses the worker's own builders (apps/voice/gnani_speech.py) and GNANI_* settings, so it checks exactly what the voice
worker will run. Without GNANI_API_KEY it prints a skip line and exits 0. It makes exactly two Gnani calls (one TTS
request, one STT session), writes the synthesized clip as WAV and a JSON summary; the key is never printed.
"""
import argparse
import asyncio
import json
import os
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "apps" / "voice"), str(ROOT / "apps" / "api" / "src")]

TEXT = {"en": "Your loan application is under review. We will update you within two working days.",
        "hi": "आपका लोन आवेदन समीक्षा में है। हम दो कार्य दिवसों में आपको सूचित करेंगे।",
        "te": "మీ లోన్ దరఖాస్తు పరిశీలనలో ఉంది. రెండు పని దినాల్లో మీకు తెలియజేస్తాము."}


async def synth(cfg, locale, text):
    import aiohttp
    from gnani_speech import build_tts
    tts = build_tts(cfg, locale)
    pcm, first, t0 = bytearray(), None, time.perf_counter()
    async with aiohttp.ClientSession() as http:
        tts._session = http  # outside a LiveKit job there is no shared http context
        async with tts.synthesize(text) as stream:
            async for ev in stream:
                first = first or time.perf_counter()
                pcm += ev.frame.data.tobytes()
    done = time.perf_counter()
    return bytes(pcm), tts.sample_rate, {"model": tts.model, "voice": tts.voice, "language": tts.language, "speed": tts.speed,
                                         "first_audio_ms": round((first - t0) * 1000, 1) if first else None,
                                         "total_ms": round((done - t0) * 1000, 1), "audio_s": round(len(pcm) / 2 / tts.sample_rate, 2)}


async def transcribe(cfg, locale, pcm, sr):
    from livekit import rtc
    from livekit.agents import stt as lk_stt
    from gnani_speech import build_stt
    stt = build_stt(cfg, locale)
    stream = stt.stream()
    n = sr // 50  # 20 ms frames, paced in real time
    texts, t_end = [], None

    async def feed():
        nonlocal t_end
        silence = b"\x00\x00" * n
        for i in range(0, len(pcm), n * 2):
            chunk = pcm[i:i + n * 2].ljust(n * 2, b"\x00")
            stream.push_frame(rtc.AudioFrame(chunk, sr, 1, n))
            await asyncio.sleep(0.02)
        t_end = time.perf_counter()
        for _ in range(75):  # 1.5 s of silence so the server closes the segment
            stream.push_frame(rtc.AudioFrame(silence, sr, 1, n))
            await asyncio.sleep(0.02)

    async def read():
        async for ev in stream:
            if ev.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives[0].text:
                texts.append((time.perf_counter(), ev.alternatives[0].text))
                return

    feeder = asyncio.create_task(feed())
    try:
        await asyncio.wait_for(read(), timeout=30)
    finally:
        feeder.cancel()
        await stream.aclose()
    t_text = texts[0][0] if texts else None
    return {"model": stt.model, "language_code": stt.language_code, "format": stt.output_format,
            "transcript": texts[0][1] if texts else None,
            "after_audio_end_ms": round((t_text - t_end) * 1000, 1) if t_text and t_end else None}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--locale", default="en", choices=sorted(TEXT))
    ap.add_argument("--text")
    a = ap.parse_args()
    if not os.environ.get("GNANI_API_KEY"):
        print("SKIP: GNANI_API_KEY is not set; no Gnani call made")
        return 0
    from loan_advisory.agent.settings import GnaniSpeechSettings
    cfg = GnaniSpeechSettings.from_env()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    text = a.text or TEXT[a.locale]
    summary = {"locale": a.locale, "text": text}
    try:
        pcm, sr, summary["tts"] = await synth(cfg, a.locale, text)
        wav = out / f"timbre_{a.locale}.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm)
        summary["tts"]["wav"] = str(wav)
    except Exception as e:  # report and stop: no STT call without audio
        summary["tts_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        pcm = None
    if pcm:
        try:
            summary["stt"] = await transcribe(cfg, a.locale, pcm, sr)
        except Exception as e:
            summary["stt_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    (out / "gnani_roundtrip.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if pcm and summary.get("stt", {}).get("transcript") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
