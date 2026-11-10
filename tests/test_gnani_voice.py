"""Gnani speech in the voice worker: Prisma (STT) + Timbre (TTS) configuration and plugin wiring.

The settings tests need nothing but the repo. The wiring tests need `livekit-plugins-gnani` (apps/voice/
requirements.txt) and skip without it; they never reach Gnani: the network calls are replaced by fakes and the
API key is a dummy set for the test only.
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

from loan_advisory.agent import settings as S

ROOT = Path(__file__).resolve().parents[1]
VOICE = ROOT / "apps" / "voice"
DUMMY_KEY = "test-dummy-gnani-key"


# ----------------------------------------------------------------------------------------------------- settings only
def test_defaults_are_prisma_v3_and_timbre_v3_with_tarika():
    g = S.GnaniSpeechSettings()
    assert g.stt_model == "prisma-v3.0" and g.stt_format == "codemix"
    assert g.tts_model == "timbre-v3.0" and g.tts_voice == "Tarika" and g.tts_speed == 1.0
    assert g.tts_voice != "Avinash" and "timbre-v2.0" not in S.GNANI_TTS_MODELS


def test_voice_settings_carry_gnani_and_no_sidecar_urls():
    v = S.VoiceSettings.from_env({"GNANI_TTS_VOICE": "Shivani", "GNANI_TTS_SPEED": "0.9", "GNANI_STT_LANGUAGE": "indic-auto"})
    assert v.speech.tts_voice == "Shivani" and v.speech.tts_speed == 0.9 and v.speech.stt_language == "indic-auto"
    for gone in ("stt_http", "tts_http", "tts_gender"):
        assert not hasattr(v, gone)
    assert S.VoiceSettings().speech == S.GnaniSpeechSettings()


def test_env_cannot_select_retired_timbre_v2_0():
    with pytest.raises(ValueError, match="retired|one of"):
        S.GnaniSpeechSettings.from_env({"GNANI_TTS_MODEL": "timbre-v2.0"})


@pytest.mark.parametrize("voice", ["", "Pranav", "Nalini"])  # empty, the plugin default, a v2.5-only voice
def test_v3_voice_must_be_an_explicit_catalog_voice(voice):
    with pytest.raises(ValueError):
        S.GnaniSpeechSettings(tts_voice=voice)


def test_v2_5_accepts_its_own_voices():
    assert S.GnaniSpeechSettings(tts_model="timbre-v2.5", tts_voice="Lehara").tts_voice == "Lehara"


@pytest.mark.parametrize("speed", [0.85, 1.0, 1.15])
def test_speed_inside_documented_range(speed):
    assert S.GnaniSpeechSettings(tts_speed=speed).tts_speed == speed


@pytest.mark.parametrize("speed", [0.84, 1.16, 0, 2, True, "fast", None])
def test_speed_outside_range_is_refused_not_clamped(speed):
    with pytest.raises(ValueError):
        S.check_tts_speed(speed)


def test_locale_maps_to_gnani_languages():
    g = S.GnaniSpeechSettings()
    assert [g.stt_language_for(x) for x in ("en", "hi", "te", "xx")] == ["en-IN", "hi-IN", "te-IN", "en-IN"]
    assert [g.tts_language_for(x) for x in ("en", "hi", "te")] == ["en-IN", "hi-IN", "te-IN"]
    fixed = S.GnaniSpeechSettings(stt_language="te-IN", tts_language="auto")
    assert fixed.stt_language_for("hi") == "te-IN" and fixed.tts_language_for("hi") == "auto"


@pytest.mark.parametrize("lang,ok", [("en-IN", "en-IN"), ("indic-auto", "indic-auto"), ("hi-IN,en-IN", "hi-IN"),
                                     ("te-IN,hi-IN,en-IN,ta-IN", "te-IN"), ("te", False), ("fr-FR", False),
                                     ("te-IN,fr-FR", False)])
def test_stt_language_validation(lang, ok):
    # one locale only: an old "<language>,en-IN" list keeps its first locale (Prisma streaming refuses lists)
    if ok:
        assert S.GnaniSpeechSettings(stt_language=lang).stt_language == ok
    else:
        with pytest.raises(ValueError):
            S.GnaniSpeechSettings(stt_language=lang)


def test_codemix_needs_prisma_v3():
    with pytest.raises(ValueError):
        S.GnaniSpeechSettings(stt_model="prisma-v2.5", stt_format="codemix")
    assert S.GnaniSpeechSettings(stt_model="prisma-v2.5", stt_format="verbatim", stt_language="te-IN")


# ------------------------------------------------------------------------------------------------- source-level
def test_worker_speech_path_is_gnani_only():
    src = (VOICE / "agent_worker.py").read_text(encoding="utf-8")
    assert "build_stt(CFG.speech, locale)" in src and "build_tts(CFG.speech, locale)" in src
    for old in ("svara", "Svara", "SidecarSTT", "hyperion", "Hyperion", "STT_HTTP", "TTS_HTTP", ":8791", ":8793"):
        assert old not in src, old
    assert not (VOICE / "tts_sidecar.py").exists()
    for path in ("infra/voice/up.sh", "infra/voice/side_stack.sh"):
        text = (ROOT / path).read_text(encoding="utf-8")
        assert "svara" not in text and "hx-loan-stt" not in text and "8793" not in text, path


def test_no_gnani_key_in_tracked_voice_code():
    for path in [VOICE / "gnani_speech.py", VOICE / "agent_worker.py", ROOT / ".env.example"]:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("GNANI_API_KEY="):
                assert line.strip() == "GNANI_API_KEY=", "a value must never be committed"


# ------------------------------------------------------------------------------------------- plugin wiring (mocked)
@pytest.fixture
def gs(monkeypatch):
    pytest.importorskip("livekit.plugins.gnani")
    monkeypatch.setenv("GNANI_API_KEY", DUMMY_KEY)
    monkeypatch.syspath_prepend(str(VOICE))
    sys.modules.pop("gnani_speech", None)
    import gnani_speech
    return gnani_speech


def test_plugin_is_the_pinned_version(gs):
    assert gs.PLUGIN_VERSION == "1.8.11"


def test_build_tts_sends_model_voice_language_and_speed(gs):
    from livekit.plugins.gnani import tts as gt
    t = gs.build_tts(S.GnaniSpeechSettings(tts_speed=0.9), "hi")
    assert t.provider == "Gnani" and t.model == "timbre-v3.0" and t.voice == "Tarika"
    body = gt._build_payload(t._opts, "नमस्ते")
    assert body["model"] == "timbre-v3.0" and body["voice"] == "Tarika"
    assert body["language"] == "hi-IN" and body["speed"] == 0.9
    assert body["audio_config"]["container"] == "raw" and body["audio_config"]["encoding"] == "linear_pcm"
    assert DUMMY_KEY not in json.dumps(body) and DUMMY_KEY not in repr(t.model)


def test_payload_hook_leaves_plain_plugin_options_alone(gs):
    from livekit.plugins.gnani import tts as gt
    plain = gt.GnaniTTSOptions(api_key="x", model="timbre-v3.0", voice="Shivani", language="te-IN")
    body = gt._build_payload(plain, "hi")
    assert "speed" not in body and "language" not in body  # the plugin's own behaviour for non-Timbre options


def test_tts_speed_is_settable_within_bounds_only(gs):
    t = gs.build_tts(S.GnaniSpeechSettings(), "en")
    t.set_speed(1.15)
    assert t.speed == 1.15
    t.update_options(speed=0.85)
    assert t.speed == 0.85
    with pytest.raises(ValueError):
        t.set_speed(1.3)
    assert t.speed == 0.85


def test_tts_refuses_retired_model_even_when_built_directly(gs):
    with pytest.raises(ValueError):
        gs.TimbreTTS(model="timbre-v2.0", voice="Tarika", language="en-IN")
    t = gs.build_tts(S.GnaniSpeechSettings(), "en")
    with pytest.raises(ValueError):
        t.update_options(model="timbre-v2.0")
    with pytest.raises(ValueError):
        gs.TimbreTTS(model="timbre-v3.0", voice="", language="en-IN")


def test_build_stt_streams_prisma_v3_headers(gs):
    stt = gs.build_stt(S.GnaniSpeechSettings(), "te")
    assert stt.provider == "Gnani" and stt.model == "prisma-v3.0" and stt.capabilities.streaming
    assert stt.language_code == "te-IN" and stt.output_format == "codemix"

    async def go():
        stream = stt.stream()
        try:
            return stream.connect_headers()
        finally:
            await stream.aclose()
    h = asyncio.run(go())
    assert h["model"] == "prisma-v3.0" and h["language_code"] == "te-IN" and h["format"] == "codemix"
    assert h["x-sample-rate"] == "16000" and h["x-api-key-id"] == DUMMY_KEY


def test_stt_stream_round_trip_with_fake_socket(gs, monkeypatch):
    """Frames go out as 1024-byte PCM chunks; a `transcript` message comes back as a FINAL_TRANSCRIPT event."""
    import websockets
    from livekit import rtc
    from livekit.agents import stt as lk_stt

    sent, seen_headers = [], {}

    class FakeWS:
        def __init__(self):
            self.inbox = asyncio.Queue()
            self.inbox.put_nowait(json.dumps({"type": "connected"}))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def recv(self):
            return await self.inbox.get()

        async def send(self, data):
            sent.append(data)
            if sum(len(d) for d in sent) >= 2048 and self.inbox.empty():
                self.inbox.put_nowait(json.dumps({"type": "transcript", "text": "నా లోన్ EMI", "segment_id": "s-1"}))
                self.inbox.put_nowait(None)

        def __aiter__(self):
            return self

        async def __anext__(self):
            msg = await self.inbox.get()
            if msg is None:
                raise StopAsyncIteration
            return msg

    def fake_connect(url, additional_headers=None, **kw):
        seen_headers.update(url=url, **additional_headers)
        return FakeWS()

    monkeypatch.setattr(websockets, "connect", fake_connect)
    stt = gs.build_stt(S.GnaniSpeechSettings(), "te")

    async def go():
        stream = stt.stream()
        silence = rtc.AudioFrame(b"\x00\x00" * 1600, 16000, 1, 1600)  # 100 ms
        for _ in range(3):
            stream.push_frame(silence)
        stream.end_input()
        events = []
        async for ev in stream:
            events.append(ev)
        return events
    events = asyncio.run(asyncio.wait_for(go(), timeout=10))
    finals = [e for e in events if e.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert finals and finals[0].alternatives[0].text == "నా లోన్ EMI"
    assert seen_headers["url"] == "wss://api.vachana.ai/stt/v3/stream" and seen_headers["model"] == "prisma-v3.0"
    assert all(len(c) == 1024 for c in sent[:-1])


def test_tts_sse_round_trip_with_fake_http(gs):
    """The SSE body carries model/voice/language/speed; base64 PCM chunks come back as audio frames."""
    import base64

    pcm = b"\x01\x00" * 2205  # 100 ms at 22050 Hz
    lines = [b"event: start\n", b'data: {"status": "streaming_started"}\n', b"\n",
             b"event: chunk\n", f'data: {{"chunk_index": 1, "audio": "{base64.b64encode(pcm).decode()}", "is_final": false}}\n'.encode(),
             b"\n", b"event: complete\n", b'data: {"chunk_index": 2, "audio": "", "is_final": true}\n']
    posted = {}

    class Resp:
        status = 200

        def __init__(self):
            self.content = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            for line in lines:
                yield line

    class Session:
        def post(self, url, json=None, headers=None, **kw):
            posted.update(url=url, body=json, headers=headers)
            return Resp()

    t = gs.build_tts(S.GnaniSpeechSettings(tts_speed=1.1), "te")
    t._session = Session()

    async def go():
        n = 0
        async with t.synthesize("నమస్కారం") as stream:
            async for ev in stream:
                n += ev.frame.samples_per_channel
        return n
    samples = asyncio.run(asyncio.wait_for(go(), timeout=10))
    assert samples >= 2205
    assert posted["url"] == "https://api.vachana.ai/api/v1/tts/sse"
    assert posted["body"]["model"] == "timbre-v3.0" and posted["body"]["voice"] == "Tarika"
    assert posted["body"]["language"] == "te-IN" and posted["body"]["speed"] == 1.1
    assert posted["headers"]["X-API-Key-ID"] == DUMMY_KEY


# --------------------------------------------------------------- 2026-11-10 call audit: one locale, one TLS context
ALL_LOCALES = ("en-IN", "hi-IN", "te-IN", "ta-IN", "kn-IN", "ml-IN", "mr-IN", "bn-IN", "gu-IN", "pa-IN")


@pytest.mark.parametrize("locale", ALL_LOCALES)
def test_phone_call_language_code_is_exactly_one_locale(gs, locale):
    """Prisma streaming refused "te-IN,en-IN" ("exactly one locale is required"): every call language sends one."""
    assert gs.call_stt_language(locale) == locale
    stt = gs.build_stt_for_language(S.GnaniSpeechSettings(), locale)

    async def go():
        stream = stt.stream()
        try:
            return stream.connect_headers()
        finally:
            await stream.aclose()
    h = asyncio.run(go())
    assert h["language_code"] == locale and "," not in h["language_code"]
    assert "multi_lang_codes" not in h  # not accepted on streaming


@pytest.mark.parametrize("locale", ["en", "hi", "te", "xx"])
def test_voice_mode_language_code_is_exactly_one_locale(gs, locale):
    stt = gs.build_stt(S.GnaniSpeechSettings(), locale)
    assert "," not in stt.language_code and stt.language_code in ALL_LOCALES
    legacy = gs.build_stt(S.GnaniSpeechSettings.from_env({"GNANI_STT_LANGUAGE": "te-IN,en-IN"}), locale)
    assert legacy.language_code == "te-IN"


def test_a_stray_comma_list_never_reaches_the_header(gs):
    stt = gs.PrismaSTT(model="prisma-v3.0", language="hi-IN,en-IN", fmt="codemix")

    async def go():
        stream = stt.stream(language="te-IN,en-IN")
        try:
            return stt.language_code, stream.connect_headers()["language_code"]
        finally:
            await stream.aclose()
    assert asyncio.run(go()) == ("hi-IN", "te-IN")
    assert gs.single_locale("indic-auto") == "indic-auto" and gs.single_locale("") == "en-IN"


def test_ssl_context_is_built_once_and_passed_to_every_stt_socket(gs, monkeypatch):
    import ssl

    import websockets
    from livekit import rtc

    gs.ssl_context.cache_clear()
    built = []
    real = ssl.create_default_context

    def counting(*a, **kw):
        built.append(kw.get("cafile"))
        return real(*a, **kw)
    monkeypatch.setattr(ssl, "create_default_context", counting)
    seen = []

    class FakeWS:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def recv(self):
            return json.dumps({"type": "connected"})

        async def send(self, data):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    def fake_connect(url, additional_headers=None, **kw):
        seen.append(kw.get("ssl"))
        return FakeWS()
    monkeypatch.setattr(websockets, "connect", fake_connect)
    stt = gs.build_stt(S.GnaniSpeechSettings(), "te")

    async def one_session():
        stream = stt.stream()
        stream.push_frame(rtc.AudioFrame(b"\x00\x00" * 1600, 16000, 1, 1600))
        stream.end_input()
        async for _ in stream:
            pass
    for _ in range(3):
        asyncio.run(asyncio.wait_for(one_session(), timeout=10))
    assert len(built) == 1 and len(seen) == 3
    assert all(isinstance(c, ssl.SSLContext) for c in seen) and seen[0] is seen[1] is seen[2] is gs.ssl_context()
    try:
        import certifi
        assert built[0] == certifi.where()  # certifi's bundle, not the per-call Windows store walk
    except ImportError:
        pass
