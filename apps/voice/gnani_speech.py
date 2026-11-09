"""Gnani speech for the voice worker: Prisma (STT, streaming WebSocket) and Timbre (TTS), built on the official
LiveKit plugin `livekit-plugins-gnani` (pinned 1.8.11, installed as `livekit-agents[gnani]`).

The plugin covers transport, retries and audio framing. Two gaps in 1.8.11 are closed here, and only here:

* STT: the plugin never sends the `model` header, so the server falls back to Prisma v2.5. `PrismaSTT` streams with
  `model=prisma-v3.0` plus `language_code` (one code, a list of up to four, or `indic-auto`) and `format` (e.g.
  `codemix`), using the documented header names (docs.gnani.ai/api/STT/stt-websocket).
* TTS: the plugin defaults to the retired `timbre-v2.0` with the voice `Pranav`, sends `language` for v2.5 only and
  never sends `speed`. `TimbreTTS` always takes an explicit model and voice, sends `language` for v2.5 and v3.0, and
  sends `speed` (0.85-1.15) so the tone engine can change it per turn with `set_speed()`.

Authentication: the plugin reads GNANI_API_KEY from the environment. Nothing here reads, stores or prints the key.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, fields, replace
from typing import Any

from livekit.agents import (DEFAULT_API_CONNECT_OPTIONS, APIConnectionError, APIConnectOptions, APIStatusError,
                            APITimeoutError, utils)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils.misc import is_given
from livekit.plugins import gnani
from livekit.plugins.gnani import stt as gnani_stt
from livekit.plugins.gnani import tts as gnani_tts

from loan_advisory.agent.settings import GnaniSpeechSettings, check_tts_speed

PLUGIN_VERSION = gnani.__version__


# --------------------------------------------------------------------------------------------------------------- STT
class PrismaSpeechStream(gnani_stt.SpeechStream):
    """The plugin's WebSocket stream, connecting with the Prisma v3.0 headers. Sending audio and reading transcripts
    are the plugin's own (`_send_audio`, `_recv_messages`)."""

    def __init__(self, *, stt: "PrismaSTT", opts: gnani_stt.GnaniSTTOptions, conn_options: APIConnectOptions,
                 model: str, language_code: str, fmt: str) -> None:
        super().__init__(stt=stt, opts=opts, conn_options=conn_options)
        self._model, self._language_code, self._fmt = model, language_code, fmt

    def connect_headers(self) -> dict[str, str]:
        return {"x-api-key-id": self._opts.api_key, "model": self._model, "language_code": self._language_code,
                "format": self._fmt, "x-sample-rate": str(self._opts.sample_rate)}

    async def _run(self) -> None:
        import websockets

        try:
            async with websockets.connect(self._build_ws_url(), additional_headers=self.connect_headers(),
                                          ping_interval=20, ping_timeout=20, close_timeout=10) as ws:
                hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if hello.get("type") != "connected":
                    raise APIStatusError(message=f"Gnani STT refused the session: {hello.get('message', hello.get('type'))}",
                                         status_code=500, body=hello)
                send_task = asyncio.create_task(self._send_audio(ws), name="prisma-stt-send")
                recv_task = asyncio.create_task(self._recv_messages(ws), name="prisma-stt-recv")
                try:
                    done, _ = await asyncio.wait([send_task, recv_task], return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                    if send_task.done() and not recv_task.done():
                        with contextlib.suppress(asyncio.TimeoutError):
                            await asyncio.wait_for(asyncio.shield(recv_task), timeout=1.0)
                finally:
                    await utils.aio.gracefully_cancel(send_task, recv_task)
        except websockets.exceptions.ConnectionClosed as e:
            raise APIConnectionError(f"Gnani STT WebSocket closed unexpectedly: {e.code}") from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError("Gnani STT WebSocket connection timed out") from e
        except (APIConnectionError, APIStatusError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError(f"Gnani STT WebSocket error: {type(e).__name__}") from e


class PrismaSTT(gnani.STT):
    """Gnani Prisma streaming STT. `language` is one Gnani code, a comma list of up to four (a hint, not a lock), or
    `indic-auto`; transcripts are labelled with the first listed language."""

    def __init__(self, *, model: str, language: str, fmt: str, sample_rate: int = 16000) -> None:
        first = "en-IN" if language == "indic-auto" else language.split(",")[0].strip()
        super().__init__(language=first, sample_rate=sample_rate, use_streaming=True)
        self._prisma_model, self._language_code, self._fmt = model, language, fmt

    @property
    def model(self) -> str:
        return self._prisma_model

    @property
    def language_code(self) -> str:
        return self._language_code

    @property
    def output_format(self) -> str:
        return self._fmt

    def stream(self, *, language: NotGivenOr[str] = NOT_GIVEN,
               conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> PrismaSpeechStream:
        code = language if is_given(language) else self._language_code
        return PrismaSpeechStream(stt=self, opts=replace(self._opts), conn_options=self._single_attempt(conn_options),
                                  model=self._prisma_model, language_code=code, fmt=self._fmt)


# --------------------------------------------------------------------------------------------------------------- TTS
@dataclass
class TimbreOptions(gnani_tts.GnaniTTSOptions):
    speed: float = 1.0


_PLUGIN_BUILD_PAYLOAD = gnani_tts._build_payload


def timbre_payload(opts: gnani_tts.GnaniTTSOptions, text: str) -> dict:
    """The plugin's request body, plus `language` for Timbre v3.0 and `speed`, for TimbreTTS options only."""
    payload = _PLUGIN_BUILD_PAYLOAD(opts, text)
    if isinstance(opts, TimbreOptions):
        if opts.language is not None:
            payload["language"] = opts.language
        payload["speed"] = opts.speed
    return payload


def _install_payload_hook() -> None:
    # The plugin's three synthesis streams (REST, SSE, WebSocket) all build their body with this module-level
    # function; wrapping it is the smallest change that reaches all three. Options that are not TimbreOptions are
    # passed through untouched, and the hook fails loudly if a plugin upgrade drops the function.
    if not callable(getattr(gnani_tts, "_build_payload", None)):
        raise RuntimeError(f"livekit-plugins-gnani {PLUGIN_VERSION} no longer has tts._build_payload; re-check gnani_speech.py")
    gnani_tts._build_payload = timbre_payload


_install_payload_hook()


class TimbreTTS(gnani.TTS):
    """Gnani Timbre TTS with an explicit model and voice (never the plugin's v2.0 / Pranav defaults) and a speed."""

    def __init__(self, *, model: str, voice: str, language: str, speed: float = 1.0, sample_rate: int = 22050,
                 method: str = "sse") -> None:
        if model not in ("timbre-v3.0", "timbre-v2.5"):
            raise ValueError(f"Timbre model {model!r} is not allowed (timbre-v2.0 is retired)")
        if not voice:
            raise ValueError("a Timbre voice must be given explicitly")
        # container=raw + linear_pcm: every streamed chunk is plain 16-bit PCM, so chunks concatenate without headers.
        super().__init__(model=model, voice=voice, language=language, sample_rate=sample_rate, container="raw",
                         encoding="linear_pcm", synthesize_method=method)
        base = {f.name: getattr(self._opts, f.name) for f in fields(gnani_tts.GnaniTTSOptions)}
        self._opts = TimbreOptions(**base, speed=check_tts_speed(speed))

    @property
    def voice(self) -> str:
        return self._opts.voice

    @property
    def language(self) -> str | None:
        return self._opts.language

    @property
    def speed(self) -> float:
        return self._opts.speed

    def set_speed(self, speed: float) -> None:
        """Speed for the next synthesis (0.85-1.15). For the tone engine; a stream already running keeps its speed."""
        self._opts.speed = check_tts_speed(speed)

    def update_options(self, *, speed: float | None = None, **kwargs: Any) -> None:
        if kwargs.get("model") not in (None, "timbre-v3.0", "timbre-v2.5"):
            raise ValueError(f"Timbre model {kwargs['model']!r} is not allowed (timbre-v2.0 is retired)")
        super().update_options(**kwargs)
        if speed is not None:
            self.set_speed(speed)


# ----------------------------------------------------------------------------------------------------------- factory
def build_stt(cfg: GnaniSpeechSettings, locale: str) -> PrismaSTT:
    return PrismaSTT(model=cfg.stt_model, language=cfg.stt_language_for(locale), fmt=cfg.stt_format)


def build_tts(cfg: GnaniSpeechSettings, locale: str) -> TimbreTTS:
    return TimbreTTS(model=cfg.tts_model, voice=cfg.tts_voice, language=cfg.tts_language_for(locale),
                     speed=cfg.tts_speed, sample_rate=cfg.tts_sample_rate, method=cfg.tts_method)


__all__ = ["PLUGIN_VERSION", "PrismaSTT", "PrismaSpeechStream", "TimbreOptions", "TimbreTTS", "build_stt", "build_tts",
           "timbre_payload"]
