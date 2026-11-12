import asyncio
import logging
from types import SimpleNamespace

import pytest

from loan_advisory.agent.voice_log import (
    TTS_PLUGIN_LOGGER,
    TTSPrivacyFilter,
    record_tts_error,
    tts_error_context,
    wire_tts_telemetry,
)


@pytest.mark.parametrize("message, args", [
    ("Gnani TTS SSE error: %s - %s", (503, "CALLER_BODY_SENTINEL")),
    ("Gnani TTS REST error: %s - %s", (503, "CALLER_BODY_SENTINEL")),
    ("Gnani TTS WS error: %s", ("CALLER_BODY_SENTINEL",)),
])
def test_provider_error_bodies_are_redacted_for_every_transport(message, args):
    record = logging.LogRecord(TTS_PLUGIN_LOGGER, logging.ERROR, __file__, 1, message, args, None)

    assert TTSPrivacyFilter().filter(record)
    assert "CALLER_BODY_SENTINEL" not in record.getMessage()
    assert record.args == ()
    assert record.exc_info is None and record.exc_text is None and record.stack_info is None
    if args[0] == 503:
        assert record.getMessage() == "Gnani TTS provider event (HTTP 503)"
    else:
        assert record.getMessage() == "Gnani TTS provider event"


def test_non_http_tts_error_clears_stale_http_status():
    class APIStatusError(Exception):
        status_code = 503

    class APIConnectionError(Exception):
        pass

    info = {}
    record_tts_error(info, APIStatusError(), recoverable=True)
    assert info["tts_http_status"] == 503

    record_tts_error(info, APIConnectionError(), recoverable=False)
    assert "tts_http_status" not in info
    assert info["tts_status"] == "failed"


class FakeEmitter:
    def on(self, name, callback):
        assert name == "error"
        self.callback = callback

    def emit(self, name, event):
        assert name == "error"
        self.callback(event)


def test_late_tts_error_stays_with_the_synthesis_turn_that_created_its_task():
    class APIStatusError(Exception):
        status_code = 503

    emitter = FakeEmitter()
    wire_tts_telemetry(emitter)
    abandoned_turn, current_turn = {}, {}
    event = SimpleNamespace(error=APIStatusError(), recoverable=True)

    async def emit_late():
        await asyncio.sleep(0)
        emitter.emit("error", event)

    async def run():
        with tts_error_context(abandoned_turn):
            task = asyncio.create_task(emit_late())
        with tts_error_context(current_turn):
            await task

    asyncio.run(run())

    assert abandoned_turn["tts_error_count"] == 1
    assert abandoned_turn["tts_http_status"] == 503
    assert abandoned_turn["tts_retry_count"] == 1
    assert current_turn == {}

    emitter.emit("error", event)  # e.g. synthesis outside a tracked, verified turn
    assert current_turn == {}
