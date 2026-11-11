"""Per-turn voice metrics and privacy-safe TTS telemetry; never persist caller text or provider error bodies."""
import logging

DROPPED = {"transcript"}
TTS_PLUGIN_LOGGER = "livekit.plugins.gnani.log"
TTS_RETRY_LOGGER = "livekit.agents"
TTS_LOGGERS = {TTS_PLUGIN_LOGGER, TTS_RETRY_LOGGER}


class TTSPrivacyFilter(logging.Filter):
    """Replace SDK warning/error records with fixed messages before they reach handlers."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING:
            return True
        message = record.getMessage()
        if record.name == TTS_PLUGIN_LOGGER and message.startswith("Gnani TTS SSE error:"):
            args = record.args if isinstance(record.args, tuple) else ()
            status = None
            if args and type(args[0]) is int and 100 <= args[0] <= 599:
                status = args[0]
            record.msg = f"Gnani TTS provider event (HTTP {status})" if status else "Gnani TTS provider event"
        elif record.name == TTS_RETRY_LOGGER and message.startswith("failed to synthesize speech:"):
            record.msg = "LiveKit TTS retry/failure event"
        else:
            return True
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


def install_tts_log_sanitizer() -> None:
    """Install the sanitizer on the pinned SDK loggers; safe to call repeatedly."""
    sanitizer = TTSPrivacyFilter()
    for name in TTS_LOGGERS:
        logger = logging.getLogger(name)
        if not any(isinstance(f, TTSPrivacyFilter) for f in logger.filters):
            logger.addFilter(sanitizer)


def record_tts_error(info: dict, error, recoverable: bool) -> None:
    """Store only fixed error classes, retry counts and numeric HTTP status; never the error message/body."""
    info["tts_error_count"] = max(0, int(info.get("tts_error_count") or 0)) + 1
    info["tts_retry_count"] = max(0, int(info.get("tts_retry_count") or 0)) + int(bool(recoverable))
    name = type(error).__name__ if error is not None else ""
    info["tts_error_kind"] = {
        "APIConnectionError": "connection",
        "APIStatusError": "http_status",
        "APITimeoutError": "timeout",
        "APIError": "provider",
    }.get(name, "provider_error")
    status = getattr(error, "status_code", None)
    if type(status) is int and 100 <= status <= 599:
        info["tts_http_status"] = status
    info["tts_status"] = "retrying" if recoverable else "failed"


def record_tts_error_event(info: dict, event) -> None:
    record_tts_error(info, getattr(event, "error", None), bool(getattr(event, "recoverable", False)))


def mark_tts_success(info: dict) -> None:
    info["tts_status"] = "ok"
    info["tts_recovered"] = bool(info.get("tts_retry_count"))


def turn_row(info: dict, marks: dict, now: float) -> dict:
    base = marks["user_stopped"]
    row = {"ts": now, **{k: v for k, v in info.items() if k not in DROPPED},
           **{f"{k}_ms": round((v - base) * 1000, 1) for k, v in marks.items() if k != "user_stopped"}}
    if "agent_speaking_ms" in row:
        # LiveKit's speaking-state event is a proxy for first audible output, not a handset playback acknowledgement.
        row["first_audible_ms"] = row["agent_speaking_ms"]
        row["first_audible_stage"] = "livekit_agent_speaking"
    return row


__all__ = ["DROPPED", "TTSPrivacyFilter", "install_tts_log_sanitizer", "record_tts_error", "record_tts_error_event",
           "mark_tts_success", "turn_row"]
