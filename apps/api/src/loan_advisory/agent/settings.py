"""Typed runtime settings of the advisory agent (env-driven; no hard-coded budgets in the graph).

    AGENT_VOICE_DEADLINE_S   per-turn deadline for voice turns (s). Must stay well under the voice worker's HTTP timeout.
    AGENT_TEXT_DEADLINE_S    per-turn deadline for text turns (s).
    AGENT_CLASSIFY_BUDGET_S  max time the LLM intent classifier may take before the deterministic rules decide.
    AGENT_CLASSIFY_BUDGET_INDIC_S  the same for Hindi / Telugu turns (longer prompts in tokens; agent eval: 15 hi/te
                             turns passed the 6 s budget and fell to the regex rules).
    AGENT_MIN_COMPOSE_S      minimum time left in the turn to start (or retry) an LLM compose; otherwise the
                             deterministic template is used straight away.
    SARAL_INTENT_PROVIDER / _MODEL / _FALLBACK_MODEL / _GROQ_BACKUP
                             the intent classifier's own fast client (see loan_advisory.agent.intent_llm).
    AGENT_FAST_CLASSIFY      off | voice | all — high-precision regex intents skip the LLM classifier.
    AGENT_CITATION_MODE      how policy citations are produced and checked (see loan_advisory.agent.citations).
    AGENT_VERIFY_MIN_LEFT_S  llm_verify only: minimum time left in the turn to run the citation-verifier call.
    AGENT_STREAM_BRIDGE      phrase spoken before a corrected answer when a streamed draft was replaced.

Voice worker speech (Gnani only; GnaniSpeechSettings):
    GNANI_API_KEY            read by the Gnani plugin from the environment; never stored, logged or passed around here.
    GNANI_STT_MODEL          prisma-v3.0 (default) | prisma-v2.5
    GNANI_STT_LANGUAGE       locale (default: en-IN / hi-IN,en-IN / te-IN,en-IN by session locale) | indic-auto |
                             a code such as te-IN | a list of up to four such as hi-IN,en-IN (Prisma v3.0)
    GNANI_STT_FORMAT         codemix (default) | verbatim | transcribe | transliterate
    GNANI_TTS_MODEL          timbre-v3.0 (default) | timbre-v2.5   (timbre-v2.0 is retired and refused)
    GNANI_TTS_VOICE          Tarika (default); a Timbre v3.0 catalog voice when the model is v3.0
    GNANI_TTS_LANGUAGE       locale (default: en-IN / hi-IN / te-IN) | auto | hi-en | any Timbre language code
    GNANI_TTS_SPEED          0.85-1.15, default 1.0
    GNANI_TTS_SAMPLE_RATE    8000 | 16000 | 22050 (default) | 44100
    GNANI_TTS_METHOD         sse (default) | websocket | rest
    GNANI_TTS_VERBALISE_NUMBERS  on (default) | off: in te-IN / hi-IN speech, say rupee amounts, percentages and numbers
                             as Telugu / Hindi words (apps/voice/spoken_numbers.py); the TTS request only, never the transcript
"""
import os
from dataclasses import dataclass, fields

FAST_CLASSIFY = ("off", "voice", "all")
CITATION_MODES = ("labels", "verify", "spans", "spans_verify", "llm_verify", "primary_source", "one_source", "one_source_soft")


@dataclass(frozen=True)
class AgentSettings:
    voice_deadline_s: float = 25.0
    text_deadline_s: float = 75.0
    classify_budget_s: float = 8.0
    classify_budget_indic_s: float = 14.0
    min_compose_s: float = 4.0
    fast_classify: str = "voice"
    citation_mode: str = "labels"
    stream_bridge: str = "To be precise:"
    verify_min_left_s: float = 3.0
    intent_provider: str = ""
    intent_model: str = ""
    intent_fallback_model: str = "antigravity/gemini-3.5-flash-lite"  # = intent_llm.DEFAULT_INTENT_FALLBACK_MODEL
    intent_groq_backup: str = "openai/gpt-oss-20b"  # = intent_llm.DEFAULT_GROQ_BACKUP_MODEL

    def __post_init__(self):
        if self.fast_classify not in FAST_CLASSIFY:
            raise ValueError(f"AGENT_FAST_CLASSIFY must be one of {FAST_CLASSIFY}, got {self.fast_classify!r}")
        if self.citation_mode not in CITATION_MODES:
            raise ValueError(f"AGENT_CITATION_MODE must be one of {CITATION_MODES}, got {self.citation_mode!r}")
        if not (0 < self.voice_deadline_s <= self.text_deadline_s):
            raise ValueError("need 0 < AGENT_VOICE_DEADLINE_S <= AGENT_TEXT_DEADLINE_S")

    def deadline_for(self, channel: str) -> float:
        return self.voice_deadline_s if channel == "voice" else self.text_deadline_s

    def classify_budget_for(self, locale: str) -> float:
        return max(self.classify_budget_s, self.classify_budget_indic_s) if locale in ("hi", "te") else self.classify_budget_s

    def fast_classify_for(self, channel: str) -> bool:
        return self.fast_classify == "all" or (self.fast_classify == "voice" and channel == "voice")

    @classmethod
    def from_env(cls, env=None) -> "AgentSettings":
        env = os.environ if env is None else env
        kw = {}
        for f in fields(cls):
            if f.name.startswith("intent_"):  # SARAL_INTENT_*: an empty value is meaningful (auto / no fallback)
                raw = env.get(f"SARAL_{f.name.upper()}")
                if raw is not None:
                    kw[f.name] = raw.strip()
                continue
            raw = env.get(f"AGENT_{f.name.upper()}")
            if raw is None or raw == "":
                continue
            kw[f.name] = float(raw) if f.type in (float, "float") else raw
        return cls(**kw)


def _f(env, key, default):
    raw = env.get(key)
    return float(raw) if raw not in (None, "") else default


# ---------------------------------------------------------------------------------------- Gnani speech (voice worker)
# Speech in the voice worker is Gnani only: Prisma (STT, streaming WebSocket) and Timbre (TTS). The values below are
# the documented ones (docs.gnani.ai, read 2026-11-06). Timbre v2.0 is retired and is refused outright.
GNANI_STT_MODELS = ("prisma-v3.0", "prisma-v2.5")
GNANI_STT_FORMATS = ("verbatim", "transcribe", "codemix", "transliterate")
GNANI_V3_ONLY_FORMATS = ("codemix", "transliterate")
GNANI_LANGS = ("bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN")
GNANI_TTS_MODELS = ("timbre-v3.0", "timbre-v2.5")
GNANI_TTS_LANGS = GNANI_LANGS + ("auto", "hi-en")
# Timbre v3.0 voice catalog; every voice speaks all ten languages. Avinash, Prakash and Harshad are described by
# Gnani as collections / payment-recovery voices, so none of them is ever the default.
TIMBRE_V3_VOICES = ("Shivani", "Lakshmi", "Kalpana", "Chetana", "Vasudha", "Riyaa", "Tarika",
                    "Avinash", "Shubham", "Prakash", "Devansh", "Yashraj", "Harshad")
TTS_SPEED_MIN, TTS_SPEED_MAX = 0.85, 1.15
TTS_SAMPLE_RATES = (8000, 16000, 22050, 44100)  # what livekit-plugins-gnani 1.8.11 accepts
TTS_METHODS = ("sse", "websocket", "rest")
# Session locale (en/hi/te, from the Frappe assertion) -> Gnani language. STT also hears English, because borrowers
# mix it in ("EMI", "CIBIL"); Prisma v3.0 treats the list as a hint, not a lock.
LOCALE_STT = {"en": "en-IN", "hi": "hi-IN,en-IN", "te": "te-IN,en-IN"}
LOCALE_TTS = {"en": "en-IN", "hi": "hi-IN", "te": "te-IN"}


def _stt_language_ok(value: str, model: str) -> bool:
    if value == "indic-auto":
        return model == "prisma-v3.0"
    parts = [p.strip() for p in value.split(",")]
    if not parts or any(p not in GNANI_LANGS for p in parts) or len(set(parts)) != len(parts):
        return False
    return len(parts) == 1 or (model == "prisma-v3.0" and len(parts) <= 4)


@dataclass(frozen=True)
class GnaniSpeechSettings:
    """Gnani Prisma (STT) + Timbre (TTS) configuration. The API key is never stored here: the plugin reads
    GNANI_API_KEY from the environment itself, so it never reaches a log line or a repr."""
    stt_model: str = "prisma-v3.0"
    stt_language: str = "locale"      # "locale" = derive from the session locale (LOCALE_STT); else a Gnani code/list
    stt_format: str = "codemix"       # Prisma v3.0: English words stay in Latin script
    tts_model: str = "timbre-v3.0"
    tts_voice: str = "Tarika"         # warm banking / customer-care voice; never the plugin default
    tts_language: str = "locale"      # "locale" = LOCALE_TTS; else a Timbre language code
    tts_speed: float = 1.0            # 0.85-1.15; the tone engine will set this per turn later
    tts_sample_rate: int = 22050
    tts_method: str = "sse"           # SSE streams chunks; documented with model/voice/language/speed
    tts_verbalise_numbers: bool = True  # te-IN / hi-IN: "₹4,100" is sent to Timbre as "నాలుగు వేల వంద రూపాయలు"

    def __post_init__(self):
        if self.stt_model not in GNANI_STT_MODELS:
            raise ValueError(f"GNANI_STT_MODEL must be one of {GNANI_STT_MODELS}, got {self.stt_model!r}")
        if self.stt_format not in GNANI_STT_FORMATS:
            raise ValueError(f"GNANI_STT_FORMAT must be one of {GNANI_STT_FORMATS}, got {self.stt_format!r}")
        if self.stt_format in GNANI_V3_ONLY_FORMATS and self.stt_model != "prisma-v3.0":
            raise ValueError(f"GNANI_STT_FORMAT={self.stt_format} needs prisma-v3.0")
        if self.stt_language != "locale" and not _stt_language_ok(self.stt_language, self.stt_model):
            raise ValueError(f"GNANI_STT_LANGUAGE {self.stt_language!r} is not a valid language (or list) for {self.stt_model}")
        if self.tts_model not in GNANI_TTS_MODELS:  # timbre-v2.0 is retired
            raise ValueError(f"GNANI_TTS_MODEL must be one of {GNANI_TTS_MODELS}, got {self.tts_model!r}")
        if not self.tts_voice:
            raise ValueError("GNANI_TTS_VOICE must be set explicitly")
        if self.tts_model == "timbre-v3.0" and self.tts_voice not in TIMBRE_V3_VOICES:
            raise ValueError(f"GNANI_TTS_VOICE {self.tts_voice!r} is not a Timbre v3.0 voice ({', '.join(TIMBRE_V3_VOICES)})")
        if self.tts_language != "locale" and self.tts_language not in GNANI_TTS_LANGS:
            raise ValueError(f"GNANI_TTS_LANGUAGE must be 'locale' or one of {GNANI_TTS_LANGS}, got {self.tts_language!r}")
        check_tts_speed(self.tts_speed)
        if self.tts_sample_rate not in TTS_SAMPLE_RATES:
            raise ValueError(f"GNANI_TTS_SAMPLE_RATE must be one of {TTS_SAMPLE_RATES}, got {self.tts_sample_rate}")
        if self.tts_method not in TTS_METHODS:
            raise ValueError(f"GNANI_TTS_METHOD must be one of {TTS_METHODS}, got {self.tts_method!r}")

    def stt_language_for(self, locale: str) -> str:
        return LOCALE_STT.get(locale, LOCALE_STT["en"]) if self.stt_language == "locale" else self.stt_language

    def tts_language_for(self, locale: str) -> str:
        return LOCALE_TTS.get(locale, LOCALE_TTS["en"]) if self.tts_language == "locale" else self.tts_language

    @classmethod
    def from_env(cls, env=None) -> "GnaniSpeechSettings":
        env = os.environ if env is None else env
        d = cls()

        def g(key, default):
            raw = env.get(key)
            return raw.strip() if raw not in (None, "") else default
        return cls(stt_model=g("GNANI_STT_MODEL", d.stt_model), stt_language=g("GNANI_STT_LANGUAGE", d.stt_language),
                   stt_format=g("GNANI_STT_FORMAT", d.stt_format), tts_model=g("GNANI_TTS_MODEL", d.tts_model),
                   tts_voice=g("GNANI_TTS_VOICE", d.tts_voice), tts_language=g("GNANI_TTS_LANGUAGE", d.tts_language),
                   tts_speed=_f(env, "GNANI_TTS_SPEED", d.tts_speed),
                   tts_sample_rate=int(_f(env, "GNANI_TTS_SAMPLE_RATE", d.tts_sample_rate)),
                   tts_method=g("GNANI_TTS_METHOD", d.tts_method),
                   tts_verbalise_numbers=_flag(g("GNANI_TTS_VERBALISE_NUMBERS", "on"), "GNANI_TTS_VERBALISE_NUMBERS"))


def _flag(raw: str, key: str) -> bool:
    value = raw.strip().lower()
    if value in ("1", "true", "on", "yes"):
        return True
    if value in ("0", "false", "off", "no"):
        return False
    raise ValueError(f"{key} must be on or off, got {raw!r}")


def check_tts_speed(speed: float) -> float:
    """Timbre's documented speed range is 0.85-1.15; anything outside is a bug, not something to clamp quietly."""
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not (TTS_SPEED_MIN <= float(speed) <= TTS_SPEED_MAX):
        raise ValueError(f"Timbre speed must be within {TTS_SPEED_MIN}-{TTS_SPEED_MAX}, got {speed!r}")
    return float(speed)


@dataclass(frozen=True)
class VoiceSettings:
    """LiveKit voice worker settings (env). The worker's HTTP timeout must exceed the graph's voice deadline, so the
    graph's own template fallback always answers first and the worker never waits out its timeout."""
    agent_http: str = "http://127.0.0.1:8095"
    turn_log: str = "/root/hx/loan/voice/turns.jsonl"
    http_timeout_s: float = 40.0
    filler_after_s: float = 4.0          # say "one moment" if no guarded sentence has arrived by then
    stream: bool = True                  # use /api/chat/stream (sentence streaming) instead of /api/chat
    min_endpointing_s: float = 0.5       # LiveKit AgentSession end-of-turn delay after VAD silence (library default 0.5)
    vad_min_silence_s: float = 0.55      # Silero VAD silence that ends speech (library default 0.55)
    frappe_url: str = "http://127.0.0.1:8080"   # Frappe origin the worker asks for assertions
    frappe_api_key: str = ""                    # integration account (Agent Integration role); never logged
    frappe_api_secret: str = ""
    filler: dict = None
    unreachable: dict = None
    speech: GnaniSpeechSettings = None   # Gnani Prisma STT + Timbre TTS

    def __post_init__(self):
        object.__setattr__(self, "filler", self.filler or {
            "en": "One moment, please.", "hi": "एक पल रुकिए।", "te": "ఒక్క క్షణం ఆగండి."})
        object.__setattr__(self, "unreachable", self.unreachable or {
            "en": "Sorry, I couldn't reach the advisory service just now. Please ask again in a moment.",
            "hi": "माफ़ कीजिए, अभी सलाह सेवा से जुड़ नहीं पाया। कृपया एक पल बाद फिर पूछें।",
            "te": "క్షమించండి, ఇప్పుడు సలహా సేవను చేరుకోలేకపోయాను. దయచేసి ఒక్క క్షణం తర్వాత మళ్ళీ అడగండి."})
        object.__setattr__(self, "speech", self.speech or GnaniSpeechSettings())

    @classmethod
    def from_env(cls, env=None) -> "VoiceSettings":
        env = os.environ if env is None else env
        d = cls()
        return cls(agent_http=env.get("AGENT_HTTP", d.agent_http), turn_log=env.get("VOICE_TURN_LOG", d.turn_log),
                   http_timeout_s=_f(env, "VOICE_HTTP_TIMEOUT_S", d.http_timeout_s),
                   filler_after_s=_f(env, "VOICE_FILLER_AFTER_S", d.filler_after_s),
                   stream=env.get("VOICE_STREAM", "1") not in ("0", "false", "no"),
                   min_endpointing_s=_f(env, "VOICE_MIN_ENDPOINTING_S", d.min_endpointing_s),
                   vad_min_silence_s=_f(env, "VOICE_VAD_MIN_SILENCE_S", d.vad_min_silence_s),
                   frappe_url=env.get("FRAPPE_URL", d.frappe_url),
                   frappe_api_key=env.get("FRAPPE_API_KEY", d.frappe_api_key),
                   frappe_api_secret=env.get("FRAPPE_API_SECRET", d.frappe_api_secret),
                   speech=GnaniSpeechSettings.from_env(env))
