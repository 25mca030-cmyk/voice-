"""Gemini Live speech-to-speech engine for VoiceOS calls.

One model listens and speaks directly (no separate STT -> LLM -> TTS hops), which is what made the
built-in VoiceOS relay answer in 0.1-0.3 s on 23 Oct while the three-step pipeline took 2-4 s.

The live model is chosen by a quick handshake with the caller's Gemini key (first call only, then
cached), so an unavailable model name can never leave a call silent. If no live model answers, the
server falls back to the three-step pipeline.
"""
import asyncio
import hashlib
import os
import time

from loguru import logger

LIVE_MODELS = [m.strip() for m in os.environ.get(
    "LIVE_MODELS",
    "gemini-3.1-flash-live-preview,gemini-3.8-live,"
    "gemini-2.5-flash-native-audio-preview-12-2025,gemini-2.5-flash-native-audio-preview-09-2025",
).split(",") if m.strip()]
LIVE_VOICES = {"Puck", "Charon", "Kore", "Fenrir", "Aoede", "Leda", "Orus", "Zephyr"}
DEFAULT_VOICE = os.environ.get("LIVE_VOICE", "Zephyr")
LIVE_LANGS = {
    "English": "en-IN", "English (India)": "en-IN", "Hindi": "hi-IN", "Gujarati": "gu-IN", "Marathi": "mr-IN",
    "Bengali": "bn-IN", "Tamil": "ta-IN", "Telugu": "te-IN", "Kannada": "kn-IN", "Malayalam": "ml-IN",
    "Punjabi": "pa-IN", "Urdu": "ur-IN",
}

_GOOD: dict[str, str] = {}  # key hash -> live model that answered
# Known-good model (answered on 27 Oct): used straight away, skipping the ~2.6 s model check.
# If it ever fails on a call, it is forgotten and the next call runs the full check again.
PINNED = os.environ.get("LIVE_MODEL", "gemini-3.1-flash-live-preview").strip()
_BAD_PINNED: set[str] = set()
SILENCE_MS = int(os.environ.get("LIVE_SILENCE_MS", "250"))


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def fast_thinking(model: str):
    """Lowest thinking each live model accepts: Gemini 3 thinks before speaking unless told not to."""
    from google.genai.types import ThinkingConfig
    if "gemini-3" in model:
        return ThinkingConfig(thinking_level="MINIMAL")
    if "2.5" in model:
        return ThinkingConfig(thinking_budget=0)
    return None


_THINKING_OK: dict[str, bool] = {}  # model -> Google accepted the fast-thinking setting


async def _handshake(key: str, model: str, timeout: float, thinking=None) -> bool:
    from google import genai
    client = genai.Client(api_key=key)
    try:
        config = {"response_modalities": ["AUDIO"]}
        if thinking is not None:
            config["thinking_config"] = thinking

        async def _try():
            async with client.aio.live.connect(model=model, config=config):
                return True
        return await asyncio.wait_for(_try(), timeout)
    except Exception as e:
        logger.warning(f"live model {model} unavailable ({type(e).__name__}: {str(e)[:120]})")
        return False


async def pick_live_model(key: str) -> str | None:
    """First live model the key can open a session with; cached per key."""
    if not key:
        return None
    h = _hash(key)
    if h in _GOOD:
        return _GOOD[h]
    if PINNED and h not in _BAD_PINNED:
        return PINNED
    t = time.time()
    # Check all candidates at once (first call only), then keep the highest-priority one that answered.
    results = await asyncio.gather(*(_handshake(key, m, 4) for m in LIVE_MODELS))
    for model, ok in zip(LIVE_MODELS, results):
        if ok:
            _GOOD[h] = model
            logger.info(f"live model {model} selected in {(time.time() - t) * 1000:.0f} ms")
            return model
    return None


async def thinking_supported(key: str, model: str) -> bool:
    """Checked once per model (about 0.5 s, run while the call is being set up), then cached."""
    if model == PINNED and os.environ.get("LIVE_FAST_THINKING", "on") == "on":
        return True  # Google accepted it for this model on every call so far; skip the ~1 s check
    if model not in _THINKING_OK:
        cfg = fast_thinking(model)
        _THINKING_OK[model] = bool(cfg) and await _handshake(key, model, 3, cfg)
        logger.info(f"fast thinking for {model}: {'on' if _THINKING_OK[model] else 'off'}")
    return _THINKING_OK[model]


def forget_live_model(key: str):
    h = _hash(key)
    _GOOD.pop(h, None)
    _BAD_PINNED.add(h)


def live_instructions(agent: dict) -> str:
    return (agent["instructions"]
            + "\n\nThis is a live phone call: answer immediately in one or two short spoken sentences"
              " (under 25 words) unless asked for detail. Do not think out loud or pause before answering."
            + "\nWhen the caller says bye or wants to end the call, reply only with a short polite goodbye"
              " (for example \"Thank you for your time. Goodbye!\") in their language and ask nothing more;"
              " the call ends automatically after your goodbye.")


# ---------------------------------------------------------------- cached greeting audio
# The greeting is the same text every call. Gemini's spoken greeting from the first call is kept and
# replayed the moment later calls connect, instead of waiting ~1.5 s for Gemini to connect and speak it.
GREETING_DIR = os.environ.get("GREETING_CACHE_DIR", "/tmp/voiceos-greetings")
_GREETINGS: dict[str, tuple[int, bytes]] = {}


def greeting_key(agent: dict, model: str) -> str | None:
    opening = (agent.get("opening") or "").strip()
    if not opening:
        return None
    voice = agent.get("voice") if agent.get("voice") in LIVE_VOICES else DEFAULT_VOICE
    raw = "|".join([model, voice, agent.get("language") or "English", opening])
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def cached_greeting(k: str | None):
    if not k:
        return None
    if k in _GREETINGS:
        return _GREETINGS[k]
    try:
        for name in os.listdir(GREETING_DIR):
            if name.startswith(k + "-"):
                rate = int(name.rsplit("-", 1)[1].split(".")[0])
                with open(os.path.join(GREETING_DIR, name), "rb") as f:
                    _GREETINGS[k] = (rate, f.read())
                return _GREETINGS[k]
    except OSError:
        pass
    return None


def store_greeting(k: str, rate: int, pcm: bytes):
    seconds = len(pcm) / 2 / rate
    if not (0.8 <= seconds <= 20):
        return
    _GREETINGS[k] = (rate, pcm)
    try:
        os.makedirs(GREETING_DIR, exist_ok=True)
        with open(os.path.join(GREETING_DIR, f"{k}-{rate}.pcm"), "wb") as f:
            f.write(pcm)
    except OSError:
        pass
    logger.info(f"greeting audio cached ({seconds:.1f} s); next calls greet instantly")


def remember_agent_greeting(agent_id: str, k: str | None):
    """Links a VoiceOS agent id to its greeting so the next call can greet before the agent loads."""
    if not agent_id or not k:
        return
    try:
        os.makedirs(GREETING_DIR, exist_ok=True)
        with open(os.path.join(GREETING_DIR, f"agent-{agent_id}"), "w") as f:
            f.write(k)
    except OSError:
        pass


def greeting_for_agent(agent_id: str):
    """(sample_rate, pcm, key) for this agent's last greeting, or None."""
    try:
        with open(os.path.join(GREETING_DIR, f"agent-{agent_id}")) as f:
            k = f.read().strip()
    except OSError:
        return None
    g = cached_greeting(k)
    return (g[0], g[1], k) if g else None


def to_phone_rate(rate: int, pcm: bytes, target: int = 16000) -> bytes:
    if rate == target:
        return pcm
    import numpy as np
    import soxr
    samples = np.frombuffer(pcm, dtype=np.int16)
    return soxr.resample(samples, rate, target).astype(np.int16).tobytes()


def already_greeted_agent(agent: dict) -> dict:
    """Agent copy whose instructions say the greeting has already been played."""
    opening = (agent.get("opening") or "").strip()
    a = dict(agent)
    a["instructions"] = (agent["instructions"] + f'\n\nThe call is already connected and you have just greeted the caller with:'
                         f' "{opening}". Do not greet again; wait for the caller to answer, then continue.')
    return a


def opening_message(agent: dict) -> dict:
    opening = (agent.get("opening") or "").strip()
    language = agent.get("language") or "English"
    text = (f'Start the call now in {language}. Greet the caller with exactly this opening, then stop and listen: "{opening}"'
            if opening else
            f"Start the call now in {language}. Give a brief greeting suited to your role, ask one opening question, then listen.")
    return {"role": "user", "content": text}


def build_live_llm(*, key: str, model: str, agent: dict, tools_schema, phone: bool, thinking: bool = False,
                   speak_first: bool = True):
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, GeminiVADParams
    from google.genai.types import EndSensitivity, StartSensitivity

    voice = agent.get("voice") if agent.get("voice") in LIVE_VOICES else DEFAULT_VOICE
    language = LIVE_LANGS.get(agent.get("language") or "English", "en-IN")
    settings = GeminiLiveLLMService.Settings(
        model=f"models/{model}",
        voice=voice,
        language=language,
        system_instruction=live_instructions(agent),
        # Phone lines carry hiss and echo: only clear speech should cut the agent off,
        # but the end of the caller's sentence should be detected quickly.
        vad=GeminiVADParams(
            # Phone audio from Vobiz is quiet: with LOW sensitivity Gemini missed callers' speech (one call got no
            # reply at all, another waited 7 s). HIGH hears them at once; the echo guard keeps the agent's own
            # voice out, so it does not interrupt itself.
            start_sensitivity=StartSensitivity.START_SENSITIVITY_HIGH,
            end_sensitivity=EndSensitivity.END_SENSITIVITY_HIGH,
            # Silence before Gemini treats the caller's turn as finished: this is most of the reply delay.
            silence_duration_ms=SILENCE_MS,
            prefix_padding_ms=100,
        ),
        **({"thinking": fast_thinking(model)} if thinking else {}),
    )
    llm = GeminiLiveLLMService(api_key=key, settings=settings, tools=tools_schema,
                               inference_on_context_initialization=speak_first)
    logger.info(f"live engine: model={model} voice={voice} language={language} fast_thinking={thinking}")
    return llm
