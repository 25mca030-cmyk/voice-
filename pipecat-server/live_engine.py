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


async def _handshake(key: str, model: str, timeout: float) -> bool:
    from google import genai
    client = genai.Client(api_key=key)
    try:
        async def _try():
            async with client.aio.live.connect(model=model, config={"response_modalities": ["AUDIO"]}):
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


def forget_live_model(key: str):
    h = _hash(key)
    _GOOD.pop(h, None)
    _BAD_PINNED.add(h)


def live_instructions(agent: dict) -> str:
    return (agent["instructions"]
            + "\n\nThis is a live phone call: reply in 1-2 short spoken sentences unless asked for detail."
            + "\nWhen the caller says bye or wants to end the call, reply only with a short polite goodbye"
              " (for example \"Thank you for your time. Goodbye!\") in their language and ask nothing more;"
              " the call ends automatically after your goodbye.")


def opening_message(agent: dict) -> dict:
    opening = (agent.get("opening") or "").strip()
    language = agent.get("language") or "English"
    text = (f'Start the call now in {language}. Greet the caller with exactly this opening, then stop and listen: "{opening}"'
            if opening else
            f"Start the call now in {language}. Give a brief greeting suited to your role, ask one opening question, then listen.")
    return {"role": "user", "content": text}


def build_live_llm(*, key: str, model: str, agent: dict, tools_schema, phone: bool):
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
            start_sensitivity=StartSensitivity.START_SENSITIVITY_LOW if phone else StartSensitivity.START_SENSITIVITY_HIGH,
            end_sensitivity=EndSensitivity.END_SENSITIVITY_HIGH,
            # Silence before Gemini treats the caller's turn as finished: this is most of the reply delay.
            silence_duration_ms=SILENCE_MS,
            prefix_padding_ms=100,
        ),
    )
    llm = GeminiLiveLLMService(api_key=key, settings=settings, tools=tools_schema)
    logger.info(f"live engine: model={model} voice={voice} language={language}")
    return llm
