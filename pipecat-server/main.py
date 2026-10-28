"""VoiceOS Pipecat server.

Runs real-time voice conversations (STT -> LLM -> ElevenLabs TTS) with interruptions.
VoiceOS stays the source of truth: this server claims a call with its one-time token, loads the
agent from VoiceOS, runs tools through VoiceOS and saves the finished call back to VoiceOS.

Endpoints
  GET  /health
  WS   /ws?call=<id>&token=<token>      browser test calls (VoiceOS JSON protocol, 16k in / 24k out)
  WS   /phone?call=<id>&token=<token>   Vobiz / Plivo-compatible media streams (L16)
"""

import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import time

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMTextFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.frames.frames import LLMRunFrame
from audio_cleanup import make_noise_filter
from call_recording import CallRecorder, save_recording
from live_engine import (already_greeted_agent, build_live_llm, cached_greeting, forget_live_model, greeting_for_agent,
                         greeting_key, opening_message, pick_live_model, remember_agent_greeting, store_greeting,
                         thinking_supported, to_phone_rate)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
try:
    from pipecat.serializers.base_serializer import FrameSerializerType
except ImportError:  # newer Pipecat infers text/binary
    FrameSerializerType = None
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

try:  # renamed across Pipecat versions
    from pipecat.frames.frames import InterruptionFrame as _Interrupt
except ImportError:  # pragma: no cover
    from pipecat.frames.frames import StartInterruptionFrame as _Interrupt
try:
    from pipecat.metrics.metrics import LLMUsageMetricsData
except ImportError:  # pragma: no cover
    LLMUsageMetricsData = None

VOICEOS_URL = os.environ.get("VOICEOS_URL", "https://chat-loom-builder.lovable.app").rstrip("/")
SECRET = os.environ.get("PIPECAT_SHARED_SECRET", "")
DEFAULT_VOICE = os.environ.get("ELEVENLABS_DEFAULT_VOICE", "EXAVITQu4vr4xnSDxMaL")  # Sarah
LANG_CODES = {
    "English": "en", "Hindi": "hi", "Gujarati": "gu", "Marathi": "mr", "Bengali": "bn", "Tamil": "ta",
    "Telugu": "te", "Kannada": "kn", "Malayalam": "ml", "Punjabi": "pa", "Urdu": "ur", "Odia": "or",
    "Spanish": "es", "French": "fr", "German": "de", "Portuguese": "pt", "Italian": "it", "Japanese": "ja",
    "Arabic": "ar", "Chinese": "zh",
}

# DEBUG logging wrote a line for many audio events; on a 0.15-CPU server that competes with the audio.
import sys  # noqa: E402
logger.remove()
logger.add(sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO"))

PHONE_GAIN = float(os.environ.get("PHONE_GAIN", "1.5"))
PHONE_IN_BYTES = int(os.environ.get("PHONE_IN_MS", "40")) * 32  # 16 kHz, 16-bit

app = FastAPI(title="VoiceOS Pipecat")

# Load the heavy provider SDKs once at boot. Importing them on the first call took ~30 s on Render,
# so the caller heard silence and hung up before the agent was ready.
try:
    from pipecat.services.google.llm import GoogleLLMService as _GoogleLLM  # noqa: F401
    from pipecat.services.sarvam.stt import SarvamSTTService as _SarvamSTT  # noqa: F401
    import sarvam_flash as _sarvam_flash  # noqa: F401
    import sarvam_live as _sarvam_live  # noqa: F401
    SileroVADAnalyzer()  # downloads/loads the VAD model now, not during a call
except Exception as _e:  # pragma: no cover
    logger.warning(f"preload skipped: {_e}")


# ---------------------------------------------------------------- VoiceOS bridge (signed)
# Agent setup VoiceOS returned for each agent's last call (memory only; refreshed on every call).
_SESSION_CACHE: dict[str, dict] = {}

# One keep-alive client for every VoiceOS callback: a new client per request paid a fresh TLS
# handshake (~150-300 ms) on session start and on every tool call.
_BRIDGE: httpx.AsyncClient | None = None


def _bridge() -> httpx.AsyncClient:
    global _BRIDGE
    if _BRIDGE is None or _BRIDGE.is_closed:
        _BRIDGE = httpx.AsyncClient(timeout=20, limits=httpx.Limits(max_keepalive_connections=20, keepalive_expiry=120))
    return _BRIDGE


async def voiceos(path: str, payload: dict) -> dict:
    body = json.dumps(payload, separators=(",", ":"))
    ts = str(int(time.time()))
    sig = hmac.new(SECRET.encode(), f"{ts}.{body}".encode(), hashlib.sha256).hexdigest()
    r = await _bridge().post(
        f"{VOICEOS_URL}/api/public/pipecat/{path}",
        content=body,
        headers={"content-type": "application/json", "x-pipecat-timestamp": ts, "x-pipecat-signature": sig},
    )
    data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code >= 400:
        raise RuntimeError(data.get("error") or f"VoiceOS {path} failed ({r.status_code})")
    return data


def key(keys: dict, name: str, env: str) -> str:
    return (keys.get(name) or os.environ.get(env, "")).strip()


# ---------------------------------------------------------------- services
SARVAM_LANGS = {"Hindi": "hi-IN", "English": "en-IN", "Bengali": "bn-IN", "Gujarati": "gu-IN", "Kannada": "kn-IN",
                "Malayalam": "ml-IN", "Marathi": "mr-IN", "Odia": "od-IN", "Punjabi": "pa-IN", "Tamil": "ta-IN", "Telugu": "te-IN"}


def make_stt(provider: str, language: str, keys: dict):
    code = LANG_CODES.get(language, "en")
    if provider == "sarvam":
        from sarvam_live import SafeSarvamSTT
        return SafeSarvamSTT(api_key=key(keys, "sarvam", "SARVAM_API_KEY"))
    if provider == "openai":
        from pipecat.services.openai.stt import OpenAISTTService
        return OpenAISTTService(api_key=key(keys, "openai", "OPENAI_API_KEY"), model="gpt-4o-transcribe", language=code)
    # deepgram-sdk v5 (pulled in by Pipecat 0.0.108) no longer exports LiveOptions; Pipecat ships the wrapper.
    try:
        from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
    except ImportError:  # older Pipecat
        from deepgram import LiveOptions
        from pipecat.services.deepgram.stt import DeepgramSTTService
    return DeepgramSTTService(
        api_key=key(keys, "deepgram", "DEEPGRAM_API_KEY"),
        # Short endpointing finalises the transcript ~200 ms after speech ends instead of Deepgram's default.
        live_options=LiveOptions(language="multi" if code not in ("en", "hi", "es", "fr", "de") else code, model="nova-3",
                                 interim_results=True, punctuate=True, smart_format=False, endpointing=200),
    )


def make_llm(spec: str, keys: dict):
    provider, _, model = spec.partition("/")
    if provider == "openai":
        from pipecat.services.openai.llm import OpenAILLMService
        return OpenAILLMService(api_key=key(keys, "openai", "OPENAI_API_KEY"), model=model or "gpt-4.1-mini",
                                params=OpenAILLMService.InputParams(max_tokens=MAX_REPLY_TOKENS, temperature=0.6))
    if provider == "anthropic":
        from pipecat.services.anthropic.llm import AnthropicLLMService
        return AnthropicLLMService(api_key=key(keys, "anthropic", "ANTHROPIC_API_KEY"), model=model or "claude-haiku-4-5",
                                   params=AnthropicLLMService.InputParams(max_tokens=MAX_REPLY_TOKENS, temperature=0.6))
    return FallbackGoogleLLM(api_key=key(keys, "gemini", "GEMINI_API_KEY"), model=current_gemini(model),
                             params=GoogleLLMService.InputParams(max_tokens=MAX_REPLY_TOKENS, temperature=0.6))


# Spoken replies are short; a small cap keeps the model from drafting long answers.
MAX_REPLY_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "300"))


def thinking_for(model: str):
    """Gemini 3 thinks by default (Pipecat only disables it for 2.5), which added seconds before the
    first word. Voice replies need the lowest level each model accepts."""
    m = model or ""
    if m.startswith("gemini-2.5") and "pro" not in m:
        return GoogleLLMService.ThinkingConfig(thinking_budget=0)
    if m.startswith("gemini-3"):
        return GoogleLLMService.ThinkingConfig(thinking_level="low" if "pro" in m else "minimal")
    return None  # "-latest" aliases: leave Pipecat's default


# Models the saved Google key answered for (see src/lib/model-catalog.ts). Tried in order when the chosen
# model is overloaded (503), rate-limited (429), missing (404) or slow, so the caller always gets a reply.
# Measured first-token times on 26 Oct: 3.1-flash-lite 1.2-1.8 s, flash-lite-latest ~3.9 s,
# 3.5-flash-lite 4-5.5 s; 3.6-flash and 3.5-flash timed out or answered 503. Phone calls need speed,
# so the fastest models go first and the agent's own choice follows.
FAST_MODELS = [m.strip() for m in os.environ.get(
    "PHONE_FAST_MODELS", "gemini-3.1-flash-lite,gemini-flash-lite-latest,gemini-3.5-flash-lite").split(",") if m.strip()]
GEMINI_FALLBACKS = ["gemini-3.5-flash", "gemini-3.6-flash"]
# With thinking at "minimal" the fast models answer well inside 3 s; a stuck model is skipped sooner.
FIRST_TOKEN_TIMEOUT = float(os.environ.get("LLM_FIRST_TOKEN_TIMEOUT", "3.5"))
HEDGE_AFTER = float(os.environ.get("LLM_HEDGE_AFTER", "1.1"))
# Whichever model answered last is tried first next time (shared across calls), so only the first turn
# after an outage pays for the fallback search.
_LAST_GOOD_MODEL: dict = {}

from pipecat.services.google.llm import GoogleLLMService  # noqa: E402


from google.genai.types import Content as _GContent  # noqa: E402
from google.genai.types import GenerateContentConfig  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMSpecificMessage as _LLMSpecific  # noqa: E402
from pipecat.services.google.llm import GoogleLLMContext as _GoogleLLMContext  # noqa: E402

_orig_from_standard = _GoogleLLMContext.from_standard_message


def _from_standard_message(self, message):
    # Gemini 3 returns "thought signature" messages wrapped in LLMSpecificMessage. The legacy Google
    # context crashed on them ('LLMSpecificMessage' object is not subscriptable), so the agent's own
    # replies were not saved and later requests ended with a model turn (400).
    if isinstance(message, _LLMSpecific):
        return message.message if isinstance(message.message, _GContent) else None
    return _orig_from_standard(self, message)


_GoogleLLMContext.from_standard_message = _from_standard_message


class FallbackGoogleLLM(GoogleLLMService):
    """Gemini with hedged requests, per-model timeout and automatic fallback.

    On 26 Oct gemini-3.5-flash hung 45 s and then answered 503 "high demand": the caller heard silence.
    On 27 Oct gemini-3.1-flash-lite's first token took 1.0-2.9 s for the same short prompt. A second,
    identical request after HEDGE_AFTER seconds cuts those slow tails: whichever answers first is used.
    """

    def _config_for(self, params_from_context, model: str):
        tools = params_from_context["tools"] or self._tools or []
        gp = self._build_generation_params(
            system_instruction=params_from_context["system_instruction"], tools=tools, tool_config=self._tool_config)
        gp.pop("thinking_config", None)
        thinking = thinking_for(model)
        if thinking is not None:
            gp["thinking_config"] = thinking.model_dump(exclude_unset=True)
        return GenerateContentConfig(**gp)

    async def _open(self, params_from_context, model: str):
        # Model and config are passed explicitly, so parallel requests never share mutable settings.
        stream = await self._client.aio.models.generate_content_stream(
            model=model, contents=params_from_context["messages"], config=self._config_for(params_from_context, model))
        try:
            first = await stream.__anext__()
        except StopAsyncIteration:
            return None, None
        return first, stream

    async def _hedged(self, params_from_context, model: str):
        tasks = [asyncio.create_task(self._open(params_from_context, model))]
        deadline = time.monotonic() + FIRST_TOKEN_TIMEOUT
        hedged = False
        error = None
        try:
            while tasks:
                wait = (HEDGE_AFTER if not hedged else deadline - time.monotonic())
                done, _ = await asyncio.wait(tasks, timeout=max(0.0, min(wait, deadline - time.monotonic())),
                                             return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    tasks.remove(t)
                    if t.exception() is None:
                        return t.result()
                    error = t.exception()
                if time.monotonic() >= deadline:
                    raise asyncio.TimeoutError()
                if not hedged and (not done or not tasks):
                    hedged = True  # slow (or failed) first try: race a second identical request
                    tasks.append(asyncio.create_task(self._open(params_from_context, model)))
                elif not tasks:
                    break
            raise error or asyncio.TimeoutError()
        finally:
            for t in tasks:
                t.cancel()

    async def _stream_content(self, params_from_context):
        messages = params_from_context["messages"]
        if messages and getattr(messages[-1], "role", None) == "model":
            # Gemini rejects requests ending with a model turn (400); nothing new to answer yet.
            async def _empty():
                if False:
                    yield None
            return _empty()
        chosen = self._settings.model
        preferred = _LAST_GOOD_MODEL.get(chosen)
        models = ([preferred] if preferred else []) + FAST_MODELS + [chosen] + GEMINI_FALLBACKS
        models = list(dict.fromkeys(models))
        last_error = None
        t_start = time.time()
        await self.start_ttfb_metrics()
        for model in models:
            try:
                first, stream = await self._hedged(params_from_context, model)
            except Exception as e:  # 503 / 429 / 404 / timeout -> next model
                last_error = e
                logger.warning(f"LLM {model} failed ({type(e).__name__}: {str(e)[:120]}); trying next model")
                continue
            logger.info(f"LLM first token from {model} in {(time.time() - t_start) * 1000:.0f} ms")
            _LAST_GOOD_MODEL[chosen] = model

            async def _chain(first=first, stream=stream):
                if first is not None:
                    yield first
                if stream is not None:
                    async for chunk in stream:
                        yield chunk
            return _chain()
        raise last_error or RuntimeError("all Gemini models failed")


# Google retired the Gemini 2.x models: a retired name makes the agent hear the caller but never reply.
def current_gemini(model: str) -> str:
    model = (model or "").strip()
    if not model or model.startswith(("gemini-2.", "gemini-1.")):
        return "gemini-3.5-flash-lite" if "lite" in model else "gemini-3.5-flash"
    return model


# ---------------------------------------------------------------- serializers
class BrowserSerializer(FrameSerializer):
    """Same JSON protocol as the VoiceOS relay: {type:"audio",data}, interrupted, transport messages."""

    def __init__(self, on_close):
        super().__init__()
        self._on_close = on_close

    @property
    def type(self):
        return FrameSerializerType.TEXT if FrameSerializerType else "text"

    async def serialize(self, frame: Frame):
        if isinstance(frame, OutputAudioRawFrame):
            return json.dumps({"type": "audio", "data": base64.b64encode(frame.audio).decode()})
        if isinstance(frame, _Interrupt):
            return json.dumps({"type": "interrupted"})
        if isinstance(frame, (TransportMessageFrame, TransportMessageUrgentFrame)):
            return json.dumps(frame.message)
        return None

    async def deserialize(self, data):
        try:
            msg = json.loads(data)
        except Exception:
            return None
        if msg.get("type") == "audio" and msg.get("data"):
            return InputAudioRawFrame(audio=base64.b64decode(msg["data"]), sample_rate=16000, num_channels=1)
        if msg.get("type") == "session.close":
            self._on_close("ended")
        return None


class VobizSerializer(FrameSerializer):
    """Vobiz/Plivo-compatible stream: L16 16 kHz in, L16 16 kHz playAudio out, clearAudio on barge-in."""

    def __init__(self, stream_id: str, on_close):
        super().__init__()
        self.stream_id = stream_id
        self._on_close = on_close
        self.recorder = CallRecorder()  # both sides, as sent/received, for the Recordings page
        self._pending = bytearray()

    @property
    def type(self):
        return FrameSerializerType.TEXT if FrameSerializerType else "text"

    async def serialize(self, frame: Frame):
        if isinstance(frame, OutputAudioRawFrame):
            self.recorder.add_bot(frame.audio, frame.sample_rate)
            payload = base64.b64encode(frame.audio).decode()
            return json.dumps({"event": "playAudio", "streamId": self.stream_id, "media": {"contentType": "audio/x-l16", "sampleRate": frame.sample_rate, "payload": payload}})
        if isinstance(frame, _Interrupt):
            return json.dumps({"event": "clearAudio", "streamId": self.stream_id})
        return None

    async def deserialize(self, data):
        try:
            msg = json.loads(data)
        except Exception:
            return None
        media = msg.get("media") or {}
        # Bidirectional streams can echo the agent's own playback as an "outbound" track. Feeding it
        # to VAD makes the agent interrupt itself (clearAudio) so the caller hears silence.
        if msg.get("event") == "media" and media.get("payload") and media.get("track", "inbound") == "inbound":
            pcm = base64.b64decode(media["payload"])
            self.recorder.add_user(pcm)
            if PHONE_GAIN != 1.0 and pcm:
                # Caller audio arrives quiet; a fixed boost (clipped) lets speech detection hear it reliably.
                # Soft limiter instead of hard clipping: loud callers were distorted, which garbled what Gemini heard.
                x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) * (PHONE_GAIN / 32768.0)
                pcm = (np.tanh(x) * 32767.0).astype(np.int16).tobytes()
            # Vobiz sends 20 ms packets; passing 40 ms at a time halves the per-frame work on the small server.
            self._pending.extend(pcm)
            if len(self._pending) < PHONE_IN_BYTES:
                return None
            pcm, self._pending = bytes(self._pending), bytearray()
            return InputAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1)
        if msg.get("event") == "stop":
            self._on_close("hangup")
        return None


# ---------------------------------------------------------------- call tracking
# Caller farewells (English, Hindi, Hinglish, Gujarati). Matched on whole words of the final transcript.
_BYE_WORDS = re.compile(
    r"\b(bye|bye[- ]?bye|goodbye|good bye|ta[- ]?ta|alvida|अलविदा|बाय|टाटा|आवजो|avjo|aavjo)\b"
    r"|\b(ok|okay|chalo|theek hai|thank you|thanks)\W+(bye|bye[- ]?bye)\b"
    r"|\b(phone|call)\s+(rakh|rakhta|rakhti|kaat|cut)", re.IGNORECASE)
_AGENT_BYE = re.compile(r"\b(goodbye|good bye|bye|alvida|अलविदा|have a (great|good|nice) day)\b", re.IGNORECASE)


def is_farewell(text: str) -> bool:
    return bool(_BYE_WORDS.search(text or ""))


class Tracker(FrameProcessor):
    """Collects transcript, interruptions, latency and token usage; mirrors transcript to the browser."""

    def __init__(self, stats: dict, role: str, ui: bool):
        super().__init__()
        self.stats, self.role, self.ui = stats, role, ui

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        s = self.stats
        if self.role == "stt":
            if isinstance(frame, TranscriptionFrame) and frame.text.strip():
                add_turn(s, "user", frame.text.strip() + " ")
                if is_farewell(frame.text):
                    s["caller_bye"] = True
                    s["bye_reply_started"] = False
                    logger.info(f"caller said goodbye ({frame.text.strip()[:40]!r}); agent will say goodbye and hang up")
                if self.ui:
                    await self.push_frame(TransportMessageUrgentFrame(message={"type": "session.input_transcript.delta", "delta": frame.text.strip() + " "}))
        elif self.role == "input":
            if isinstance(frame, VADUserStoppedSpeakingFrame):
                s["user_stopped"] = time.time() - s.get("vad_stop_secs", 0)  # moment speech actually ended
            elif isinstance(frame, BotStartedSpeakingFrame):
                if not s.get("greeted"):
                    s["greeted"] = True
                    logger.info(f"greeting started {(time.time() - s.get('t_stream', time.time())) * 1000:.0f} ms "
                                f"after the phone stream opened")
                lv = s.get("last_voice")
                if lv and lv > s.get("bot_stopped_at", 0) and not s.get("user_stopped"):
                    s["latencies"].append((time.time() - lv) * 1000)
                    logger.info(f"turn latency {s['latencies'][-1]:.0f} ms (caller's last sound -> agent audio sent)")
                if s.get("caller_bye"):
                    s["bye_reply_started"] = True
                if s.get("user_stopped"):
                    s["latencies"].append((time.time() - s.pop("user_stopped")) * 1000)
                    logger.info(f"turn latency {s['latencies'][-1]:.0f} ms (caller stopped -> agent audio)")
            elif isinstance(frame, BotStoppedSpeakingFrame):
                s["bot_stopped_at"] = time.time()
                if not s.get("greeting_done"):
                    s["greeting_done"] = True
                    return await self.push_frame(frame, direction)
                # The goodbye has finished playing: end the call from our side.
                if s.get("hangup") and s.get("caller_bye") and s.get("bye_reply_started"):
                    s["bye_reply_started"] = False
                    s["hangup"]()
            elif isinstance(frame, _Interrupt):
                s["interruptions"] += 1
        else:
            if isinstance(frame, TTSTextFrame) and frame.text:
                add_turn(s, "assistant", frame.text + " ")
                # The agent saying "goodbye" must NOT end the call (28 Oct: it hung up on callers mid-conversation).
                # Only the caller saying bye ends it (see caller_bye).
                if self.ui:
                    await self.push_frame(TransportMessageUrgentFrame(message={"type": "session.output_transcript.delta", "delta": frame.text + " "}))
            elif isinstance(frame, MetricsFrame) and LLMUsageMetricsData:
                for d in frame.data:
                    if isinstance(d, LLMUsageMetricsData):
                        s["input_tokens"] += d.value.prompt_tokens or 0
                        s["output_tokens"] += d.value.completion_tokens or 0
        await self.push_frame(frame, direction)


class GreetingPlayer(FrameProcessor):
    """Live engine: plays the cached greeting instantly, or records Gemini's greeting to cache it."""

    def __init__(self, key: str | None, cached, text: str, on_saved=None):
        super().__init__()
        self._key, self._cached, self._text, self._on_saved = key, cached, text, on_saved
        self._recording = key is not None and cached is None
        self._pcm, self._rate = bytearray(), 24000
        self._started = False

    def played_already(self):
        self._recording = False

    async def play(self):
        for _ in range(150):  # frames may only be pushed once the pipeline has started (normally instant)
            if self._started:
                break
            await asyncio.sleep(0.02)
        rate, pcm = self._cached
        await self.push_frame(TTSStartedFrame())
        await self.push_frame(TTSTextFrame(self._text, aggregated_by="sentence"))
        step = rate // 10 * 2  # 100 ms per frame
        for i in range(0, len(pcm), step):
            await self.push_frame(TTSAudioRawFrame(bytes(pcm[i:i + step]), rate, 1))
        await self.push_frame(TTSStoppedFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self._started = True
        if self._recording:
            if isinstance(frame, OutputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
                self._pcm.extend(frame.audio)
                self._rate = frame.sample_rate
            elif isinstance(frame, _Interrupt):
                self._recording = False  # an interrupted greeting is not worth keeping
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._recording = False
                store_greeting(self._key, self._rate, bytes(self._pcm))
                if self._on_saved:
                    self._on_saved()
        await self.push_frame(frame, direction)


class GeminiAudioDrop(FrameProcessor):
    """Sarvam voice on Gemini Live: Gemini's own audio is dropped and its live transcript goes to Sarvam.

    If Sarvam fails during the call (bad key, quota, network), this switches back to Gemini's voice for
    the rest of the call and stops feeding Sarvam, so the caller never gets silence or a double voice."""

    def __init__(self, sarvam):
        super().__init__()
        self._sarvam = sarvam
        self.fallback = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.UPSTREAM:
            if isinstance(frame, ErrorFrame) and not self.fallback and getattr(frame, "processor", None) is self._sarvam:
                self.fallback = True
                logger.warning(f"Sarvam voice failed ({str(frame.error)[:100]}); using Gemini's voice for this call")
                if not frame.fatal:
                    return  # handled here: the call carries on in Gemini's voice
            await self.push_frame(frame, direction)
            return
        if self.fallback:
            if isinstance(frame, LLMTextFrame):
                return  # Gemini speaks itself now; Sarvam must not speak too
        elif isinstance(frame, (TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame, TTSTextFrame)):
            return  # Sarvam speaks (and transcribes) the reply instead
        await self.push_frame(frame, direction)


class BargeInGuard(FrameProcessor):
    """While the agent talks, only sustained caller speech reaches Gemini.

    Line noise, "hmm" and background voices made Gemini cut the agent off 10 times in one 85 s call
    (28 Oct) - that was the "breaking" voice. Short sounds during the agent's speech are replaced with
    silence; speech that stays loud for BARGE_MS is a real interruption and is passed on in full."""

    def __init__(self, barge_ms: int = 350):
        super().__init__()
        self._barge_ms = barge_ms
        self._bot = False
        self._loud_ms = 0.0
        self._quiet_ms = 0.0
        self._open = False
        self._held: list = []
        self._floor = 300.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot, self._open, self._loud_ms, self._held = True, False, 0.0, []
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot = False
            for held in self._held:  # anything held back is still the caller's speech
                await self.push_frame(held, FrameDirection.DOWNSTREAM)
            self._held = []
        elif isinstance(frame, InputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM and self._bot:
            samples = np.frombuffer(frame.audio, dtype=np.int16)
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) if samples.size else 0.0
            ms = 1000 * samples.size / max(1, frame.sample_rate)
            threshold = max(900.0, self._floor * 3)
            if rms > threshold:
                self._loud_ms += ms
                self._quiet_ms = 0.0
            else:
                self._floor = 0.95 * self._floor + 0.05 * rms
                self._quiet_ms += ms
                if self._quiet_ms > 250:
                    self._loud_ms = 0.0
            if self._open or self._loud_ms >= self._barge_ms:
                if not self._open:
                    self._open = True
                    for held in self._held:  # let Gemini hear the start of the interruption
                        await self.push_frame(held, FrameDirection.DOWNSTREAM)
                    self._held = []
                return await self.push_frame(frame, direction)
            if self._loud_ms > 0:
                self._held.append(frame)
                self._held = self._held[-20:]
            silent = InputAudioRawFrame(audio=b"\x00" * len(frame.audio), sample_rate=frame.sample_rate,
                                        num_channels=frame.num_channels)
            return await self.push_frame(silent, direction)
        await self.push_frame(frame, direction)


class VoiceClock(FrameProcessor):
    """Live engine: notes when the caller last made sound, to time each reply (cheap RMS, no model)."""

    def __init__(self, stats: dict, threshold: int = 600):
        super().__init__()
        self.stats, self.threshold = stats, threshold

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and frame.audio:
            samples = np.frombuffer(frame.audio, dtype=np.int16)
            if samples.size and float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) > self.threshold:
                self.stats["last_voice"] = time.time()
        await self.push_frame(frame, direction)


class LiveErrors(FrameProcessor):
    """Gemini Live failed to connect: forget the pinned model so the next call checks models again."""

    def __init__(self, key: str):
        super().__init__()
        self._key = key
        self._spoke = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._spoke = True
        elif isinstance(frame, ErrorFrame) and not self._spoke:
            logger.warning("Gemini Live failed before speaking; next call will re-check live models")
            forget_live_model(self._key)
        await self.push_frame(frame, direction)


class CallErrors(FrameProcessor):
    """Provider ErrorFrames travel upstream; persist a safe failure rather than silent success."""

    def __init__(self, stats: dict):
        super().__init__()
        self.stats = stats

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, ErrorFrame):
            # SDK error strings can include credentials or request URLs: never persist them.
            self.stats.setdefault("error", "Voice provider failed; check the Render voice server and provider settings.")
            logger.warning("voice provider emitted an error frame")
        await self.push_frame(frame, direction)


class OpeningGuard(FrameProcessor):
    """Phone only: keep line noise/echo away from STT until the greeting has finished playing.

    Sarvam turns a stray "okay"/"hmm" into a user turn, which cancelled the greeting within 0.2 s.
    """

    def __init__(self, max_secs: float = 12.0):
        super().__init__()
        self._open_at = None
        self._max_secs = max_secs
        self._done = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not self._done:
            if self._open_at is None:
                self._open_at = time.time()
            if isinstance(frame, BotStoppedSpeakingFrame) or time.time() - self._open_at > self._max_secs:
                self._done = True
                logger.info("greeting finished; listening to caller")
            elif direction == FrameDirection.DOWNSTREAM and isinstance(
                frame, (InputAudioRawFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
                        VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)):
                return  # drop caller audio/VAD while the greeting plays (no STT flush errors)
        await self.push_frame(frame, direction)


class ShortReplyFilter(FrameProcessor):
    """Phone only: ignore 1-2 word fragments ("haan", "hmm", line noise) heard while the agent speaks.

    Without barge-in each fragment queued its own LLM reply, so answers piled up and arrived late.
    """

    def __init__(self, min_words: int = 3):
        super().__init__()
        self._min_words = min_words
        self._bot_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif (self._bot_speaking and direction == FrameDirection.DOWNSTREAM
              and isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame))
              and len(frame.text.split()) < self._min_words and not is_farewell(frame.text)):
            logger.debug(f"ignored short fragment while agent speaks: {frame.text!r}")
            return
        await self.push_frame(frame, direction)


def add_turn(stats: dict, role: str, text: str):
    t = stats["transcript"]
    if t and t[-1]["role"] == role:
        t[-1]["text"] += text
    else:
        t.append({"role": role, "text": text})


def build_services(agent: dict, keys: dict, phone: bool = False):
    stt = make_stt(agent.get("stt", "deepgram"), agent.get("language", "English"), keys)
    llm = make_llm(agent.get("llm") or "google/gemini-3.5-flash", keys)
    if agent.get("tts") == "sarvam":
        lang = SARVAM_LANGS.get(agent.get("language", "Hindi"), "hi-IN")
        voice = (agent.get("voice") or "shubh").removeprefix("sarvam:")
        from sarvam_flash import PERSONAS, SarvamFlashTTSService
        if voice in PERSONAS:
            persona_language = PERSONAS[voice]["languages"]
            chosen_language = agent.get("language", "Hindi").replace("English (India)", "English")
            if chosen_language not in persona_language:
                raise ValueError(f"This Sarvam persona speaks {persona_language}; choose that language or a v3 voice")
        tts = None
        if os.environ.get("SARVAM_TTS_MODE", "stream") != "http":
            # Streaming WebSocket for v3 speakers: first words in a few hundred ms, no gaps between sentences.
            from sarvam_live import make_streaming_tts
            tts = make_streaming_tts(api_key=key(keys, "sarvam", "SARVAM_API_KEY"), voice=voice, language=lang,
                                     sample_rate=16000 if phone else 24000)
        if tts is None:
            # v4 personas / v2 voices (or SARVAM_TTS_MODE=http): versioned HTTPS adapter.
            tts = SarvamFlashTTSService(api_key=key(keys, "sarvam", "SARVAM_API_KEY"), voice_id=voice, language=lang)
    else:
        tts = ElevenLabsTTSService(
            api_key=key(keys, "elevenlabs", "ELEVENLABS_API_KEY"),
            voice_id=agent.get("voice") or DEFAULT_VOICE,
            model="eleven_flash_v2_5",
        )
    return stt, llm, tts


# ---------------------------------------------------------------- call runner
async def run_call(ws: WebSocket, call_id: str, token: str, phone: bool, stream_id: str = "", session_task=None,
                   stream_opened_at: float | None = None, agent_id: str = "", pre_greeted_until: float = 0.0):
    t_open = time.time()
    t_stream = stream_opened_at or t_open
    session_task = session_task or asyncio.ensure_future(voiceos("session", {"call_id": call_id, "token": token}))
    cached = _SESSION_CACHE.get(agent_id) if agent_id else None
    try:
        if cached and not session_task.done():
            # The database already confirmed (with this call's secret) that the call belongs to this agent, and
            # VoiceOS loaded this agent before: start now with that copy; VoiceOS still claims the call meanwhile.
            session = cached
            logger.info(f"call {call_id[:8]} started from the saved agent (VoiceOS still loading in the background)")
        else:
            session = await session_task
            logger.info(f"call {call_id[:8]} agent loaded from VoiceOS in {(time.time() - t_open) * 1000:.0f} ms")
            if agent_id:
                _SESSION_CACHE[agent_id] = session
    except Exception as e:
        logger.warning(f"session rejected: {e}")
        if not phone:
            await ws.send_text(json.dumps({"type": "app.error", "error": {"message": str(e)}}))
        await ws.close()
        return

    agent, keys = session["agent"], session["keys"]
    creds = {"call_id": call_id, "token": token}
    stats = {"transcript": [], "interruptions": 0, "latencies": [], "input_tokens": 0, "output_tokens": 0, "tools": []}
    stats["vad_stop_secs"] = 0.4
    stats["t_stream"] = t_stream  # local VAD reports "stopped" this long after speech actually ends
    started = time.time()
    finished = asyncio.Event()
    task_ref: dict = {}

    def hangup():
        """Farewell: let the last audio reach the caller, then close the stream (Vobiz ends the call)."""
        if "reason" in stats:
            return
        stats["reason"] = "farewell"

        async def _end():
            await asyncio.sleep(0.8)
            logger.info(f"call {call_id[:8]} ending after farewell")
            await finish()
            await asyncio.sleep(0.5)
            try:
                await ws.close()
            except Exception:
                pass
        asyncio.get_event_loop().create_task(_end())

    stats["hangup"] = hangup

    async def _confirm_session():
        """Started from the saved agent: VoiceOS must still accept the call, or the call ends."""
        try:
            fresh = await session_task
            if agent_id:
                _SESSION_CACHE[agent_id] = fresh
            logger.info(f"call {call_id[:8]} confirmed by VoiceOS {(time.time() - t_open) * 1000:.0f} ms after start")
        except Exception as e:
            logger.warning(f"call {call_id[:8]} rejected by VoiceOS: {e}")
            stats.setdefault("reason", "rejected")
            await finish("Call was not accepted by VoiceOS")
            try:
                await ws.close()
            except Exception:
                pass
    if session is cached and cached is not None:
        asyncio.get_event_loop().create_task(_confirm_session())

    def on_close(reason: str):
        if "reason" not in stats:
            stats["reason"] = reason
            asyncio.get_event_loop().create_task(finish())

    async def finish(error: str | None = None):
        if finished.is_set():
            return
        finished.set()
        error = error or stats.get("error")
        lat = stats["latencies"]
        try:  # the call must be claimed by VoiceOS before it can be saved
            await asyncio.wait_for(asyncio.shield(session_task), 15)
        except Exception:
            pass
        if phone and os.environ.get("RECORDINGS") == "1":
            try:
                await save_recording(_bridge(), call_id, token, serializer.recorder.wav())
            except Exception as e:  # a recording must never block saving the call
                logger.warning(f"recording skipped: {type(e).__name__}")
        try:
            await voiceos("finish", {
                **creds,
                "transcript": [{"role": t["role"], "text": t["text"].strip()} for t in stats["transcript"]],
                "duration_seconds": round(time.time() - started),
                "interruptions": stats["interruptions"],
                "avg_latency_ms": round(sum(lat) / len(lat)) if lat else 0,
                "input_tokens": stats["input_tokens"],
                "output_tokens": stats["output_tokens"],
                "tool_events": stats["tools"],
                "end_reason": stats.get("reason", "ended"),
                **({"error": error} if error else {}),
            })
        except Exception as e:
            logger.error(f"finish failed: {e}")
        task = task_ref.get("task")
        if task:
            if not phone:
                await task.queue_frame(TransportMessageUrgentFrame(message={"type": "session.closed"}))
            await task.queue_frame(EndFrame())

    # Gemini Live (one model hears and speaks) is the default; the three-step pipeline is the fallback.
    live_model = None
    gemini_key = key(keys, "gemini", "GEMINI_API_KEY")
    if os.environ.get("VOICE_ENGINE", "live") == "live":
        try:
            live_model = await pick_live_model(gemini_key)
        except Exception:
            logger.exception("live model check failed; using the three-step pipeline")
    thinking_task = asyncio.create_task(thinking_supported(gemini_key, live_model)) if live_model else None

    serializer = VobizSerializer(stream_id, on_close) if phone else BrowserSerializer(on_close)
    transport = FastAPIWebsocketTransport(
        websocket=ws,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            # 40 ms packets: half the per-packet work of 20 ms. On the free 0.15-CPU server the call hit the
            # CPU limit (28 Oct, 10:08 IST) and the agent's voice broke up; 40 ms is within Vobiz's 20-60 ms range.
            audio_out_10ms_chunks=4,
            # stop_secs is the silence the agent waits for before it answers: 0.8 s added a noticeable pause
            # to every turn. Phone keeps higher confidence/volume so line noise still isn't speech.
            # Phone audio from Vobiz is quiet: min_volume 0.6 missed most real speech, so turns only ended when
            # Sarvam's own endpointing fired (+0.5 s emulation). 0.3 still ignores line hiss.
            # Gemini Live detects speech itself; skipping the local detector frees CPU on the small server,
            # which was near its limit (0.12 of 0.15 CPU) and made the voice break up.
            vad_analyzer=None if live_model else (
                SileroVADAnalyzer(params=VADParams(confidence=0.65, start_secs=0.25, stop_secs=0.4, min_volume=0.3))
                if phone else SileroVADAnalyzer(params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=0.4, min_volume=0.6))),
            serializer=serializer,
            # Phone calls: strip hum/rumble and turn background noise down between words (see audio_cleanup).
            audio_in_filter=make_noise_filter() if phone else None,
        ),
    )


    schemas = []
    handlers = {}
    for t in agent.get("tools", []):
        params = t.get("parameters") or {}
        schemas.append(FunctionSchema(name=t["name"], description=t["description"],
                                      properties=params.get("properties", {}), required=params.get("required", [])))

        async def handler(p, _name=t["name"]):
            if not phone:
                await task_ref["task"].queue_frame(TransportMessageUrgentFrame(message={"type": "app.delegation.pending"}))
            try:
                result = await voiceos("tool", {**creds, "name": _name, "arguments": p.arguments})
            except Exception as e:
                result = {"tool": _name, "status": "failed", "error": str(e)}
            stats["tools"].append(result)
            if not phone:
                await task_ref["task"].queue_frame(TransportMessageUrgentFrame(message={"type": "app.tool.result", "plan": result}))
            await p.result_callback(result.get("result", result.get("error")))

        handlers[t["name"]] = handler
    tools_schema = ToolsSchema(standard_tools=schemas) if schemas else None

    try:
        sarvam_voice = None
        if live_model:
            stt = tts = None
            try:
                fast = await asyncio.wait_for(thinking_task, 3.5)
            except Exception:
                fast = False
            sarvam_voice = None
            if os.environ.get("LIVE_TTS", "sarvam") == "sarvam":
                # Gemini Live listens and thinks; Sarvam Bulbul speaks in the agent's chosen voice.
                from sarvam_live import make_live_sarvam_voice
                try:
                    sarvam_voice = make_live_sarvam_voice(
                        api_key=key(keys, "sarvam", "SARVAM_API_KEY"), voice=agent.get("voice"),
                        agent_language=agent.get("language"), sample_rate=16000 if phone else 24000)
                except Exception as e:
                    logger.warning(f"Sarvam voice unavailable ({type(e).__name__}); using Gemini's voice")
            g_key = greeting_key(agent, live_model + (f"|sarvam:{sarvam_voice.speaker}" if sarvam_voice else ""))
            g_cached = cached_greeting(g_key)
            pre_greeted = bool(pre_greeted_until) and g_cached is not None
            if g_cached:
                remember_agent_greeting(agent_id, g_key)
            elif agent_id:
                stats["remember_greeting_for"] = (agent_id, g_key)
            llm = build_live_llm(key=gemini_key, model=live_model,
                                 agent=already_greeted_agent(agent) if g_cached else agent,
                                 tools_schema=tools_schema, phone=phone, thinking=fast, speak_first=not g_cached)
            greeter = GreetingPlayer(g_key, g_cached, (agent.get("opening") or "").strip(),
                                     on_saved=lambda: remember_agent_greeting(*stats.get("remember_greeting_for", ("", None))))
            if pre_greeted:
                greeter.played_already()
                add_turn(stats, "assistant", (agent.get("opening") or "").strip() + " ")
            logger.info(f"call {call_id[:8]} engine: gemini live ({live_model})")
        else:
            stt, llm, tts = build_services(agent, keys, phone)
            logger.info(f"call {call_id[:8]} services: stt={type(stt).__name__} llm={agent.get('llm')} "
                        f"tts={type(tts).__name__} voice={agent.get('voice')} lang={agent.get('language')}")
    except Exception as e:
        # A setup failure must end the call visibly instead of leaving a silent open line.
        logger.exception("voice services failed to start")
        message = str(e) if isinstance(e, ValueError) else "Voice services failed to start; check the voice server logs."
        await finish(message)
        try:
            await ws.close()
        except Exception:
            pass
        return

    for name, handler in handlers.items():
        llm.register_function(name, handler)

    if live_model:
        live_context = LLMContext([] if g_cached else [opening_message(agent)])
        live_user, live_assistant = LLMContextAggregatorPair(live_context)
        pipeline = Pipeline([
            transport.input(),
            CallErrors(stats),
            LiveErrors(gemini_key),
            *([OpeningGuard(max_secs=max(0.0, pre_greeted_until - time.time()) if pre_greeted else 12.0)]
              if phone and agent.get("opening") else []),
            VoiceClock(stats),
            *([BargeInGuard()] if phone else []),
            Tracker(stats, "input", not phone),  # VAD, latency, goodbye hang-up
            live_user,
            # Caller transcript: Gemini Live sends it upstream and the user aggregator swallows it, so this
            # must sit right next to the model (it was before the aggregator: no caller text, no "bye" hang-up).
            Tracker(stats, "stt", not phone),
            llm,
            *([GeminiAudioDrop(sarvam_voice), sarvam_voice] if sarvam_voice else []),  # Sarvam speaks
            greeter,  # instant cached greeting (or records the first one)
            Tracker(stats, "assistant", not phone),  # agent transcript
            transport.output(),
            live_assistant,
        ])
    else:
        pipeline = None

    context = OpenAILLMContext(
        messages=[{"role": "system", "content": agent["instructions"]
                   + "\n\nThis is a live voice call: reply in 1-2 short spoken sentences unless asked for detail."
                   + "\nWhen the caller says bye or wants to end the call, reply only with a short polite goodbye"
                     " (for example \"Thank you for your time. Goodbye!\") in their language and ask nothing more;"
                     " the call ends automatically after your goodbye."}],
        tools=tools_schema,
    )
    agg = None if live_model else llm.create_context_aggregator(context)

    pipeline = pipeline or Pipeline([
        transport.input(),
        CallErrors(stats),
        *([OpeningGuard()] if phone and agent.get("opening") else []),
        Tracker(stats, "input", not phone),  # VAD, interruptions, latency
        stt,
        *([ShortReplyFilter()] if phone else []),
        Tracker(stats, "stt", not phone),  # final transcriptions
        agg.user(),
        llm,
        tts,
        Tracker(stats, "assistant", not phone),
        transport.output(),
        agg.assistant(),
    ])
    task = PipelineTask(pipeline, params=PipelineParams(
        # Phone lines carry noise and echo that kept cancelling the agent's speech before the caller heard
        # anything. On phone calls the agent always finishes its sentence, then listens.
        # Gemini Live only cuts in on clear speech (low start sensitivity), so callers can interrupt it.
        allow_interruptions=bool(live_model) or not phone,
        enable_metrics=True,
        enable_usage_metrics=True,
        audio_in_sample_rate=16000,
        # Vobiz/Plivo playAudio only accepts L16 at 8000 or 16000 Hz; 24 kHz audio is never played.
        audio_out_sample_rate=16000 if phone else 24000,
    ), enable_rtvi=not phone)  # phone calls have no web client: skip the RTVI layer that touches every frame
    task_ref["task"] = task

    @transport.event_handler("on_client_connected")
    async def _connected(_t, _c):
        if not phone:
            await task.queue_frame(TransportMessageUrgentFrame(message={"type": "app.connected"}))
        if live_model:
            if g_cached and not pre_greeted:
                await greeter.play()  # caller hears the greeting now; Gemini connects meanwhile
            await task.queue_frame(LLMRunFrame())  # Gemini Live speaks the opening (first call) / gets context
        elif agent.get("opening"):
            context.add_messages([{"role": "assistant", "content": agent["opening"]}])
            await task.queue_frame(TTSSpeakFrame(agent["opening"]))

    @transport.event_handler("on_client_disconnected")
    async def _disconnected(_t, _c):
        stats.setdefault("reason", "disconnected")
        await finish()
        await task.cancel()

    async def heartbeat():
        while not finished.is_set():
            await asyncio.sleep(10)
            lat = stats["latencies"]
            try:
                await voiceos("event", {**creds, "interruptions": stats["interruptions"],
                                        "avg_latency_ms": round(sum(lat) / len(lat)) if lat else 0,
                                        "live_status": f"gemini live · {live_model}" if live_model else "pipecat · live"})
            except Exception:
                pass

    hb = asyncio.create_task(heartbeat())
    try:
        await PipelineRunner(handle_sigint=False).run(task)
    except Exception as e:
        logger.exception("pipeline failed")
        await finish(str(e))
    finally:
        hb.cancel()
        if not finished.is_set():
            await finish()


async def lookup_call_agent(call_id: str, token: str) -> str:
    """Agent id for this call via the token-checked pipecat_call_agent database function ('' if unavailable)."""
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "")
    if not url or not key or not re.fullmatch(r"[0-9a-fA-F-]{36}", call_id or ""):
        return ""
    try:
        r = await _bridge().post(f"{url}/rest/v1/rpc/pipecat_call_agent", headers={"apikey": key},
                                 json={"p_call": call_id, "p_token": token}, timeout=2.0)
        aid = r.json() if r.status_code == 200 else None
        return aid if isinstance(aid, str) and re.fullmatch(r"[0-9a-fA-F-]{36}", aid) else ""
    except Exception:
        return ""


async def play_instant_greeting(ws: WebSocket, stream_id: str, agent_id: str, opened_at: float) -> float:
    """Sends this agent's saved greeting to Vobiz the moment the call connects, before the agent loads.

    Only pre-recorded greeting audio is played here: no API key or agent data is used until VoiceOS has
    confirmed the call. Returns when the greeting will have finished playing (0 if none was sent).
    """
    g = greeting_for_agent(agent_id)
    if not g:
        return 0.0
    rate, pcm, _ = g
    pcm16 = to_phone_rate(rate, pcm)
    step = 1280  # 40 ms at 16 kHz
    try:
        for i in range(0, len(pcm16), step):
            await ws.send_text(json.dumps({"event": "playAudio", "streamId": stream_id, "media": {
                "contentType": "audio/x-l16", "sampleRate": 16000,
                "payload": base64.b64encode(pcm16[i:i + step]).decode()}}))
    except Exception:
        return 0.0
    logger.info(f"instant greeting sent {(time.time() - opened_at) * 1000:.0f} ms after the phone stream opened")
    return time.time() + len(pcm16) / 2 / 16000


@app.on_event("startup")
async def _keep_awake():
    """Render's free plan puts the server to sleep after 15 idle minutes; waking takes ~70 s, so Vobiz
    gave up and the call hung up (27 Oct, 16:33 and 16:34 IST). With KEEP_AWAKE=1 the server pings its
    own public address every 10 minutes so it never sleeps. Set it on one service only: Render's free
    hours (750/month) cover a single always-on service."""
    url = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
    if os.environ.get("KEEP_AWAKE") != "1" or not url:
        return

    async def loop():
        while True:
            await asyncio.sleep(int(os.environ.get("KEEP_AWAKE_SECS", "600")))
            try:
                await _bridge().get(f"{url}/health", timeout=30)
            except Exception as e:
                logger.warning(f"keep-awake ping failed: {e}")
    asyncio.get_event_loop().create_task(loop())
    logger.info(f"keep-awake on: pinging {url}/health every 10 minutes")


@app.get("/health")
async def health():
    return {"ok": True, "configured": bool(SECRET), "voiceos": VOICEOS_URL, "revision": "steady-voice-2026-10-28"}


@app.websocket("/ws")
async def browser_ws(ws: WebSocket):
    await ws.accept()
    await run_call(ws, ws.query_params.get("call", ""), ws.query_params.get("token", ""), phone=False)


@app.websocket("/phone")
async def phone_ws(ws: WebSocket):
    await ws.accept()
    stream_opened_at = time.time()
    call_id = ws.query_params.get("call", "")
    logger.info(f"phone stream opened for call {call_id[:8]}")
    # Load the agent while Vobiz is still sending its start event.
    session_task = asyncio.create_task(voiceos("session", {"call_id": call_id, "token": ws.query_params.get("token", "")}))
    # Which agent is this call for? A one-row database lookup (~0.2 s) instead of waiting ~5 s for VoiceOS,
    # so the agent's saved greeting can play the moment the phone is picked up.
    agent_task = None
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", ws.query_params.get("agent", "") or ""):
        agent_task = asyncio.create_task(lookup_call_agent(call_id, ws.query_params.get("token", "")))
    stream_id = ""
    # Vobiz sends a "start" event first (text or binary JSON); never crash or hang before the call starts.
    try:
        for _ in range(10):
            raw = await asyncio.wait_for(ws.receive(), timeout=10)
            if raw.get("type") == "websocket.disconnect":
                logger.warning("phone stream closed before start")
                return
            data = raw.get("text") or (raw.get("bytes") or b"").decode("utf-8", "ignore")
            try:
                msg = json.loads(data)
            except Exception:
                continue
            if msg.get("event") == "start":
                logger.info(f"phone stream start: mediaFormat={msg.get('start', {}).get('mediaFormat') or msg.get('mediaFormat')}")
                stream_id = msg.get("start", {}).get("streamId") or msg.get("streamId", "")
                break
    except asyncio.TimeoutError:
        logger.warning("phone stream sent no start event; continuing")
    except Exception as e:
        logger.exception(f"phone stream failed before start: {e}")
        return
    try:
        agent_id = ws.query_params.get("agent", "")
        if agent_task:
            try:
                agent_id = await asyncio.wait_for(agent_task, 2.0) or ""
            except Exception:
                agent_id = ""
        pre_greeted_until = 0.0
        if re.fullmatch(r"[0-9a-fA-F-]{36}", agent_id or "") and stream_id:
            pre_greeted_until = await play_instant_greeting(ws, stream_id, agent_id, stream_opened_at)
        await run_call(ws, call_id, ws.query_params.get("token", ""), phone=True, stream_id=stream_id,
                       session_task=session_task, stream_opened_at=stream_opened_at,
                       agent_id=agent_id if re.fullmatch(r"[0-9a-fA-F-]{36}", agent_id or "") else "",
                       pre_greeted_until=pre_greeted_until)
    except Exception as e:
        logger.exception(f"phone call failed: {e}")
