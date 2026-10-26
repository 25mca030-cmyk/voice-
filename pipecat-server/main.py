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
import time

import httpx
from fastapi import FastAPI, WebSocket
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
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

app = FastAPI(title="VoiceOS Pipecat")


# ---------------------------------------------------------------- VoiceOS bridge (signed)
async def voiceos(path: str, payload: dict) -> dict:
    body = json.dumps(payload, separators=(",", ":"))
    ts = str(int(time.time()))
    sig = hmac.new(SECRET.encode(), f"{ts}.{body}".encode(), hashlib.sha256).hexdigest()
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(
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
        from pipecat.services.sarvam.stt import SarvamSTTService
        return SarvamSTTService(api_key=key(keys, "sarvam", "SARVAM_API_KEY"))
    if provider == "openai":
        from pipecat.services.openai.stt import OpenAISTTService
        return OpenAISTTService(api_key=key(keys, "openai", "OPENAI_API_KEY"), model="gpt-4o-transcribe", language=code)
    from deepgram import LiveOptions
    from pipecat.services.deepgram.stt import DeepgramSTTService
    return DeepgramSTTService(
        api_key=key(keys, "deepgram", "DEEPGRAM_API_KEY"),
        live_options=LiveOptions(language="multi" if code not in ("en", "hi", "es", "fr", "de") else code, model="nova-3"),
    )


def make_llm(spec: str, keys: dict):
    provider, _, model = spec.partition("/")
    if provider == "openai":
        from pipecat.services.openai.llm import OpenAILLMService
        return OpenAILLMService(api_key=key(keys, "openai", "OPENAI_API_KEY"), model=model or "gpt-4.1-mini")
    if provider == "anthropic":
        from pipecat.services.anthropic.llm import AnthropicLLMService
        return AnthropicLLMService(api_key=key(keys, "anthropic", "ANTHROPIC_API_KEY"), model=model or "claude-sonnet-4-5")
    from pipecat.services.google.llm import GoogleLLMService
    return GoogleLLMService(api_key=key(keys, "gemini", "GEMINI_API_KEY"), model=model or "gemini-2.5-flash")


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
    """Vobiz/Plivo-compatible stream: L16 16 kHz in, L16 24 kHz playAudio out, clearAudio on barge-in."""

    def __init__(self, stream_id: str, on_close):
        super().__init__()
        self.stream_id = stream_id
        self._on_close = on_close

    @property
    def type(self):
        return FrameSerializerType.TEXT if FrameSerializerType else "text"

    async def serialize(self, frame: Frame):
        if isinstance(frame, OutputAudioRawFrame):
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
            return InputAudioRawFrame(audio=base64.b64decode(media["payload"]), sample_rate=16000, num_channels=1)
        if msg.get("event") == "stop":
            self._on_close("hangup")
        return None


# ---------------------------------------------------------------- call tracking
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
                if self.ui:
                    await self.push_frame(TransportMessageUrgentFrame(message={"type": "session.input_transcript.delta", "delta": frame.text.strip() + " "}))
        elif self.role == "input":
            if isinstance(frame, UserStoppedSpeakingFrame):
                s["user_stopped"] = time.time()
            elif isinstance(frame, BotStartedSpeakingFrame) and s.get("user_stopped"):
                s["latencies"].append((time.time() - s.pop("user_stopped")) * 1000)
            elif isinstance(frame, _Interrupt):
                s["interruptions"] += 1
        else:
            if isinstance(frame, TTSTextFrame) and frame.text:
                add_turn(s, "assistant", frame.text + " ")
                if self.ui:
                    await self.push_frame(TransportMessageUrgentFrame(message={"type": "session.output_transcript.delta", "delta": frame.text + " "}))
            elif isinstance(frame, MetricsFrame) and LLMUsageMetricsData:
                for d in frame.data:
                    if isinstance(d, LLMUsageMetricsData):
                        s["input_tokens"] += d.value.prompt_tokens or 0
                        s["output_tokens"] += d.value.completion_tokens or 0
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


def add_turn(stats: dict, role: str, text: str):
    t = stats["transcript"]
    if t and t[-1]["role"] == role:
        t[-1]["text"] += text
    else:
        t.append({"role": role, "text": text})


def build_services(agent: dict, keys: dict):
    stt = make_stt(agent.get("stt", "deepgram"), agent.get("language", "English"), keys)
    llm = make_llm(agent.get("llm", "google/gemini-2.5-flash"), keys)
    if agent.get("tts") == "sarvam":
        lang = SARVAM_LANGS.get(agent.get("language", "Hindi"), "hi-IN")
        voice = (agent.get("voice") or "shubh").removeprefix("sarvam:")
        from sarvam_flash import PERSONAS, SarvamFlashTTSService
        if voice in PERSONAS:
            persona_language = PERSONAS[voice]["languages"]
            chosen_language = agent.get("language", "Hindi").replace("English (India)", "English")
            if chosen_language not in persona_language:
                raise ValueError(f"This Sarvam persona speaks {persona_language}; choose that language or a v3 voice")
        # One versioned HTTPS adapter for v2/v3/v4: SDK WebSocket configs drift between releases.
        tts = SarvamFlashTTSService(api_key=key(keys, "sarvam", "SARVAM_API_KEY"), voice_id=voice, language=lang)
    else:
        tts = ElevenLabsTTSService(
            api_key=key(keys, "elevenlabs", "ELEVENLABS_API_KEY"),
            voice_id=agent.get("voice") or DEFAULT_VOICE,
            model="eleven_flash_v2_5",
        )
    return stt, llm, tts


# ---------------------------------------------------------------- call runner
async def run_call(ws: WebSocket, call_id: str, token: str, phone: bool, stream_id: str = ""):
    try:
        session = await voiceos("session", {"call_id": call_id, "token": token})
    except Exception as e:
        logger.warning(f"session rejected: {e}")
        if not phone:
            await ws.send_text(json.dumps({"type": "app.error", "error": {"message": str(e)}}))
        await ws.close()
        return

    agent, keys = session["agent"], session["keys"]
    creds = {"call_id": call_id, "token": token}
    stats = {"transcript": [], "interruptions": 0, "latencies": [], "input_tokens": 0, "output_tokens": 0, "tools": []}
    started = time.time()
    finished = asyncio.Event()
    task_ref: dict = {}

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

    serializer = VobizSerializer(stream_id, on_close) if phone else BrowserSerializer(on_close)
    transport = FastAPIWebsocketTransport(
        websocket=ws,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            audio_out_10ms_chunks=4,
            # Phone lines carry noise: require longer, more confident speech before barging in.
            vad_analyzer=SileroVADAnalyzer(params=VADParams(confidence=0.75, start_secs=0.4, stop_secs=0.8, min_volume=0.6))
            if phone else SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    try:
        stt, llm, tts = build_services(agent, keys)
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

    schemas = []
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

        llm.register_function(t["name"], handler)

    context = OpenAILLMContext(
        messages=[{"role": "system", "content": agent["instructions"]}],
        tools=ToolsSchema(standard_tools=schemas) if schemas else None,
    )
    agg = llm.create_context_aggregator(context)

    pipeline = Pipeline([
        transport.input(),
        CallErrors(stats),
        Tracker(stats, "input", not phone),  # VAD, interruptions, latency
        stt,
        Tracker(stats, "stt", not phone),  # final transcriptions
        agg.user(),
        llm,
        tts,
        Tracker(stats, "assistant", not phone),
        transport.output(),
        agg.assistant(),
    ])
    task = PipelineTask(pipeline, params=PipelineParams(
        allow_interruptions=True,
        enable_metrics=True,
        enable_usage_metrics=True,
        audio_in_sample_rate=16000,
        audio_out_sample_rate=24000,
    ))
    task_ref["task"] = task

    @transport.event_handler("on_client_connected")
    async def _connected(_t, _c):
        if not phone:
            await task.queue_frame(TransportMessageUrgentFrame(message={"type": "app.connected"}))
        if agent.get("opening"):
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
                                        "live_status": "pipecat · live"})
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


@app.get("/health")
async def health():
    return {"ok": True, "configured": bool(SECRET), "voiceos": VOICEOS_URL, "revision": "sarvam-phone-settings-2026-10-26"}


@app.websocket("/ws")
async def browser_ws(ws: WebSocket):
    await ws.accept()
    await run_call(ws, ws.query_params.get("call", ""), ws.query_params.get("token", ""), phone=False)


@app.websocket("/phone")
async def phone_ws(ws: WebSocket):
    await ws.accept()
    call_id = ws.query_params.get("call", "")
    logger.info(f"phone stream opened for call {call_id[:8]}")
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
                stream_id = msg.get("start", {}).get("streamId") or msg.get("streamId", "")
                break
    except asyncio.TimeoutError:
        logger.warning("phone stream sent no start event; continuing")
    except Exception as e:
        logger.exception(f"phone stream failed before start: {e}")
        return
    try:
        await run_call(ws, call_id, ws.query_params.get("token", ""), phone=True, stream_id=stream_id)
    except Exception as e:
        logger.exception(f"phone call failed: {e}")
