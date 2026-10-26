"""Offline audio-boundary regression tests (no provider keys or phone calls)."""
import ast
import asyncio
import base64
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import wave

ROOT = Path(__file__).resolve().parents[1]


class Frame:
    def __init__(self, *args, **kwargs):
        self.audio = args[0] if args else kwargs.get("audio", b"")
        self.sample_rate = args[1] if len(args) > 1 else kwargs.get("sample_rate", 24000)
        self.num_channels = args[2] if len(args) > 2 else kwargs.get("num_channels", 1)
        self.context_id = kwargs.get("context_id")
        self.error = args[0] if args else kwargs.get("error")


class TTSService:
    def __init__(self, **kwargs):
        self.settings = kwargs

    # Pipecat 0.0.108 has no set_model_name; the adapter must not depend on it.


FRAMES = {n: type(n, (Frame,), {}) for n in (
    "ErrorFrame", "TTSStartedFrame", "TTSStoppedFrame", "TTSAudioRawFrame",
    "OutputAudioRawFrame", "InputAudioRawFrame", "InterruptionFrame")}


def load_adapter():
    # Stub only Pipecat's dependency-heavy base classes; execute the real HTTP adapter.
    modules = {n: types.ModuleType(n) for n in (
        "pipecat", "pipecat.services", "pipecat.services.tts_service",
        "pipecat.frames", "pipecat.frames.frames")}
    modules["pipecat.services.tts_service"].TTSService = TTSService
    for name, cls in FRAMES.items():
        setattr(modules["pipecat.frames.frames"], name, cls)
    spec = importlib.util.spec_from_file_location("voiceos_test_sarvam", ROOT / "sarvam_flash.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


ADAPTER = load_adapter()


def wav_payload(rate=24000, channels=1):
    pcm = bytes([10, 0, 20, 0]) * 240
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return base64.b64encode(output.getvalue()).decode(), pcm


class SarvamAudioTests(unittest.IsolatedAsyncioTestCase):
    async def synthesize(self, voice="ishita", audios=None, status=200):
        calls = []
        encoded, pcm = wav_payload()
        class Client:
            def __init__(self, **kwargs):
                self.options = kwargs
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, **kwargs):
                calls.append((url, kwargs, self.options))
                return types.SimpleNamespace(status_code=status, json=lambda: {"audios": [encoded] if audios is None else audios})
        service = ADAPTER.SarvamFlashTTSService(api_key="test-only", voice_id=voice, language="en-IN")
        with patch.object(ADAPTER.httpx, "AsyncClient", Client):
            frames = [frame async for frame in service.run_tts("Hello", "speech-context")]
        return frames, calls, pcm

    async def test_saved_v3_voice_generates_pcm_with_context(self):
        frames, calls, pcm = await self.synthesize()
        audio = next(f for f in frames if isinstance(f, FRAMES["TTSAudioRawFrame"]))
        self.assertEqual((audio.audio, audio.sample_rate, audio.num_channels, audio.context_id), (pcm, 24000, 1, "speech-context"))
        payload = calls[0][1]["json"]
        self.assertEqual((payload["model"], payload["speaker"], payload["target_language_code"]), ("bulbul:v3", "ishita", "en-IN"))
        self.assertFalse(calls[0][2]["follow_redirects"])

    async def test_legacy_voice_keeps_v2(self):
        _, calls, _ = await self.synthesize("anushka")
        self.assertEqual(calls[0][1]["json"]["model"], "bulbul:v2")

    async def test_persona_keeps_v4(self):
        _, calls, _ = await self.synthesize("ishita_en_customer")
        self.assertEqual(calls[0][1]["json"]["model"], "bulbul:v4-flash")

    async def test_provider_rejection_emits_error_not_audio(self):
        frames, _, _ = await self.synthesize(status=401)
        self.assertTrue(any(isinstance(f, FRAMES["ErrorFrame"]) for f in frames))
        self.assertFalse(any(isinstance(f, FRAMES["TTSAudioRawFrame"]) for f in frames))

    async def test_empty_audio_emits_error(self):
        frames, _, _ = await self.synthesize(audios=[])
        self.assertTrue(any(isinstance(f, FRAMES["ErrorFrame"]) for f in frames))

    async def test_wrong_rate_is_not_sent_as_24khz(self):
        encoded, _ = wav_payload(rate=8000)
        frames, _, _ = await self.synthesize(audios=[encoded])
        self.assertTrue(any(isinstance(f, FRAMES["ErrorFrame"]) for f in frames))
        self.assertFalse(any(isinstance(f, FRAMES["TTSAudioRawFrame"]) for f in frames))


class PhoneBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def load_class(self, name, globals):
        tree = ast.parse((ROOT / "main.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), globals)
        return globals[name]

    async def test_sarvam_pcm_reaches_phone_playaudio(self):
        frames, _, pcm = await SarvamAudioTests().synthesize()
        cls = self.load_class("VobizSerializer", {
            "FrameSerializer": object, "FrameSerializerType": None, "Frame": Frame,
            "OutputAudioRawFrame": FRAMES["TTSAudioRawFrame"], "_Interrupt": FRAMES["InterruptionFrame"],
            "InputAudioRawFrame": FRAMES["InputAudioRawFrame"], "json": json, "base64": base64})
        serializer = cls("STREAM_1", lambda reason: None)
        audio = next(f for f in frames if isinstance(f, FRAMES["TTSAudioRawFrame"]))
        event = json.loads(await serializer.serialize(audio))
        self.assertEqual((event["event"], event["streamId"], event["media"]["sampleRate"]), ("playAudio", "STREAM_1", 24000))
        self.assertEqual(base64.b64decode(event["media"]["payload"]), pcm)

    async def test_provider_error_is_retained_without_credentials(self):
        class Processor:
            async def process_frame(self, *args):
                pass
            async def push_frame(self, *args):
                pass
        cls = self.load_class("CallErrors", {"FrameProcessor": Processor, "Frame": Frame,
            "FrameDirection": object, "ErrorFrame": FRAMES["ErrorFrame"],
            "logger": types.SimpleNamespace(warning=lambda *args: None)})
        stats = {}
        await cls(stats).process_frame(FRAMES["ErrorFrame"]("private-provider-value"), "upstream")
        self.assertIn("Voice provider failed", stats["error"])
        self.assertNotIn("private-provider-value", stats["error"])


if __name__ == "__main__":
    unittest.main()