import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from pipecat.frames.frames import ErrorFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection

import sarvam_live


class SafeSarvamSTTTests(unittest.IsolatedAsyncioTestCase):
    def make(self):
        stt = sarvam_live.SafeSarvamSTT(api_key="test-only")
        stt._socket_client = AsyncMock()
        stt.push_frame = AsyncMock()
        return stt

    async def test_empty_flush_is_skipped(self):
        stt = self.make()
        with patch.object(sarvam_live.STTService, "process_frame", AsyncMock()):
            await stt.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        stt._socket_client.flush.assert_not_called()

    async def test_flush_after_audio(self):
        stt = self.make()
        async def ok(self_, audio):
            yield None
        with patch.object(sarvam_live.SarvamSTTService, "run_stt", ok):
            [f async for f in stt.run_stt(b"\0" * 640)]
        with patch.object(sarvam_live.STTService, "process_frame", AsyncMock()):
            await stt.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        stt._socket_client.flush.assert_awaited_once()

    async def test_send_error_reconnects_instead_of_failing_call(self):
        stt = self.make()
        async def broken(self_, audio):
            yield ErrorFrame(error="Error sending audio to Sarvam: received 1000 (OK)")
        stt._disconnect = AsyncMock()
        stt._connect = AsyncMock()
        with patch.object(sarvam_live.SarvamSTTService, "run_stt", broken):
            frames = [f async for f in stt.run_stt(b"\0" * 640)]
        self.assertFalse(any(isinstance(f, ErrorFrame) for f in frames))
        stt._connect.assert_awaited_once()


class StreamingTTSTests(unittest.TestCase):
    def test_v3_voice_streams_at_phone_rate(self):
        tts = sarvam_live.make_streaming_tts(api_key="t", voice="ishita", language="en-IN", sample_rate=16000)
        self.assertEqual((tts._settings.model, tts._settings.voice, tts._speech_sample_rate), ("bulbul:v3", "ishita", "16000"))

    def test_persona_uses_http_adapter(self):
        self.assertIsNone(sarvam_live.make_streaming_tts(api_key="t", voice="ishita_en_customer", language="en-IN", sample_rate=16000))


if __name__ == "__main__":
    unittest.main()
