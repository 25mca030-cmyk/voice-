import asyncio
import io
import os
import unittest
import wave
from unittest.mock import patch

import numpy as np

os.environ.setdefault("PIPECAT_SHARED_SECRET", "test-only")
import call_recording  # noqa: E402


class Recorder(unittest.TestCase):
    def test_mixes_both_sides_into_wav(self):
        r = call_recording.CallRecorder()
        tone = (np.ones(320, dtype=np.int16) * 1000).tobytes()
        for _ in range(100):  # 2 s of caller audio
            r.add_user(tone)
        r.add_bot(tone, 16000)
        r.add_bot(tone, 24000)  # wrong rate ignored
        data = r.wav()
        with wave.open(io.BytesIO(data)) as w:
            self.assertEqual((w.getframerate(), w.getnchannels()), (16000, 1))
            self.assertGreaterEqual(w.getnframes(), 32000)

    def test_quiet_caller_is_audible_next_to_loud_agent(self):
        # 28 Oct: the customer could not be heard in the recording (phone audio is much quieter).
        import time
        r = call_recording.CallRecorder()
        t = np.arange(320) / 16000
        quiet = (np.sin(2 * np.pi * 300 * t) * 700).astype(np.int16).tobytes()
        for _ in range(100):  # 2 s of quiet caller speech
            r.add_user(quiet)
        r._bot = [(r._t0 + 2.0, (np.sin(2 * np.pi * 300 * np.arange(32000) / 16000) * 20000).astype(np.int16).tobytes())]
        with wave.open(io.BytesIO(r.wav())) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        caller_peak, agent_peak = np.abs(x[:32000]).max(), np.abs(x[32000:]).max()
        self.assertGreater(caller_peak, agent_peak * 0.6)

    def test_turn_gaps(self):
        r = call_recording.CallRecorder()
        speech = (np.sin(2 * np.pi * 300 * np.arange(320) / 16000) * 2000).astype(np.int16).tobytes()
        silence = b"\x00\x00" * 320
        for i in range(100):  # caller talks for 1 s, then 1 s of silence
            r.add_user(speech if i < 50 else silence)
        r._bot = [(r._t0 + 1.8, b"\x01\x00" * 16000)]  # agent starts 0.8 s after the caller stopped
        gaps = r.turn_gaps()
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0], 0.8, delta=0.1)

    def test_too_short_is_skipped(self):
        r = call_recording.CallRecorder()
        r.add_user(b"\x00\x00" * 100)
        self.assertIsNone(r.wav())


class Upload(unittest.TestCase):
    def test_disabled_without_setting(self):
        os.environ.pop("RECORDINGS", None)
        self.assertFalse(asyncio.run(call_recording.save_recording(None, "c", "t", b"x")))

    def test_upload_flow(self):
        calls = []

        class Resp:
            def __init__(self, code, body):
                self.status_code, self._b = code, body

            def json(self):
                return self._b

        class Client:
            async def post(self, url, **kw):
                calls.append(url.split("/v1/")[1])
                if "pipecat_recording_path" in url:
                    return Resp(200, "u/c.wav")
                if "storage" in url:
                    return Resp(200, {})
                return Resp(200, True)
        env = {"RECORDINGS": "1", "SUPABASE_URL": "https://x.supabase.co", "SUPABASE_PUBLISHABLE_KEY": "k"}
        with patch.dict(os.environ, env):
            self.assertTrue(asyncio.run(call_recording.save_recording(Client(), "c", "t", b"RIFF")))
        self.assertEqual(calls, ["rpc/pipecat_recording_path", "object/recordings/u/c.wav", "rpc/pipecat_attach_recording"])


if __name__ == "__main__":
    unittest.main()
