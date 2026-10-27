import asyncio
import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("PIPECAT_SHARED_SECRET", "test-only")
import live_engine  # noqa: E402
import main  # noqa: E402

AGENT = {"name": "a", "instructions": "Be helpful.", "opening": "Hi, this is Sam.", "language": "Hindi",
         "llm": "google/gemini-3.6-flash", "stt": "sarvam", "tts": "sarvam", "voice": "ishita", "tools": []}


class PickModel(unittest.IsolatedAsyncioTestCase):
    async def test_highest_priority_working_model_is_cached(self):
        tried = []

        async def fake(key, model, timeout):
            tried.append(model)
            return model in ("gemini-3.8-live", "gemini-2.5-flash-native-audio-preview-12-2025")
        live_engine._GOOD.clear()
        with patch.object(live_engine, "_handshake", fake), patch.object(live_engine, "PINNED", ""):
            self.assertEqual(await live_engine.pick_live_model("k"), "gemini-3.8-live")
            n = len(tried)
            self.assertEqual(await live_engine.pick_live_model("k"), "gemini-3.8-live")
            self.assertEqual(len(tried), n)  # cached

    async def test_no_live_model_returns_none(self):
        async def fake(key, model, timeout):
            return False
        live_engine._GOOD.clear()
        with patch.object(live_engine, "_handshake", fake), patch.object(live_engine, "PINNED", ""):
            self.assertIsNone(await live_engine.pick_live_model("other"))

    async def test_pinned_model_skips_check_until_it_fails(self):
        calls = []

        async def fake(key, model, timeout):
            calls.append(model)
            return model == "gemini-3.8-live"
        live_engine._GOOD.clear()
        with patch.object(live_engine, "_handshake", fake):
            self.assertEqual(await live_engine.pick_live_model("p"), live_engine.PINNED)
            self.assertEqual(calls, [])
            live_engine.forget_live_model("p")
            self.assertEqual(await live_engine.pick_live_model("p"), "gemini-3.8-live")


class LatencyClock(unittest.IsolatedAsyncioTestCase):
    async def test_reply_time_measured_from_callers_last_sound(self):
        import numpy as np
        from pipecat.frames.frames import BotStartedSpeakingFrame, InputAudioRawFrame
        from pipecat.processors.frame_processor import FrameDirection
        stats = {"transcript": [], "latencies": []}
        clock, tracker = main.VoiceClock(stats), main.Tracker(stats, "input", False)
        for p in (clock, tracker):
            async def push(frame, direction=FrameDirection.DOWNSTREAM):
                return None
            p.push_frame = push
        loud = (np.ones(320, dtype=np.int16) * 3000).tobytes()
        quiet = np.zeros(320, dtype=np.int16).tobytes()
        await clock.process_frame(InputAudioRawFrame(audio=loud, sample_rate=16000, num_channels=1), FrameDirection.DOWNSTREAM)
        await clock.process_frame(InputAudioRawFrame(audio=quiet, sample_rate=16000, num_channels=1), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.05)
        await tracker.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        self.assertEqual(len(stats["latencies"]), 1)
        self.assertGreaterEqual(stats["latencies"][0], 40)

    def test_fast_thinking_choice(self):
        self.assertEqual(str(live_engine.fast_thinking("gemini-3.1-flash-live-preview").thinking_level.value), "MINIMAL")
        self.assertEqual(live_engine.fast_thinking("gemini-2.5-flash-native-audio-preview-12-2025").thinking_budget, 0)


class InstantGreeting(unittest.TestCase):
    def run_call(self, cached):
        sent, out = [], []

        async def fake_voiceos(path, payload):
            sent.append(path)
            return {"agent": AGENT, "keys": {"gemini": "k"}} if path == "session" else {}

        async def fake_pick(key):
            return "gemini-3.8-live"

        async def ok(key, model):
            return True

        async def no_connect(self, *a, **k):
            return None
        with patch.object(main, "voiceos", fake_voiceos), patch.object(main, "pick_live_model", fake_pick), \
                patch.object(main, "thinking_supported", ok), patch.object(main, "cached_greeting", lambda k: cached), \
                patch("pipecat.services.google.gemini_live.llm.GeminiLiveLLMService._connect", no_connect):
            from fastapi.testclient import TestClient
            with TestClient(main.app).websocket_connect("/ws?call=c&token=t") as ws:
                for _ in range(40):
                    m = json.loads(ws.receive_text())
                    out.append(m.get("type"))
                    if m.get("type") == "audio":
                        break
                ws.send_text(json.dumps({"type": "session.close"}))
        return out

    def test_cached_greeting_plays_without_gemini(self):
        out = self.run_call((24000, b"\x10\x00" * 24000))  # 1 s of audio
        self.assertIn("audio", out)

    def test_cache_key_and_store(self):
        k = live_engine.greeting_key(AGENT, "m")
        self.assertIsNotNone(k)
        self.assertIsNone(live_engine.greeting_key({**AGENT, "opening": ""}, "m"))
        live_engine.store_greeting(k, 24000, b"\x00\x00" * 24000 * 2)
        self.assertEqual(live_engine.cached_greeting(k)[0], 24000)
        self.assertIn("Do not greet again", live_engine.already_greeted_agent(AGENT)["instructions"])


class BuildLive(unittest.TestCase):
    def test_settings(self):
        llm = live_engine.build_live_llm(key="k", model="gemini-3.8-live", agent=AGENT, tools_schema=None, phone=True)
        self.assertEqual((llm._settings.model, llm._settings.voice, llm._settings.language),
                         ("models/gemini-3.8-live", "Zephyr", "hi-IN"))
        self.assertIn("goodbye", llm._settings.system_instruction)
        fast = live_engine.build_live_llm(key="k", model="gemini-3.1-flash-live-preview", agent=AGENT, tools_schema=None,
                                          phone=True, thinking=True)
        self.assertEqual(str(fast._settings.thinking.thinking_level.value), "MINIMAL")
        self.assertIn("Hi, this is Sam.", live_engine.opening_message(AGENT)["content"])


class LiveCallStarts(unittest.TestCase):
    def test_browser_call_runs_live_pipeline_and_finishes(self):
        sent = []

        async def fake_voiceos(path, payload):
            sent.append(path)
            if path == "session":
                return {"agent": AGENT, "keys": {"gemini": "k"}}
            return {}

        async def fake_pick(key):
            return "gemini-3.8-live"

        async def no_connect(self, *a, **k):
            return None

        async def fast_ok(key, model):
            return True
        with patch.object(main, "voiceos", fake_voiceos), patch.object(main, "pick_live_model", fake_pick), \
                patch.object(main, "thinking_supported", fast_ok), \
                patch("pipecat.services.google.gemini_live.llm.GeminiLiveLLMService._connect", no_connect):
            from fastapi.testclient import TestClient
            with TestClient(main.app).websocket_connect("/ws?call=c&token=t") as ws:
                msg = json.loads(ws.receive_text())
                self.assertEqual(msg["type"], "app.connected")
                ws.send_text(json.dumps({"type": "session.close"}))
                for _ in range(5):
                    if json.loads(ws.receive_text()).get("type") == "session.closed":
                        break
        self.assertIn("finish", sent)


if __name__ == "__main__":
    unittest.main()
