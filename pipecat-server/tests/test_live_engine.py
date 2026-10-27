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


class BuildLive(unittest.TestCase):
    def test_settings(self):
        llm = live_engine.build_live_llm(key="k", model="gemini-3.8-live", agent=AGENT, tools_schema=None, phone=True)
        self.assertEqual((llm._settings.model, llm._settings.voice, llm._settings.language),
                         ("models/gemini-3.8-live", "Zephyr", "hi-IN"))
        self.assertIn("goodbye", llm._settings.system_instruction)
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
        with patch.object(main, "voiceos", fake_voiceos), patch.object(main, "pick_live_model", fake_pick), \
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
