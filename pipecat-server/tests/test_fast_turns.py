import asyncio
import os
import time
import types
import unittest

os.environ.setdefault("PIPECAT_SHARED_SECRET", "test-only")
import main  # noqa: E402
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, TranscriptionFrame, TTSTextFrame  # noqa: E402
from pipecat.processors.frame_processor import FrameDirection  # noqa: E402


class FarewellWords(unittest.TestCase):
    def test_caller_farewells(self):
        for text in ["Bye", "ok bye", "Bye bye", "goodbye", "बाय", "chalo bye", "Thank you, bye", "alvida", "phone rakhta hoon"]:
            self.assertTrue(main.is_farewell(text), text)

    def test_not_farewell(self):
        for text in ["Hello", "Yes", "maybe", "I want to buy", "byelaws question", "abhi busy hoon"]:
            self.assertFalse(main.is_farewell(text), text)


class HangupFlow(unittest.IsolatedAsyncioTestCase):
    async def test_caller_bye_then_goodbye_hangs_up_once(self):
        calls = []
        stats = {"transcript": [], "interruptions": 0, "latencies": [], "input_tokens": 0, "output_tokens": 0,
                 "hangup": lambda: calls.append(1)}
        stt = main.Tracker(stats, "stt", False)
        inp = main.Tracker(stats, "input", False)
        for t in (stt, inp):
            async def push(frame, direction=FrameDirection.DOWNSTREAM):
                return None
            t.push_frame = push
        # earlier agent speech ending must not hang up
        await inp.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        await stt.process_frame(TranscriptionFrame("Bye", "u", "now"), FrameDirection.DOWNSTREAM)
        await inp.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        self.assertEqual(calls, [])
        await inp.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)  # goodbye plays
        await inp.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        self.assertEqual(calls, [1])


class FakeStream:
    def __init__(self, delay, fail=False):
        self.delay, self.fail = delay, fail

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("503 overloaded")
        return "chunk"


class Hedging(unittest.IsolatedAsyncioTestCase):
    def llm(self, delays):
        llm = main.FallbackGoogleLLM(api_key="test-only", model="gemini-3.6-flash")
        seen = []

        async def gen(model, contents, config):
            d = delays.pop(0)
            seen.append((model, str(getattr(config.thinking_config.thinking_level, "value", "")).lower() if config.thinking_config else None))
            return FakeStream(*d) if isinstance(d, tuple) else FakeStream(d)
        llm._client = types.SimpleNamespace(aio=types.SimpleNamespace(models=types.SimpleNamespace(generate_content_stream=gen)))
        return llm, seen

    async def test_slow_first_request_is_raced(self):
        main.HEDGE_AFTER = 0.05
        llm, seen = self.llm([1.0, 0.01])
        t = time.monotonic()
        first, _ = await llm._hedged({"messages": [], "system_instruction": None, "tools": None}, "gemini-3.1-flash-lite")
        self.assertEqual(first, "chunk")
        self.assertLess(time.monotonic() - t, 0.5)
        self.assertEqual(seen, [("gemini-3.1-flash-lite", "minimal")] * 2)

    async def test_failed_request_retries_once_then_raises(self):
        main.HEDGE_AFTER = 0.05
        llm, _ = self.llm([(0.0, True), (0.0, True)])
        with self.assertRaises(RuntimeError):
            await llm._hedged({"messages": [], "system_instruction": None, "tools": None}, "gemini-3.1-flash-lite")


if __name__ == "__main__":
    unittest.main()


class LivePathFarewell(unittest.IsolatedAsyncioTestCase):
    async def test_agent_goodbye_alone_does_not_hang_up(self):
        calls = []
        stats = {"transcript": [], "latencies": [], "hangup": lambda: calls.append(1)}
        inp, asst = main.Tracker(stats, "input", False), main.Tracker(stats, "assistant", False)
        for t in (inp, asst):
            async def push(frame, direction=FrameDirection.DOWNSTREAM):
                return None
            t.push_frame = push
        await inp.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)   # greeting
        await asst.process_frame(TTSTextFrame("Hi, this is Sam. Do you have a moment?", aggregated_by="sentence"), FrameDirection.DOWNSTREAM)
        await inp.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        self.assertEqual(calls, [])
        await inp.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        await asst.process_frame(TTSTextFrame("Thank you for your time. Goodbye!", aggregated_by="sentence"), FrameDirection.DOWNSTREAM)
        await inp.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        self.assertEqual(calls, [])  # only the caller's "bye" ends the call


class LivePipelineOrder(unittest.TestCase):
    def test_caller_transcript_tracker_sits_next_to_model(self):
        src = open(main.__file__).read()
        i_user = src.index("            live_user,\n")
        i_stt = src.index('Tracker(stats, "stt", not phone),\n            llm,')
        self.assertLess(i_user, i_stt)


class InboundBatching(unittest.TestCase):
    def test_two_20ms_packets_make_one_40ms_frame(self):
        import base64
        import json
        ser = main.VobizSerializer("S", lambda r: None)
        pkt = json.dumps({"event": "media", "media": {"payload": base64.b64encode(b"\x00\x01" * 320).decode(), "track": "inbound"}})
        self.assertIsNone(asyncio.run(ser.deserialize(pkt)))
        frame = asyncio.run(ser.deserialize(pkt))
        self.assertEqual(len(frame.audio), 1280)


class BargeIn(unittest.IsolatedAsyncioTestCase):
    async def run_guard(self, loud_frames):
        import numpy as np
        from pipecat.frames.frames import InputAudioRawFrame
        g = main.BargeInGuard()
        out = []

        async def push(frame, direction=FrameDirection.DOWNSTREAM):
            if isinstance(frame, InputAudioRawFrame):
                out.append(int(np.abs(np.frombuffer(frame.audio, dtype=np.int16)).max()))
        g.push_frame = push
        await g.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        loud = (np.ones(640, dtype=np.int16) * 4000).tobytes()  # 40 ms
        for _ in range(loud_frames):
            await g.process_frame(InputAudioRawFrame(audio=loud, sample_rate=16000, num_channels=1), FrameDirection.DOWNSTREAM)
        return out

    async def test_short_noise_is_silenced_while_agent_talks(self):
        out = await self.run_guard(3)  # 120 ms "hmm"
        self.assertTrue(all(v == 0 for v in out))

    async def test_sustained_speech_interrupts(self):
        out = await self.run_guard(12)  # 480 ms of real speech
        self.assertTrue(any(v > 0 for v in out))

    async def test_echo_guarded_until_phone_finishes_playing(self):
        # Pipecat sends audio at 2x speed: "bot stopped" arrives while the phone still plays the reply.
        import numpy as np
        from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
        ser = main.VobizSerializer("S", lambda r: None)
        await ser.serialize(OutputAudioRawFrame(audio=b"\x00\x00" * 16000, sample_rate=16000, num_channels=1))  # 1 s
        g = main.BargeInGuard(clock=ser)
        out = []

        async def push(frame, direction=FrameDirection.DOWNSTREAM):
            if isinstance(frame, InputAudioRawFrame):
                out.append(int(np.abs(np.frombuffer(frame.audio, dtype=np.int16)).max()))
        g.push_frame = push
        await g.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        await g.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        echo = (np.ones(640, dtype=np.int16) * 4000).tobytes()
        for _ in range(3):  # 120 ms of the agent's own echo, still playing on the phone
            await g.process_frame(InputAudioRawFrame(audio=echo, sample_rate=16000, num_channels=1), FrameDirection.DOWNSTREAM)
        self.assertTrue(all(v == 0 for v in out))
        ser.play_until = 0.0  # playback finished
        await g.process_frame(InputAudioRawFrame(audio=echo, sample_rate=16000, num_channels=1), FrameDirection.DOWNSTREAM)
        self.assertGreater(out[-1], 0)

    def test_hangup_waits_for_playback(self):
        ser = main.VobizSerializer("S", lambda r: None)
        ser.play_until = time.monotonic() + 2.0
        self.assertGreater(ser.remaining(), 1.5)


class SavedAgentStart(unittest.TestCase):
    def test_second_call_starts_without_waiting_for_voiceos(self):
        import json
        from fastapi.testclient import TestClient
        aid = "11111111-2222-3333-4444-555555555555"
        agent = {"name": "a", "instructions": "x", "opening": "", "language": "English", "llm": "google/x",
                 "stt": "sarvam", "tts": "sarvam", "voice": "ishita", "tools": []}
        main._SESSION_CACHE[aid] = {"agent": agent, "keys": {"gemini": "k"}}
        started = []

        async def slow_voiceos(path, payload):
            if path == "session":
                await asyncio.sleep(1.5)
                return {"agent": agent, "keys": {"gemini": "k"}}
            return {}

        async def fake_pick(key):
            started.append(time.monotonic())
            return "gemini-3.8-live"

        async def ok(key, model):
            return True

        async def lookup(call_id, token):
            return aid

        async def no_connect(self, *a, **k):
            return None
        from unittest.mock import patch
        with patch.object(main, "voiceos", slow_voiceos), patch.object(main, "pick_live_model", fake_pick), \
                patch.object(main, "thinking_supported", ok), patch.object(main, "lookup_call_agent", lookup), \
                patch("pipecat.services.google.gemini_live.llm.GeminiLiveLLMService._connect", no_connect):
            t0 = time.monotonic()
            with TestClient(main.app).websocket_connect("/phone?call=c&token=t") as ws:
                ws.send_text(json.dumps({"event": "start", "start": {"streamId": "S"}}))
                time.sleep(0.6)
                ws.send_text(json.dumps({"event": "stop"}))
        self.assertTrue(started and started[0] - t0 < 1.0)
