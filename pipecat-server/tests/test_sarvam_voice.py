import os
import unittest

os.environ.setdefault("PIPECAT_SHARED_SECRET", "test-only")
import main  # noqa: E402
import sarvam_live  # noqa: E402
from pipecat.frames.frames import (ErrorFrame, LLMFullResponseEndFrame, LLMTextFrame, TTSAudioRawFrame,  # noqa: E402
                                   TTSStartedFrame, TTSTextFrame)
from pipecat.processors.frame_processor import FrameDirection  # noqa: E402

DOWN, UP = FrameDirection.DOWNSTREAM, FrameDirection.UPSTREAM


class Languages(unittest.TestCase):
    def test_script_picks_language(self):
        self.assertEqual(sarvam_live.language_for_text("नमस्ते, कैसे हैं आप?", "en-IN"), "hi-IN")
        self.assertEqual(sarvam_live.language_for_text("કેમ છો?", "en-IN"), "gu-IN")
        self.assertEqual(sarvam_live.language_for_text("வணக்கம்", "en-IN"), "ta-IN")
        self.assertEqual(sarvam_live.language_for_text("Hello, how can I help?", "en-IN"), "en-IN")
        self.assertEqual(sarvam_live.language_for_text("Aap kaise ho?", "hi-IN"), "hi-IN")
        self.assertEqual(sarvam_live.language_for_text("नमस्कार", "mr-IN"), "mr-IN")

    def test_voice_factory(self):
        self.assertIsNone(sarvam_live.make_live_sarvam_voice(api_key="", voice="ishita", agent_language="Hindi",
                                                             sample_rate=16000))
        tts = sarvam_live.make_live_sarvam_voice(api_key="test-only", voice="Zephyr", agent_language="Gujarati",
                                                 sample_rate=16000)
        self.assertEqual(tts.speaker, "ishita")  # Gemini voice name -> default Sarvam voice
        self.assertEqual(tts._default_language, "gu-IN")
        some = sorted(sarvam_live.V3_SPEAKERS)[0]
        tts = sarvam_live.make_live_sarvam_voice(api_key="test-only", voice=f"sarvam:{some}", agent_language=None,
                                                 sample_rate=16000)
        self.assertEqual(tts.speaker, some)


class AudioDrop(unittest.IsolatedAsyncioTestCase):
    def make(self):
        sarvam = object()
        drop = main.GeminiAudioDrop(sarvam)
        out = []

        async def push(frame, direction=DOWN):
            out.append(frame)
        drop.push_frame = push
        return drop, sarvam, out

    async def test_gemini_audio_dropped_text_passed(self):
        drop, _, out = self.make()
        frames = [TTSStartedFrame(), TTSAudioRawFrame(b"\0" * 480, 24000, 1), TTSTextFrame("Hi", aggregated_by="sentence"),
                  LLMTextFrame("Hi there."), LLMFullResponseEndFrame()]
        for f in frames:
            await drop.process_frame(f, DOWN)
        self.assertEqual([type(f) for f in out], [LLMTextFrame, LLMFullResponseEndFrame])

    async def test_sarvam_error_falls_back_to_gemini_voice(self):
        drop, sarvam, out = self.make()
        err = ErrorFrame(error="quota")
        err.processor = sarvam
        await drop.process_frame(err, UP)
        self.assertTrue(drop.fallback)
        out.clear()
        audio = TTSAudioRawFrame(b"\0" * 480, 24000, 1)
        await drop.process_frame(audio, DOWN)
        await drop.process_frame(LLMTextFrame("Hello"), DOWN)
        self.assertEqual(out, [audio])

    async def test_other_errors_pass(self):
        drop, _, out = self.make()
        err = ErrorFrame(error="other")
        await drop.process_frame(err, UP)
        self.assertFalse(drop.fallback)
        self.assertEqual(out, [err])


if __name__ == "__main__":
    unittest.main()
