import os
import unittest

os.environ.setdefault("PIPECAT_SHARED_SECRET", "test-only")
import main  # noqa: E402
import sarvam_live  # noqa: E402
import asyncio  # noqa: E402
from pipecat.frames.frames import (ErrorFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame,  # noqa: E402
                                   LLMTextFrame, TTSAudioRawFrame, TTSStartedFrame, TTSTextFrame)
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


class Opening(unittest.TestCase):
    def test_company_filled(self):
        agent = {"opening": "Hi, this is sam from [Company]. Do you have a moment?",
                 "instructions": "x\n\nYou work for Marketing ai. When the caller asks your company name..."}
        self.assertEqual(main.speakable_opening(agent), "Hi, this is sam from Marketing ai. Do you have a moment?")

    def test_unfilled_placeholder_left_to_gemini(self):
        self.assertEqual(main.speakable_opening({"opening": "Hi from [Company]", "instructions": ""}), "")
        self.assertEqual(main.speakable_opening({"opening": "", "instructions": ""}), "")


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
        frames = [TTSStartedFrame(), LLMFullResponseStartFrame(), TTSAudioRawFrame(b"\0" * 480, 24000, 1),
                  TTSTextFrame("Hi", aggregated_by="sentence"), LLMTextFrame("Hi there."), LLMFullResponseEndFrame()]
        for f in frames:
            await drop.process_frame(f, DOWN)
        self.assertEqual([type(f) for f in out], [LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame])

    async def test_late_transcript_is_spoken_now_not_next_turn(self):
        drop, _, out = self.make()
        await drop.process_frame(LLMFullResponseEndFrame(), DOWN)  # Gemini says the turn is over...
        await drop.process_frame(LLMTextFrame("and one more thing."), DOWN)  # ...then late words arrive
        await asyncio.sleep(drop.IDLE_FLUSH_S + 0.15)
        self.assertEqual([type(f) for f in out], [LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame])

    async def test_interruption_resets(self):
        drop, _, out = self.make()
        await drop.process_frame(LLMTextFrame("Hello"), DOWN)
        await drop.process_frame(main._Interrupt(), DOWN)
        await asyncio.sleep(drop.IDLE_FLUSH_S + 0.15)
        self.assertNotIn(LLMFullResponseEndFrame, [type(f) for f in out])

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
