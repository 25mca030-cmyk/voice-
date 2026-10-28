"""Runs the phone-call processors inside a real Pipecat pipeline (setup, clocks, tasks) and checks that
no processor raises. Unit tests that call process_frame directly missed a crash on 28 Oct that only
happened once Pipecat had set the processors up (agent silent after the greeting)."""
import asyncio
import os
import unittest

import numpy as np

os.environ.setdefault("PIPECAT_SHARED_SECRET", "test-only")
import main  # noqa: E402
from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, EndFrame, ErrorFrame,  # noqa: E402
                                   InputAudioRawFrame, LLMFullResponseEndFrame, LLMTextFrame, OutputAudioRawFrame,
                                   TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineParams, PipelineTask  # noqa: E402
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402


class ErrorCatcher(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.errors = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, ErrorFrame):
            self.errors.append(str(frame.error))
        await self.push_frame(frame, direction)


class Sink(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.seen = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.seen.append(type(frame).__name__)
        if isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            return
        await self.push_frame(frame, direction)


class PhonePipeline(unittest.IsolatedAsyncioTestCase):
    async def test_processors_run_without_errors(self):
        ser = main.VobizSerializer("S", lambda r: None)
        catcher, sink = ErrorCatcher(), Sink()
        stats = {"transcript": [], "interruptions": 0, "latencies": [], "input_tokens": 0, "output_tokens": 0}
        greeter = main.GreetingPlayer(None, None, "")
        pipeline = Pipeline([
            catcher,
            main.OpeningGuard(max_secs=0.2),
            main.VoiceClock(stats),
            main.BargeInGuard(clock=ser),
            main.Tracker(stats, "input", False),
            main.GeminiAudioDrop(object()),
            greeter,
            main.Tracker(stats, "assistant", False),
            sink,
        ])
        task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=False), enable_rtvi=False)
        loud = (np.ones(640, dtype=np.int16) * 3000).tobytes()

        async def drive():
            await asyncio.sleep(0.1)
            await ser.serialize(OutputAudioRawFrame(audio=b"\x00\x00" * 16000, sample_rate=16000, num_channels=1))
            await task.queue_frames([BotStartedSpeakingFrame(), TTSStartedFrame(),
                                     TTSAudioRawFrame(b"\x00" * 960, 24000, 1), LLMTextFrame("Hello there."),
                                     TTSStoppedFrame(), LLMFullResponseEndFrame(), BotStoppedSpeakingFrame()])
            for _ in range(30):  # caller audio while the reply still plays, then after it
                await task.queue_frame(InputAudioRawFrame(audio=loud, sample_rate=16000, num_channels=1))
            ser.play_until = 0.0
            for _ in range(10):
                await task.queue_frame(InputAudioRawFrame(audio=loud, sample_rate=16000, num_channels=1))
            await asyncio.sleep(1.2)  # let the late-text timer fire
            await task.queue_frame(EndFrame())

        await asyncio.gather(PipelineRunner(handle_sigint=False).run(task), drive())
        self.assertEqual(catcher.errors, [])
        self.assertIn("InputAudioRawFrame", sink.seen)
        self.assertIn("LLMTextFrame", sink.seen)


if __name__ == "__main__":
    unittest.main()
