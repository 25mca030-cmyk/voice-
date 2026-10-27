"""Sarvam services tuned for live phone calls.

SafeSarvamSTT
    Pipecat flushes Sarvam on every "user stopped speaking". A flush with no audio since the last one
    makes Sarvam close the socket ("Cannot flush: no audio input has been received"), after which every
    audio send failed for the rest of the call: the agent never heard the caller again (27 Oct, call
    39291213). This wrapper skips empty flushes and reconnects if the socket drops anyway.

StreamingSarvamTTS
    Sarvam's WebSocket TTS streams audio while it is generated, so the first words play in a few
    hundred ms instead of waiting for a whole HTTP response per sentence. That HTTP wait is what left
    silent gaps between sentences (the "breaking" voice). Phone calls get 16 kHz audio directly.
"""
import time

from loguru import logger
from pipecat.frames.frames import ErrorFrame, VADUserStoppedSpeakingFrame
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService, SarvamTTSSpeakerV3
from pipecat.services.stt_service import STTService

V3_SPEAKERS = {s.value for s in SarvamTTSSpeakerV3}


class SafeSarvamSTT(SarvamSTTService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._audio_since_flush = False
        self._last_reconnect = 0.0

    async def _reconnect(self, why: str):
        if time.monotonic() - self._last_reconnect < 1.0:
            return
        self._last_reconnect = time.monotonic()
        logger.warning(f"Sarvam STT reconnecting ({why[:100]})")
        try:
            await self._disconnect()
        except Exception:
            pass
        await self._connect()
        self._audio_since_flush = False

    async def process_frame(self, frame, direction):
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            if not self._audio_since_flush:
                # Nothing to finalise: skip Sarvam's flush, which would close the socket.
                await STTService.process_frame(self, frame, direction)
                return
            self._audio_since_flush = False
        await super().process_frame(frame, direction)

    async def run_stt(self, audio: bytes):
        if not self._socket_client:
            await self._reconnect("socket missing")
        self._audio_since_flush = True
        async for frame in super().run_stt(audio):
            if isinstance(frame, ErrorFrame):
                await self._reconnect(str(frame.error))
                continue  # one dropped 20 ms chunk is inaudible; a dead socket is not
            yield frame


def make_streaming_tts(*, api_key: str, voice: str, language: str, sample_rate: int):
    """Streaming WebSocket TTS for bulbul:v3 speakers; None when the voice needs the HTTP adapter."""
    if voice not in V3_SPEAKERS:
        return None
    return SarvamTTSService(
        api_key=api_key,
        sample_rate=sample_rate,
        settings=SarvamTTSService.Settings(
            model="bulbul:v3", voice=voice, language=language,
            # Start speaking after ~a short clause instead of a long buffer; chunks stay natural.
            min_buffer_size=30, max_chunk_length=150,
        ),
    )
