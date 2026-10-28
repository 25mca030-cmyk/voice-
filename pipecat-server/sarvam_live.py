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
import os
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


# ---------------------------------------------------------------- Sarvam voice for Gemini Live calls
# Gemini Live listens and thinks; Sarvam Bulbul speaks its replies. Each sentence is spoken in the
# language of its script (Devanagari -> Hindi, Gujarati script -> Gujarati, ...), so mixed-language
# calls sound natural. Latin-script sentences use the agent's own language (English or Hinglish).
_SCRIPTS = [
    (0x0900, 0x097F, "hi-IN"), (0x0980, 0x09FF, "bn-IN"), (0x0A00, 0x0A7F, "pa-IN"), (0x0A80, 0x0AFF, "gu-IN"),
    (0x0B00, 0x0B7F, "od-IN"), (0x0B80, 0x0BFF, "ta-IN"), (0x0C00, 0x0C7F, "te-IN"), (0x0C80, 0x0CFF, "kn-IN"),
    (0x0D00, 0x0D7F, "ml-IN"),
]
AGENT_LANGS = {
    "English": "en-IN", "English (India)": "en-IN", "Hindi": "hi-IN", "Gujarati": "gu-IN", "Marathi": "mr-IN",
    "Bengali": "bn-IN", "Tamil": "ta-IN", "Telugu": "te-IN", "Kannada": "kn-IN", "Malayalam": "ml-IN",
    "Punjabi": "pa-IN", "Odia": "od-IN",
}
DEFAULT_SARVAM_VOICE = os.environ.get("SARVAM_DEFAULT_VOICE", "ishita")


def language_for_text(text: str, default: str) -> str:
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi, code in _SCRIPTS:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return default
    code = max(counts, key=counts.get)
    # Marathi is written in Devanagari too: keep Marathi when that is the agent's language.
    return "mr-IN" if code == "hi-IN" and default == "mr-IN" else code


class SarvamLiveTTS(SarvamTTSService):
    """Streaming Bulbul v3 voice that switches language per sentence."""

    def __init__(self, *, default_language: str, **kwargs):
        super().__init__(**kwargs)
        self._default_language = default_language

    async def run_tts(self, text: str, context_id: str):
        lang = language_for_text(text, self._default_language)
        if lang != self._settings.language:
            self._settings.language = lang
            try:
                if self._websocket:
                    await self._send_config()
            except Exception as e:
                logger.warning(f"Sarvam language switch to {lang} failed: {type(e).__name__}")
        async for frame in super().run_tts(text, context_id):
            yield frame


def make_live_sarvam_voice(*, api_key: str, voice: str | None, agent_language: str | None, sample_rate: int):
    """Sarvam voice for a Gemini Live call, or None (no key) so Gemini keeps its own voice."""
    if not api_key:
        return None
    speaker = (voice or "").removeprefix("sarvam:")
    if speaker not in V3_SPEAKERS:
        speaker = DEFAULT_SARVAM_VOICE if DEFAULT_SARVAM_VOICE in V3_SPEAKERS else "ishita"
    default = AGENT_LANGS.get(agent_language or "English", "en-IN")
    tts = SarvamLiveTTS(
        api_key=api_key, sample_rate=sample_rate, default_language=default,
        settings=SarvamTTSService.Settings(model="bulbul:v3", voice=speaker, language=default,
                                           min_buffer_size=30, max_chunk_length=150),
    )
    tts.speaker = speaker
    logger.info(f"Sarvam voice: {speaker}, default language {default}")
    return tts
