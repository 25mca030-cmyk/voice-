"""Versioned Sarvam HTTP speech adapter; avoids SDK WebSocket model drift."""
import io
import json
import wave
from pathlib import Path

import httpx
from pipecat.services.tts_service import TTSService
from pipecat.frames.frames import ErrorFrame, TTSStartedFrame, TTSStoppedFrame, TTSAudioRawFrame

PERSONAS = {v["speaker"]: v for v in json.loads(Path(__file__).with_name("sarvam-v4-voices.json").read_text())}
LEGACY_VOICES = {"anushka", "abhilash", "manisha", "vidya", "arya", "karun", "hitesh"}


class SarvamFlashTTSService(TTSService):
    def __init__(self, *, api_key, voice_id, language, **kwargs):
        super().__init__(sample_rate=24000, push_text_frames=True, **kwargs)
        self._api_key = api_key
        self._speaker = voice_id
        self._language = language
        self._model = "bulbul:v4-flash" if voice_id in PERSONAS else "bulbul:v2" if voice_id in LEGACY_VOICES else "bulbul:v3"
        self.set_model_name(self._model)

    async def run_tts(self, text, context_id=None):
        # Pipecat >=0.0.105 passes a context ID; retain compatibility with older releases.
        frame_context = {"context_id": context_id} if context_id is not None else {}
        yield TTSStartedFrame(**frame_context)
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
                response = await client.post(
                    "https://api.sarvam.ai/text-to-speech",
                    headers={"api-subscription-key": self._api_key},
                    json={"text": text[:2500], "speaker": self._speaker, "model": self._model,
                          "target_language_code": self._language, "speech_sample_rate": 24000},
                )
                if response.status_code != 200:
                    raise RuntimeError(f"Sarvam speech failed ({response.status_code})")
                audios = response.json().get("audios", [])
                if not audios:
                    raise RuntimeError("Sarvam returned no audio")
                import base64
                for audio in audios:
                    with wave.open(io.BytesIO(base64.b64decode(audio)), "rb") as wav:
                        if wav.getsampwidth() != 2 or wav.getnchannels() != 1 or wav.getframerate() != 24000:
                            raise RuntimeError("Unsupported Sarvam audio format")
                        rate, channels = wav.getframerate(), wav.getnchannels()
                        while pcm := wav.readframes(4096):
                            yield TTSAudioRawFrame(pcm, rate, channels, **frame_context)
        except Exception as error:
            yield ErrorFrame(str(error))
        finally:
            yield TTSStoppedFrame(**frame_context)