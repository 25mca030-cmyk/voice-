"""Bulbul v4 Flash HTTP adapter for Pipecat versions predating v4 support."""
import io
import json
import wave
from pathlib import Path

import httpx
from pipecat.services.tts_service import TTSService
from pipecat.frames.frames import ErrorFrame, TTSStartedFrame, TTSStoppedFrame, TTSAudioRawFrame

PERSONAS = {v["speaker"]: v for v in json.loads(Path(__file__).with_name("sarvam-v4-voices.json").read_text())}


class SarvamFlashTTSService(TTSService):
    def __init__(self, *, api_key, voice_id, language, **kwargs):
        if voice_id not in PERSONAS:
            raise ValueError("Unknown Sarvam v4 Flash persona")
        super().__init__(sample_rate=24000, **kwargs)
        self._api_key = api_key
        self._speaker = voice_id
        self._language = language
        self.set_model_name("bulbul:v4-flash")

    async def run_tts(self, text):
        yield TTSStartedFrame()
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
                response = await client.post(
                    "https://api.sarvam.ai/text-to-speech",
                    headers={"api-subscription-key": self._api_key},
                    json={"text": text[:2500], "speaker": self._speaker, "model": "bulbul:v4-flash",
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
                        if wav.getsampwidth() != 2:
                            raise RuntimeError("Unsupported Sarvam audio format")
                        rate, channels = wav.getframerate(), wav.getnchannels()
                        while pcm := wav.readframes(4096):
                            yield TTSAudioRawFrame(pcm, rate, channels)
        except Exception as error:
            yield ErrorFrame(str(error))
        finally:
            yield TTSStoppedFrame()