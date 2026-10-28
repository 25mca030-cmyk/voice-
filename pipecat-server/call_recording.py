"""Phone call recordings for the VoiceOS Recordings page.

CallRecorder taps the Vobiz stream itself: caller audio as it arrives and agent audio as it is actually
sent (paced in real time), both 16 kHz, so the two sides line up as the caller heard them.

save_recording uploads a WAV to the VoiceOS "recordings" bucket at <user>/<call>.wav and links it to the
call. It runs only when RECORDINGS=1 and the VoiceOS database has the pipecat_recording_* functions;
each step is checked against the call's one-time session token.
"""
import io
import os
import time
import wave

import numpy as np
from loguru import logger

RATE = 16000
MAX_SECONDS = 30 * 60


def level(x: np.ndarray, target_peak: float, max_gain: float) -> np.ndarray:
    """Scales one side of the call so its speech peaks near target_peak (gain capped at max_gain)."""
    if x.size == 0:
        return x
    a = np.abs(x)
    loud = a[a > 200]  # ignore silence and line hiss when measuring the speaking level
    if loud.size < RATE // 10:
        return x
    peak = float(np.percentile(loud, 99))
    gain = min(max_gain, target_peak / max(peak, 1.0))
    return x * gain


class CallRecorder:
    def __init__(self):
        self._user = bytearray()
        self._bot: list[tuple[float, bytes]] = []
        self._t0 = None

    def add_user(self, pcm: bytes):
        if self._t0 is None:
            self._t0 = time.monotonic() - len(self._user) / 2 / RATE
        if len(self._user) < MAX_SECONDS * RATE * 2:
            self._user.extend(pcm)

    def add_bot(self, pcm: bytes, rate: int):
        if rate != RATE or not pcm:
            return
        if self._t0 is None:
            self._t0 = time.monotonic()
        self._bot.append((time.monotonic(), bytes(pcm)))

    def wav(self) -> bytes | None:
        if self._t0 is None:
            return None
        user = np.frombuffer(bytes(self._user), dtype=np.int16).astype(np.int32)
        end = len(user)
        for t, pcm in self._bot:
            end = max(end, int((t - self._t0) * RATE) + len(pcm) // 2)
        end = min(end, MAX_SECONDS * RATE)
        if end < RATE:  # under a second: nothing worth keeping
            return None
        bot = np.zeros(end, dtype=np.float32)
        for t, pcm in self._bot:
            start = max(0, int((t - self._t0) * RATE))
            chunk = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)[: max(0, end - start)]
            bot[start:start + len(chunk)] += chunk
        caller = np.zeros(end, dtype=np.float32)
        caller[:min(len(user), end)] = user[:end]
        # Phone audio from the caller arrives far quieter than the agent's voice, so in a plain mix the
        # customer could not be heard (28 Oct). Bring both sides to the same speaking level first.
        mix = level(caller, 9000.0, 12.0) + level(bot, 9000.0, 2.0)
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(np.clip(np.tanh(mix / 32768.0 * 1.1) * 32767.0, -32768, 32767).astype(np.int16).tobytes())
        return out.getvalue()


async def save_recording(client, call_id: str, token: str, wav: bytes) -> bool:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "")
    if os.environ.get("RECORDINGS") != "1" or not url or not key or not wav:
        return False
    headers = {"apikey": key}
    try:
        r = await client.post(f"{url}/rest/v1/rpc/pipecat_recording_path", headers=headers,
                              json={"p_call": call_id, "p_token": token}, timeout=15)
        path = r.json() if r.status_code == 200 else None
        if not path:
            logger.warning(f"recording not saved: no upload slot ({r.status_code})")
            return False
        up = await client.post(f"{url}/storage/v1/object/recordings/{path}", content=wav, timeout=60,
                               headers={**headers, "content-type": "audio/wav", "x-upsert": "false"})
        if up.status_code >= 300:
            logger.warning(f"recording upload failed ({up.status_code})")
            return False
        a = await client.post(f"{url}/rest/v1/rpc/pipecat_attach_recording", headers=headers,
                              json={"p_call": call_id, "p_token": token}, timeout=15)
        ok = a.status_code == 200 and a.json() is True
        logger.info(f"recording saved ({len(wav) / 32000:.0f} s)" if ok else "recording uploaded but not linked")
        return ok
    except Exception as e:
        logger.warning(f"recording not saved: {type(e).__name__}")
        return False
