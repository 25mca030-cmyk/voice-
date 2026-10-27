"""Background-noise handling for phone calls.

NoiseGate (default, almost no CPU): removes low rumble/hum and turns steady background noise (fans,
traffic, TV murmur, line hiss) down between words, so the voice model does not hear it as speech and
cut itself off. Speech passes untouched, with a short hold so word endings are not clipped.

RNNoise (set NOISE_FILTER=rnnoise): neural noise removal, much stronger, but uses about 0.07 CPU per
call, too much for Render's free 0.15-CPU server. Turn it on after upgrading to the Starter plan.
"""
import os

import numpy as np
from loguru import logger
from pipecat.audio.filters.base_audio_filter import BaseAudioFilter


class NoiseGate(BaseAudioFilter):
    def __init__(self, *, open_ratio: float = 2.0, floor_gain: float = 0.2, hold_ms: int = 300,
                 highpass_hz: float = 120.0):
        self._open_ratio = open_ratio
        self._floor_gain = floor_gain
        self._hold_ms = hold_ms
        self._highpass_hz = highpass_hz
        self._rate = 16000
        self._floor = None
        self._gain = 1.0
        self._hold = 0.0
        self._sos = None
        self._zi = None

    async def start(self, sample_rate: int):
        from scipy.signal import butter, sosfilt_zi
        self._rate = sample_rate
        # 2nd-order high-pass: removes hum, rumble and handling noise below the voice range.
        self._sos = butter(2, self._highpass_hz, btype="highpass", fs=sample_rate, output="sos")
        self._zi = sosfilt_zi(self._sos) * 0.0

    async def stop(self):
        pass

    async def process_frame(self, frame):
        pass

    async def filter(self, audio: bytes) -> bytes:
        from scipy.signal import sosfilt
        x = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        if x.size == 0 or self._sos is None:
            return audio
        y, self._zi = sosfilt(self._sos, x, zi=self._zi)

        rms = float(np.sqrt(np.mean(y * y))) + 1e-6
        if self._floor is None:
            self._floor = rms
        # Noise floor follows quiet moments quickly and creeps up slowly during speech.
        self._floor = rms if rms < self._floor else self._floor * 1.002 + (rms - self._floor) * 0.0005
        frame_ms = 1000 * x.size / self._rate
        if rms > self._floor * self._open_ratio:
            self._hold = self._hold_ms
        else:
            self._hold = max(0.0, self._hold - frame_ms)
        target = 1.0 if self._hold > 0 else self._floor_gain
        ramp = np.linspace(self._gain, target, x.size, dtype=np.float32)  # no clicks between frames
        self._gain = target
        return np.clip(y * ramp, -32768, 32767).astype(np.int16).tobytes()


def make_noise_filter():
    mode = os.environ.get("NOISE_FILTER", "gate").lower()
    if mode == "rnnoise":
        try:
            from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
            logger.info("noise filter: RNNoise")
            return RNNoiseFilter()
        except Exception as e:  # missing package: keep calls working with the gate
            logger.warning(f"RNNoise unavailable ({e}); using the noise gate")
    if mode == "off":
        return None
    return NoiseGate()
