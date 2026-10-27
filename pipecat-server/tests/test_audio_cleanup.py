import asyncio
import os
import unittest

import numpy as np

os.environ.setdefault("PIPECAT_SHARED_SECRET", "test-only")
from audio_cleanup import NoiseGate, make_noise_filter  # noqa: E402


async def run_gate(noise_std, hum):
    g = NoiseGate()
    await g.start(16000)
    rng = np.random.default_rng(1)
    t = np.arange(16000 * 6) / 16000
    talking = np.sin(2 * np.pi * 0.5 * t) > 0
    voice = (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 900 * t)) * 3000 * talking
    sig = (voice + rng.normal(0, noise_std, t.size) + hum * np.sin(2 * np.pi * 50 * t)).astype(np.int16)
    out = b"".join([await g.filter(sig[i:i + 320].tobytes()) for i in range(0, sig.size, 320)])
    o = np.frombuffer(out, dtype=np.int16).astype(float)
    quiet, speech = ~talking, talking.copy()
    quiet[:16000] = speech[:16000] = False
    return sig[quiet].std(), o[quiet].std(), voice[speech].std(), o[speech].std()


class NoiseGateTests(unittest.TestCase):
    def test_hum_and_hiss_reduced_voice_kept(self):
        noise_in, noise_out, voice_in, voice_out = asyncio.run(run_gate(300, 2000))
        self.assertLess(noise_out, noise_in * 0.25)
        self.assertGreater(voice_out, voice_in * 0.8)

    def test_output_length_matches(self):
        async def go():
            g = NoiseGate()
            await g.start(16000)
            return await g.filter(b"\x01\x00" * 320)
        self.assertEqual(len(asyncio.run(go())), 640)

    def test_modes(self):
        os.environ["NOISE_FILTER"] = "off"
        self.assertIsNone(make_noise_filter())
        os.environ["NOISE_FILTER"] = "gate"
        self.assertIsInstance(make_noise_filter(), NoiseGate)


if __name__ == "__main__":
    unittest.main()
