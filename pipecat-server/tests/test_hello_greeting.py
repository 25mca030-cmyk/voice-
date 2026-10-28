import asyncio
import base64
import json
import os
import unittest

import numpy as np

os.environ.setdefault("PIPECAT_SHARED_SECRET", "test-only")
import main  # noqa: E402


def packet(level):
    pcm = (np.random.default_rng(0).standard_normal(320) * level).astype(np.int16).tobytes()
    return {"type": "websocket.receive", "text": json.dumps(
        {"event": "media", "media": {"payload": base64.b64encode(pcm).decode(), "track": "inbound"}})}


class FakeWS:
    def __init__(self, packets, gap=0.02):
        self.packets, self.gap = list(packets), gap

    async def receive(self):
        await asyncio.sleep(self.gap)
        if not self.packets:
            await asyncio.sleep(10)
        return self.packets.pop(0)


class Hello(unittest.IsolatedAsyncioTestCase):
    async def test_greets_right_after_hello(self):
        pkts = [packet(30)] * 10 + [packet(3000)] * 25 + [packet(30)] * 30  # silence, 0.5 s "hello", silence
        t = asyncio.get_event_loop().time()
        self.assertEqual(await main.wait_for_hello(FakeWS(pkts)), "hello")
        # returns ~0.3 s after the hello ends, not after the 2.5 s timeout
        self.assertLess(asyncio.get_event_loop().time() - t, 1.6)

    async def test_silent_caller_still_greeted(self):
        old = main.HELLO_WAIT_S
        main.HELLO_WAIT_S = 0.4
        try:
            self.assertEqual(await main.wait_for_hello(FakeWS([packet(30)] * 100)), "silence")
        finally:
            main.HELLO_WAIT_S = old

    async def test_hangup_before_greeting(self):
        self.assertEqual(await main.wait_for_hello(FakeWS([{"type": "websocket.disconnect"}])), "closed")


if __name__ == "__main__":
    unittest.main()
