"""Aller-retour de l'encodage binaire côté Python : python -m unittest discover agent/tests"""

import unittest

import numpy as np

from openfront_ai.protocol import decode_body, encode_frame


class ProtocolTest(unittest.TestCase):
    def test_round_trip(self):
        buffers = {
            "agent_0/spatial": (np.arange(2 * 3 * 5) * 37 % 256).astype(np.uint8).reshape(2, 3, 5),
            "agent_0/players": np.array([[1.5, -2.25], [np.pi, 1e6]], dtype=np.float32),
            "ids": np.array([-7, 0, 123456], dtype=np.int32),
        }
        frame = encode_frame({"ok": True, "note": "é"}, buffers)
        self.assertEqual(int.from_bytes(frame[:4], "big"), len(frame) - 4)
        header, decoded = decode_body(frame[4:])
        self.assertEqual(header, {"ok": True, "note": "é"})
        for name, array in buffers.items():
            np.testing.assert_array_equal(decoded[name], array)
            self.assertEqual(decoded[name].dtype, array.dtype)

    def test_header_only(self):
        header, decoded = decode_body(encode_frame({"cmd": "step", "ticks": 10})[4:])
        self.assertEqual(header, {"cmd": "step", "ticks": 10})
        self.assertEqual(decoded, {})


if __name__ == "__main__":
    unittest.main()
