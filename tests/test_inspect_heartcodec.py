"""Protect header-only network reads and independent checkpoint byte accounting."""

import importlib.util
import io
import json
from pathlib import Path
import struct
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "inspect_heartcodec", Path(__file__).resolve().parents[1] / "tools/inspect-heartcodec.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class Response(io.BytesIO):
    def __init__(self, body, status=206, content_range="bytes 0-7/16"):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Range": content_range}


class HeaderAuditTests(unittest.TestCase):
    def test_refuses_full_download_when_range_is_ignored(self):
        response = Response(b"not read", status=200)
        with patch.object(audit.urllib.request, "urlopen", return_value=response), \
                patch.object(response, "read", wraps=response.read) as read:
            with self.assertRaisesRegex(ValueError, "did not honor Range"):
                audit.read_range("https://example.test/model", 0, 8)
            read.assert_not_called()

    def test_refuses_wrong_range_and_truncated_data(self):
        for body, content_range in [(b"12345678", "bytes 8-15/16"),
                                    (b"1234", "bytes 0-7/16")]:
            with self.subTest(content_range=content_range, body=body):
                with patch.object(audit.urllib.request, "urlopen",
                                  return_value=Response(body, content_range=content_range)):
                    with self.assertRaises(ValueError):
                        audit.read_range("https://example.test/model", 0, 8)

    def test_header_reads_stop_before_payload(self):
        header = {"weight": {"dtype": "F32", "shape": [256], "data_offsets": [0, 1024]}}
        raw = json.dumps(header).encode()
        prefix = struct.pack("<Q", len(raw))
        total = 8 + len(raw) + 1024
        responses = [Response(prefix, content_range=f"bytes 0-7/{total}"),
                     Response(prefix + raw, content_range=f"bytes 0-{7 + len(raw)}/{total}")]
        with patch.object(audit.urllib.request, "urlopen", side_effect=responses) as opened:
            actual, meta = audit.read_header("https://example.test/model")
        self.assertEqual(actual, header)
        self.assertEqual(opened.call_count, 2)
        self.assertEqual(meta["bytes_read"], 16 + len(raw))
        self.assertEqual(meta["file_bytes"], total)

    def test_rejects_oversized_header_before_second_request(self):
        prefix = struct.pack("<Q", audit.MAX_HEADER_BYTES + 1)
        with patch.object(audit.urllib.request, "urlopen", return_value=Response(
                prefix, content_range=f"bytes 0-7/{2 * audit.MAX_HEADER_BYTES}")) as opened:
            with self.assertRaisesRegex(ValueError, "header size"):
                audit.read_header("https://example.test/model")
        self.assertEqual(opened.call_count, 1)

    def test_quant_storage_includes_block_scales_and_keeps_convolution_f32(self):
        linear = audit.tensor_sizes("projection.weight", {
            "dtype": "BF16", "shape": [2, 256], "data_offsets": [0, 1024]}, 1024)
        self.assertEqual(linear["f32_bytes"], 2048)
        self.assertEqual(linear["rank2_q4_k_rest_f32_bytes"], 288)
        self.assertEqual(linear["rank2_q6_k_rest_f32_bytes"], 420)
        convolution = audit.tensor_sizes("conv.weight", {
            "dtype": "F32", "shape": [2, 256, 3], "data_offsets": [0, 6144]}, 6144)
        self.assertEqual(convolution["rank2_q4_k_rest_f32_bytes"], 6144)

    def test_rejects_invalid_tensor_storage(self):
        for dtype, shape, offsets in [("F32", [256], [0, 512]),
                                      ("F32", [-1], [0, 4]),
                                      ("F32", [1], [0, 2048]),
                                      ("UNKNOWN", [1], [0, 4])]:
            with self.subTest(dtype=dtype, shape=shape, offsets=offsets):
                with self.assertRaises(ValueError):
                    audit.tensor_sizes("weight", {
                        "dtype": dtype, "shape": shape, "data_offsets": offsets}, 1024)


if __name__ == "__main__":
    unittest.main()
