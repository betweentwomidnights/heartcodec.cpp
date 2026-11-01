"""Integration contracts for the native GGUF quantizer, using tiny synthetic weights."""
import json
import gc
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class QuantizationToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = os.environ.get("HEARTCODEC_QUANTIZER")
        if not cls.tool:
            raise unittest.SkipTest("Set HEARTCODEC_QUANTIZER to the built native tool")
        import gguf
        import numpy
        cls.gguf, cls.np = gguf, numpy

    def test_block_alignment_mixed_types_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.gguf"
            writer = self.gguf.GGUFWriter(str(source), "heartcodec-sq")
            writer.add_uint32("heartcodec.sq.format_version", 2)
            writer.add_string("heartcodec.sq.weight_layout", "matmul")
            names = ("sq.encoder.2.convs.0.conv1.weight", "sq.encoder.2.convs.0.conv2.weight",
                     "sq.decoder.1.convs.0.conv1.weight", "sq.encoder.0.weight", "sq.encoder.3.short.weight")
            for name in names:
                columns = 32 if ".short." in name else 256
                writer.add_tensor(name, self.np.linspace(-0.4, 0.4, 2 * columns, dtype=self.np.float32).reshape(2, columns))
            writer.add_tensor("sq.encoder.2.convs.0.conv1.bias", self.np.zeros(2, dtype=self.np.float32))
            writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
            output = root / "quantized.gguf"
            subprocess.run([self.tool, "--input", str(source), "--output", str(output),
                            "--mix", "q4_k_m", "--scope", "encoder"], check=True)
            reader = self.gguf.GGUFReader(str(output))
            types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            self.assertEqual(types[names[0]], self.gguf.GGMLQuantizationType.Q4_K)
            self.assertEqual(types[names[1]], self.gguf.GGMLQuantizationType.Q6_K)
            for name in names[2:]:
                self.assertEqual(types[name], self.gguf.GGMLQuantizationType.F32)
            del reader
            gc.collect() # Release GGUFReader's mmap before Windows removes the fixture.
            self.assertEqual(json.loads(Path(str(output) + ".quantization.json").read_text())["scope"], "encoder")
            requantized = root / "requantized.gguf"
            result = subprocess.run([self.tool, "--input", str(output), "--output", str(requantized),
                                     "--mix", "q4_k_m"], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(requantized.exists())


if __name__ == "__main__":
    unittest.main()
