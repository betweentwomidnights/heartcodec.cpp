#!/usr/bin/env python3
"""Strict Base SQ conversion; fold weight norm and store quantizable matrix rows."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT / ".python-deps"))
from sq_reference import load_heartlib, revision
from gguf import GGUFWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heartlib", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part", choices=("encoder", "decoder", "codec"), default="decoder")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = load_heartlib(args.heartlib, args.weights, args.config)
    config = json.loads(args.config.read_text())
    required = {"init_channel": 64, "latent_hidden_dim": 128, "num_samples": 2,
                "num_bands": 1, "delay_kernel_size": 5, "default_kernel_size": 7,
                "res_kernel_size": 7, "upsample_factors": [5, 4, 4, 4, 3],
                "upsample_kernel_sizes": [10, 8, 8, 8, 6],
                "downsample_factors": [3, 4, 4, 4, 5], "downsample_kernel_sizes": [6, 8, 8, 8, 10]}
    if any(config[key] != value for key, value in required.items()):
        raise ValueError("Unsupported SQ decoder configuration")
    tensors, covered = {}, set()
    parts = ("encoder", "decoder") if args.part == "codec" else (args.part,)
    for name, module in model.named_modules():
        if not any(name.startswith(part + ".") for part in parts):
            continue
        prefix = "sq." + name
        if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            if module.groups != 1:
                raise ValueError(f"Grouped convolution is unsupported: {name}")
            # Accessing weight evaluates PyTorch's actual parametrization, including
            # the correct axis for ConvTranspose1d. No guessed g/v reduction axes.
            weight = module.weight.detach().float().cpu().numpy()
            if isinstance(module, torch.nn.ConvTranspose1d):
                # Torch [IC, OC, K] -> GGML [IC, OC*K] (numpy reversed dimensions).
                weight = weight.transpose(1, 2, 0).reshape(-1, weight.shape[0])
            else:
                # Torch [OC, IC, K] -> GGML [IC*K, OC], keeping the dot-product row contiguous.
                weight = weight.reshape(weight.shape[0], -1)
            tensors[prefix + ".weight"] = np.ascontiguousarray(weight)
            if module.bias is None:
                raise ValueError(f"Expected convolution bias: {name}")
            tensors[prefix + ".bias"] = module.bias.detach().cpu().numpy().copy()
            covered.update(name + "." + key for key in module.state_dict())
        elif isinstance(module, torch.nn.PReLU):
            if module.num_parameters != 1:
                raise ValueError(f"Expected scalar PReLU: {name}")
            tensors[prefix + ".weight"] = module.weight.detach().cpu().numpy().copy()
            covered.add(name + ".weight")
    expected = {name for name in model.state_dict() if any(name.startswith(part + ".") for part in parts)}
    if covered != expected:
        raise ValueError(f"Unconverted or unexpected SQ tensors: {expected.symmetric_difference(covered)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(args.output), "heartcodec-sq")
    writer.add_uint32("heartcodec.sq.format_version", 2)
    writer.add_string("heartcodec.sq.weight_layout", "matmul")
    writer.add_string("heartcodec.sq.part", args.part)
    writer.add_uint32("heartcodec.sq.sample_rate", 48000)
    writer.add_uint32("heartcodec.sq.latent_dim", 128)
    writer.add_uint32("heartcodec.sq.samples_per_frame", 1920)
    writer.add_string("heartcodec.sq.source_revision", revision(args.heartlib))
    writer.add_string("heartcodec.sq.weight_precision", "F32")
    for name, tensor in tensors.items():
        writer.add_tensor(name, tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    report = {"tensor_count": len(tensors), "dtype": "float32", "part": args.part,
              "weight_layout": "matmul", "format_version": 2,
              "weight_norm": "evaluated by upstream PyTorch parametrization",
              "heartlib_revision": revision(args.heartlib),
              "tensors": {name: list(tensor.shape) for name, tensor in tensors.items()}}
    args.output.with_suffix(".conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}: {len(tensors)} F32 tensors", flush=True)


if __name__ == "__main__":
    main()
