#!/usr/bin/env python3
"""Export the strictly loaded HeartCodec decoder RVQ and flow estimator."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "reference"), str(ROOT / ".python-deps")]
from flow_reference import load_flow, get_snapshot
from sq_reference import revision
from gguf import GGUFWriter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heartlib", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    model, config = load_flow(a.heartlib, get_snapshot(a.source))
    required = dict(dim=512, codebook_size=8192, codebook_dim=32, num_quantizers=8,
                    attention_head_dim=64, in_channels=1024, norm_type="ada_norm_single",
                    num_attention_heads=24, num_layers=24, num_layers_2=6, out_channels=256)
    if any(config[k] != v for k, v in required.items()):
        raise ValueError("Unsupported flow configuration")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(a.output), "heartcodec-flow")
    writer.add_uint32("heartcodec.flow.format_version", 1)
    writer.add_string("heartcodec.flow.source_revision", revision(a.heartlib))
    writer.add_string("heartcodec.flow.checkpoint_revision", get_snapshot(a.source).name)
    shapes, omitted = {}, []
    for name, value in model.state_dict().items():
        if name.startswith("vq_embed.project_in.") or ("._codebook." in name and not name.endswith(".embed")):
            omitted.append(name)
            continue
        value = value.detach().float().cpu().numpy()
        if name.endswith("._codebook.embed"):
            value = value.squeeze(0)
        value = np.ascontiguousarray(value)
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite tensor: {name}")
        writer.add_tensor(name, value)
        shapes[name] = list(value.shape)
    if len(shapes) != 339:
        raise ValueError(f"Unexpected runtime tensor count: {len(shapes)}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    a.output.with_suffix(".conversion.json").write_text(json.dumps(dict(
        tensors=shapes, omitted_training_and_encoder_tensors=omitted,
        heartlib_revision=revision(a.heartlib), checkpoint_revision=get_snapshot(a.source).name,
        dtype="float32", rvq="sum eight 32-dimensional embeddings, then one shared projection"), indent=2) + "\n")
    print(f"Wrote {a.output}: {len(shapes)} F32 runtime tensors", flush=True)


if __name__ == "__main__":
    main()
