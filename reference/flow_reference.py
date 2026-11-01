"""Isolated, strict loading of the pinned HeartCodec RVQ/flow implementation."""
import json
from pathlib import Path
import sys
import types
import torch
from safetensors import safe_open
from sq_reference import load_module

FLOW_KEYS = ("dim", "codebook_size", "decay", "commitment_weight", "threshold_ema_dead_code",
             "use_cosine_sim", "codebook_dim", "num_quantizers", "attention_head_dim", "in_channels",
             "norm_type", "num_attention_heads", "num_layers", "num_layers_2", "out_channels")


def load_flow(heartlib, snapshot):
    snapshot = Path(snapshot)
    config = json.loads((snapshot / "config.json").read_text())
    package = types.ModuleType("heartcodec_flow_reference")
    package.__path__ = [str(Path(heartlib) / "src/heartlib/heartcodec/models")]
    sys.modules[package.__name__] = package
    module = load_module(package.__name__ + ".flow_matching", Path(package.__path__[0]) / "flow_matching.py")
    # RVQ also owns nonpersistent buffers. Construct them on CPU so a strict
    # state load does not leave meta buffers that fail at model.cuda().
    model = module.FlowMatching(**{key: config[key] for key in FLOW_KEYS})
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    for shard in sorted({value for name, value in index.items() if name.startswith("flow_matching.")}):
        with safe_open(snapshot / shard, framework="pt", device="cpu") as source:
            for name in source.keys():
                if name.startswith("flow_matching."):
                    state[name.removeprefix("flow_matching.")] = source.get_tensor(name)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().requires_grad_(False), config


def get_snapshot(source_file):
    return Path(json.loads(Path(source_file).read_text())["snapshot"])
