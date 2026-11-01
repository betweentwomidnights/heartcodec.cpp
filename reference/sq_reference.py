"""Load isolated upstream SQ modules without importing unrelated model packages."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import torch
from safetensors.torch import load_file

SQ_KEYS = ("num_bands", "sample_rate", "causal", "num_samples", "downsample_factors",
           "downsample_kernel_sizes", "upsample_factors", "upsample_kernel_sizes",
           "latent_hidden_dim", "default_kernel_size", "delay_kernel_size",
           "init_channel", "res_kernel_size")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def revision(repo):
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def load_heartlib(repo, weights, config):
    module = load_module("heartcodec_reference_sq", Path(repo) / "src/heartlib/heartcodec/models/sq_codec.py")
    settings = json.loads(Path(config).read_text())
    if settings["sample_rate"] != 48000 or settings["latent_hidden_dim"] != 128 or not settings["causal"]:
        raise ValueError("Expected released 48 kHz causal Base SQ architecture")
    with torch.device("meta"):
        model = module.ScalarModel(**{key: settings[key] for key in SQ_KEYS})
    model.load_state_dict(load_file(str(weights)), strict=True, assign=True)
    return model.eval().requires_grad_(False)


def load_spansynth(repo, weights, config):
    root = Path(repo) / "spansynth/codec"
    package = types.ModuleType("spansynth_reference_codec")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    module = load_module(package.__name__ + ".sq_codec", root / "sq_codec.py")
    configuration = sys.modules[package.__name__ + ".config"]
    with torch.device("meta"):
        model = module.ScalarModel(configuration.load_config(config))
    model.load_state_dict(load_file(str(weights)), strict=True, assign=True)
    return model.eval().requires_grad_(False)


def configure_torch():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
