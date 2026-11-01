#!/usr/bin/env python3
"""Download the pinned full decoder, reusing cached SQ shard downloads."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
repo = "HeartMuLa/HeartCodec-oss-20260123"
revision = "f889dab0532cfa4bf459f2a3367eb6d346b8eeda"
path = snapshot_download(repo, revision=revision,
                         allow_patterns=["config.json", "model.safetensors.index.json", "*.safetensors"], max_workers=2)
args.output.mkdir(parents=True, exist_ok=True)
(args.output / "source.json").write_text(json.dumps({"repo": repo, "revision": revision, "snapshot": path}, indent=2) + "\n")
print(path, flush=True)
