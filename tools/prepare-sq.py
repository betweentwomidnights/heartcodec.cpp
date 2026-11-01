#!/usr/bin/env python3
"""Extract the pinned official Base SQ tensors; download only relevant shards."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

REPO = "HeartMuLa/HeartCodec-oss-20260123"
REVISION = "f889dab0532cfa4bf459f2a3367eb6d346b8eeda"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    def fetch(name):
        return Path(hf_hub_download(REPO, name, revision=REVISION))
    config = json.loads(fetch("config.json").read_text())
    index = json.loads(fetch("model.safetensors.index.json").read_text())
    expected = {key: value for key, value in index["weight_map"].items()
                if key.startswith("scalar_model.")}
    if not expected:
        raise ValueError("No scalar_model tensors in checkpoint index")
    tensors, sources = {}, {}
    for shard in sorted(set(expected.values())):
        path = fetch(shard)
        digest = sha256(path)
        sources[shard] = {"sha256": digest, "bytes": path.stat().st_size}
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in sorted(key for key, value in expected.items() if value == shard):
                tensors[name.removeprefix("scalar_model.")] = source.get_tensor(name)
    if len(tensors) != len(expected):
        raise ValueError("Incomplete SQ tensor extraction")
    weights = args.output / "sq.safetensors"
    save_file(tensors, weights, metadata={"source_repo": REPO, "source_revision": REVISION})
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    digest = sha256(weights)
    manifest = {"source_repo": REPO, "source_revision": REVISION,
                "source_shards": sources, "tensor_count": len(tensors),
                "sq_sha256": digest, "sq_bytes": weights.stat().st_size}
    (args.output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
