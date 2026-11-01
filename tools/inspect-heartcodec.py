#!/usr/bin/env python3
"""Audit pinned HF safetensors headers without downloading tensor payloads.

Standard library only; safe to run before a PyTorch/CUDA environment exists.
The storage scenarios exclude activations, workspace, KV caches and GGUF overhead.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import urllib.parse
import urllib.request


DEFAULT_REPOS = (
    "HeartMuLa/HeartCodec-oss-20260123",
    "HeartMuLa/HeartCodec-oss-encoder",
    "HeartMuLa/MuLaCover",
    "Qwen/Qwen3-Embedding-0.6B",
)
MAX_HEADER_BYTES = 16 * 1024 * 1024
FLOAT_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2,
               "F8_E4M3": 1, "F8_E5M2": 1, "F8_E4M3FN": 1}
INTEGER_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2,
                 "U16": 2, "I32": 4, "U32": 4, "I64": 8, "U64": 8}


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def read_range(url, start, length):
    request = urllib.request.Request(url, headers={
        "Range": f"bytes={start}-{start + length - 1}",
        "Accept-Encoding": "identity",
    })
    with urllib.request.urlopen(request, timeout=60) as response:
        # Refuse servers that ignore Range, avoiding multi-GB accidental reads.
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
        if response.status != 206 or not match:
            raise ValueError(f"Server did not honor Range: {response.status}, {url}")
        first, last, total = map(int, match.groups())
        if first != start or last != start + length - 1 or last >= total:
            raise ValueError(f"Unexpected Content-Range: {content_range}")
        data = response.read(length + 1)
        if len(data) != length:
            raise ValueError(f"Expected {length} bytes; received {len(data)}")
        return data, total


def read_header(url):
    prefix, total = read_range(url, 0, 8)
    size = struct.unpack("<Q", prefix)[0]
    if size < 2 or size > MAX_HEADER_BYTES or size + 8 > total:
        raise ValueError(f"Invalid safetensors header size: {size}")
    # Request from zero: HF/CDN caches distinguish this from the first range.
    raw, second_total = read_range(url, 0, size + 8)
    if second_total != total or raw[:8] != prefix:
        raise ValueError("Safetensors changed between range requests")
    header = json.loads(raw[8:])
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    return header, {
        "file_bytes": total,
        "header_bytes": size,
        "header_sha256": hashlib.sha256(raw[8:]).hexdigest(),
        "bytes_read": 16 + size,
    }


def component(name):
    name = name.removeprefix("decoder.")
    for prefix in ("scalar_model.encoder", "scalar_model.decoder",
                   "flow_matching.vq_embed", "flow_matching.estimator",
                   "encoder.feature_extractors.muencoder",
                   "encoder.feature_extractors.wavlm_encoder",
                   "encoder.feature_extractors.whisper_encoder", "encoder.query"):
        if name == prefix or name.startswith(prefix + "."):
            return prefix
    return name.split(".")[0]


def tensor_sizes(name, tensor, payload_bytes):
    shape, dtype = tensor["shape"], tensor["dtype"]
    if any(type(dim) is not int or dim < 0 for dim in shape):
        raise ValueError(f"Invalid shape for {name}: {shape}")
    elements = math.prod(shape)
    start, end = tensor["data_offsets"]
    if not 0 <= start <= end <= payload_bytes:
        raise ValueError(f"Invalid offsets for {name}")
    width = {**FLOAT_BYTES, **INTEGER_BYTES}.get(dtype)
    if width is None or end - start != elements * width:
        raise ValueError(f"Unknown dtype or inconsistent byte count: {name}, {dtype}")
    floating = dtype in FLOAT_BYTES
    # This is only a storage scenario. Native layout and weight-norm folding
    # must be validated by the converter before deciding quantization eligibility.
    eligible = (floating and len(shape) == 2 and shape[-1] % 256 == 0
                and (name.endswith(".weight") or name.endswith(".original1")))
    f32 = elements * 4 if floating else end - start
    f16 = elements * 2 if floating else end - start
    return {
        "tensors": 1, "elements": elements, "stored_bytes": end - start,
        "f32_bytes": f32, "f16_bytes": f16,
        "rank2_q4_k_rest_f32_bytes": elements // 256 * 144 if eligible else f32,
        "rank2_q6_k_rest_f32_bytes": elements // 256 * 210 if eligible else f32,
        "rank2_k_eligible_elements": elements if eligible else 0,
    }


def audit(repo):
    info = fetch_json("https://huggingface.co/api/models/" + repo)
    revision = info["sha"]
    root = f"https://huggingface.co/{repo}/resolve/{revision}/"
    files = {item["rfilename"] for item in info["siblings"]}
    index = None
    if "model.safetensors.index.json" in files:
        index = fetch_json(root + "model.safetensors.index.json")
        shards = sorted(set(index["weight_map"].values()))
    elif "encoder.safetensors" in files:
        shards = ["encoder.safetensors"]
    elif "model.safetensors" in files:
        shards = ["model.safetensors"]
    else:
        raise ValueError(f"No supported safetensors layout in {repo}")
    groups, totals, seen, shard_reports = {}, Counter(), {}, {}
    for shard in shards:
        url = root + urllib.parse.quote(shard, safe="/")
        header, meta = read_header(url)
        shard_reports[shard] = meta
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if name in seen:
                raise ValueError(f"Duplicate tensor: {name}")
            seen[name] = shard
            sizes = tensor_sizes(name, tensor, meta["file_bytes"] - meta["header_bytes"] - 8)
            totals.update(sizes)
            groups.setdefault(component(name), Counter()).update(sizes)
    if index is not None:
        if seen != index["weight_map"]:
            raise ValueError(f"Index/header tensor mismatch in {repo}")
        expected = index.get("metadata", {}).get("total_size")
        if expected is not None and expected != totals["stored_bytes"]:
            raise ValueError(f"Index/header size mismatch in {repo}")
    return {
        "repo": repo, "revision": revision,
        "config": fetch_json(root + "config.json"),
        "shards": shard_reports, "totals": dict(totals),
        "components": {key: dict(value) for key, value in sorted(groups.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", help="HF model ID; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Header inventory only; no model execution or peak-memory measurement.",
            "F16/F32 scenarios change floating tensor storage only.",
            "Q4_K/Q6_K scenarios cover eligible rank-2 weights with all other tensors F32.",
            "Scenarios are not a validated Q4_K_M policy, GGUF conversion, or quality result.",
            "Exclude activations, workspace, KV cache, allocator and file overhead.",
        ],
        "models": [],
    }
    for repo in args.repo or DEFAULT_REPOS:
        result = audit(repo)
        report["models"].append(result)
        totals = result["totals"]
        print(f"{repo}@{result['revision']}: {totals['stored_bytes'] / 2**30:.3f} GiB stored")
        for name, sizes in result["components"].items():
            print(f"  {name}: {sizes['f32_bytes'] / 2**30:.3f} GiB F32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
