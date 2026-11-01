#!/usr/bin/env python3
"""Bundle existing golden inputs/final outputs for development without PyTorch."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, action="append", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--archive", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
manifest = None
names = set()
for source in args.source:
    data = json.loads((source / "manifest.json").read_text())
    if manifest is None:
        manifest = {key: value for key, value in data.items() if key != "cases"}
        manifest.update(cases=[], source_manifests={})
    if data["weights_sha256"] != manifest["weights_sha256"]:
        raise ValueError("Fixture sets must use identical trained weights")
    manifest["source_manifests"][source.name] = hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
    for case in data["cases"]:
        if case["name"] in names:
            raise ValueError("Duplicate case name")
        names.add(case["name"])
        case["tensors"] = {key: value for key, value in case["tensors"].items()
                           if key in ("waveform", "sq.latent", "sq.quantized", "sq.decoded")}
        case["reference_set"] = source.name
        target = args.output / case["name"]
        target.mkdir(parents=True, exist_ok=True)
        for name, info in case["tensors"].items():
            raw = source / case["name"] / (name + ".f32")
            if hashlib.sha256(raw.read_bytes()).hexdigest() != info["sha256"]:
                raise ValueError(f"Corrupt reference tensor: {raw}")
            for suffix in (".f32", ".npy"):
                shutil.copy2(source / case["name"] / (name + suffix), target / (name + suffix))
        manifest["cases"].append(case)
(args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
if args.archive:
    with tarfile.open(args.archive, "w:gz") as archive:
        archive.add(args.output, arcname=args.output.name)
print(f"Packed {len(names)} cases in {args.output}")
