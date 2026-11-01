#!/usr/bin/env python3
"""Run the native decoder with identical saved latents and gate stage parity."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--threads", default="4")
    parser.add_argument("--max-relative-l2", type=float, default=1e-3)
    parser.add_argument("--max-abs", type=float, default=1e-3)
    args = parser.parse_args()
    manifest = json.loads((args.reference / "manifest.json").read_text())
    report = {"backend": args.backend, "reference": str(args.reference),
              "ggml_cuda_tf32": os.environ.get("GGML_CUDA_TF32"),
              "max_relative_l2": args.max_relative_l2, "max_abs": args.max_abs, "cases": []}
    passed = True
    for case in manifest["cases"]:
        directory = args.output / case["name"]
        directory.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(args.cli.resolve()), "sq-decode", "--model", str(args.model),
                        "--input", str(args.reference / case["name"] / "sq.latent.f32"),
                        "--output", str(directory / "sq.decoded.f32"),
                        "--backend", args.backend, "--threads", args.threads,
                        "--dump-dir", str(directory)], check=True)
        result = {"name": case["name"], "stages": {}}
        for name in sorted(key for key in case["tensors"] if key.startswith("decoder.") or key == "sq.decoded"):
            expected = np.load(args.reference / case["name"] / (name + ".npy")).astype(np.float64).ravel()
            actual = np.fromfile(directory / (name + ".f32"), dtype="<f4").astype(np.float64)
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise ValueError(f"Invalid native output: {case['name']}/{name}")
            delta = actual - expected
            max_abs = float(np.abs(delta).max())
            relative = float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-30))
            # The absolute gate keeps near-silent signals meaningful; the relative
            # gate catches proportional drift even when output amplitude is small.
            ok = max_abs <= args.max_abs and relative <= args.max_relative_l2
            passed &= ok
            result["stages"][name] = {"max_abs": max_abs, "relative_l2": relative, "passed": ok}
            print(f"{case['name']}/{name}: max_abs={max_abs:.3g}, rel_l2={relative:.3g}, {'PASS' if ok else 'FAIL'}", flush=True)
        report["cases"].append(result)
    report["passed"] = passed
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
