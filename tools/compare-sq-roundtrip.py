#!/usr/bin/env python3
"""Evaluate local SQ encoder, teacher-forced decoder, and roundtrip from saved goldens."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def metrics(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("Wrong output shape or nonfinite values")
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    delta = a - b
    return {"max_abs": float(np.abs(delta).max()),
            "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-30)),
            "rmse": float(np.sqrt(np.mean(delta * delta)))}


def spectral_metrics(actual, expected):
    def magnitude(signal):
        signal = np.pad(signal.astype(np.float64), (0, max(0, 1024 - len(signal))))
        frames = np.lib.stride_tricks.sliding_window_view(signal, 1024)[::256]
        return np.abs(np.fft.rfft(frames * np.hanning(1024), axis=-1))
    a, b = magnitude(actual), magnitude(expected)
    return {"spectral_convergence": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)),
            "log_magnitude_l1": float(np.mean(np.abs(np.log(np.maximum(a, 1e-7)) - np.log(np.maximum(b, 1e-7)))))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--case", action="append")
    parser.add_argument("--evaluate", action="store_true", help="Record reduced-precision quality without applying FP32 parity gates")
    parser.add_argument("--stages", action="store_true", help="Compare available encoder stage goldens")
    args = parser.parse_args()
    manifest = json.loads((args.reference / "manifest.json").read_text())
    report = {"model": str(args.model), "model_bytes": args.model.stat().st_size,
              "reference": str(args.reference),
              "stft": {"fft_size": 1024, "hop": 256, "window": "symmetric Hann", "center": False, "log_floor": 1e-7},
              "backend": args.backend, "ggml_cuda_tf32": os.environ.get("GGML_CUDA_TF32"),
              "mode": "quality-evaluation" if args.evaluate else "fp32-parity", "cases": []}
    passed = True
    for case in manifest["cases"]:
        if args.case and case["name"] not in args.case:
            continue
        target = args.output / case["name"]
        target.mkdir(parents=True, exist_ok=True)
        source = args.reference / case["name"]
        result = {"name": case["name"]}

        def run(operation, input_path, output_name):
            out = target / (output_name + ".f32")
            stats = target / (output_name + ".json")
            command = [str(args.cli.resolve()), operation, "--model", str(args.model),
                            "--input", str(input_path), "--output", str(out), "--backend", args.backend,
                            "--stats", str(stats)]
            if args.stages and operation == "sq-encode":
                command.extend(["--dump-dir", str(target / "stages")])
            process = subprocess.run(command, capture_output=True, text=True)
            (target / (output_name + ".stderr.log")).write_text(process.stderr)
            if process.returncode:
                raise RuntimeError(f"Native {operation} failed ({process.returncode}): {process.stderr}")
            return np.fromfile(out, dtype="<f4"), json.loads(stats.read_text())

        reference_latent = np.load(source / "sq.latent.npy").ravel()
        reference_audio = np.load(source / "sq.decoded.npy").ravel()
        decoded, result["decode_stats"] = run("sq-decode", source / "sq.latent.f32", "decoded")
        result["decode"] = metrics(decoded, reference_audio)
        result["decode"].update(spectral_metrics(decoded, reference_audio))
        if "waveform" in case["tensors"]:
            encoded, result["encode_stats"] = run("sq-encode", source / "waveform.f32", "encoded")
            result["encode"] = metrics(encoded, reference_latent)
            quantized = np.round(np.float32(9) * encoded) / np.float32(9)
            reference_quantized = np.load(source / "sq.quantized.npy").ravel()
            mismatches = int(np.count_nonzero(quantized != reference_quantized))
            result["quantized_mismatches"] = mismatches
            result["quantized_elements"] = len(encoded)
            result["quantized_mismatch_rate"] = mismatches / len(encoded)
            result["roundtrip_reuses_reference_levels"] = mismatches == 0
            # Exercise encode + decode on one native context as hosts will do.
            roundtrip, result["roundtrip_stats"] = run("sq-roundtrip", source / "waveform.f32", "roundtrip")
            result["roundtrip"] = metrics(roundtrip, reference_audio)
            result["roundtrip"].update(spectral_metrics(roundtrip, reference_audio))
            if args.stages:
                result["encoder_stages"] = {}
                for name in sorted(key for key in case["tensors"] if key.startswith("encoder.")):
                    actual = np.fromfile(target / "stages" / (name + ".f32"), dtype="<f4")
                    expected = np.load(source / (name + ".npy")).ravel()
                    result["encoder_stages"][name] = metrics(actual, expected)
        ok = result["decode"]["max_abs"] <= 1e-3 and result["decode"]["relative_l2"] <= 1e-3
        if "encode" in result:
            ok &= result["encode"]["max_abs"] <= 1e-4 and result["encode"]["relative_l2"] <= 1e-4
            ok &= result["quantized_mismatch_rate"] <= 1e-4
            if result["roundtrip_reuses_reference_levels"]:
                ok &= result["roundtrip"]["max_abs"] <= 1e-3 and result["roundtrip"]["relative_l2"] <= 1e-3
            for stage in result.get("encoder_stages", {}).values():
                ok &= stage["max_abs"] <= 1e-3 and stage["relative_l2"] <= 1e-3
        result["fp32_gates_passed"] = bool(ok)
        passed &= ok
        report["cases"].append(result)
        print(f"{case['name']}: decode rel={result['decode']['relative_l2']:.3g}, "
              f"encode rel={result.get('encode', {}).get('relative_l2', 0):.3g}, "
              f"SQ mismatch={result.get('quantized_mismatch_rate', 0):.3g}", flush=True)
    if not report["cases"]:
        raise ValueError("No matching cases")
    report["fp32_gates_passed"] = bool(passed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    if not passed and not args.evaluate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
