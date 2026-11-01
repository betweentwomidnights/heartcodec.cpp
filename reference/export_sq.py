#!/usr/bin/env python3
"""Export deterministic, trained-weight SQ fixtures and measured CUDA peaks."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch

from sq_reference import configure_torch, load_heartlib, load_spansynth, revision


def difference(actual, expected):
    a, b = actual.double(), expected.double()
    return {"max_abs": (a - b).abs().max().item(),
            "relative_l2": ((a - b).norm() / b.norm().clamp_min(1e-30)).item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heartlib", type=Path, required=True)
    parser.add_argument("--spansynth", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", default="1920,48000", help="Comma separated 48 kHz sample counts")
    parser.add_argument("--synthetic-latents", action="store_true", help="Also test seeded arbitrary decoder inputs")
    parser.add_argument("--audio", type=Path, help="Optional 48 kHz mono/stereo real-audio fixture")
    parser.add_argument("--encoder-stages", action="store_true", help="Also save each encoder stage for native debugging")
    args = parser.parse_args()
    configure_torch()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(0.25, device)
    args.output.mkdir(parents=True, exist_ok=True)
    model = load_heartlib(args.heartlib, args.weights, args.config).to(device)
    independent = (load_spansynth(args.spansynth, args.weights, args.config).to(device)
                   if args.spansynth else None)
    digest = hashlib.sha256()
    with args.weights.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    digest = digest.hexdigest()
    manifest = {"schema_version": 1, "torch": torch.__version__, "cuda": torch.version.cuda,
                "python": platform.python_version(), "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "heartlib_revision": revision(args.heartlib),
                "spansynth_revision": revision(args.spansynth) if args.spansynth else None,
                "weights_sha256": digest, "dtype": "float32", "tf32": False,
                "deterministic_algorithms": True, "sample_rate": 48000, "cases": []}
    audio = None
    if args.audio:
        import subprocess
        # Preserve stereo channel identity; decode/resample explicitly through ffmpeg.
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(args.audio),
                                       "-t", str(max(map(int, args.samples.split(","))) / 48000),
                                       "-ar", "48000", "-ac", "2", "-f", "f32le", "pipe:1"])
        audio = np.frombuffer(raw, dtype="<f4").reshape(-1, 2)
        if not np.isfinite(audio).all():
            raise ValueError("Real-audio fixture must be finite")
        manifest["audio_source"] = str(args.audio)
        manifest["audio_source_sha256"] = hashlib.sha256(args.audio.read_bytes()).hexdigest()
        manifest["audio_preprocess"] = "ffmpeg: resample to 48000 Hz, stereo, F32LE; channels tested independently"

    def save(directory, name, tensor):
        data = tensor.detach().float().cpu().contiguous().numpy().astype("<f4")
        if not np.isfinite(data).all():
            raise ValueError(f"Nonfinite reference output: {name}")
        np.save(directory / (name + ".npy"), data)
        data.tofile(directory / (name + ".f32"))
        return {"shape": list(data.shape), "dtype": "float32",
                "sha256": hashlib.sha256(data.tobytes()).hexdigest()}

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for samples in map(int, args.samples.split(",")):
            if samples < 2:
                raise ValueError("At least two samples are needed by preprocessor pooling")
            time_axis = torch.arange(samples, dtype=torch.float32) / 48000
            rng = torch.Generator(device="cpu").manual_seed(42)
            signals = ("impulse", "sine", "noise", "synthetic") if args.synthetic_latents else ("impulse", "sine", "noise")
            if audio is not None:
                if len(audio) < samples:
                    raise ValueError("Real-audio source is shorter than requested fixture length")
                signals = tuple(f"music-channel-{channel}" for channel in range(audio.shape[1]))
            for signal in signals:
                if signal.startswith("music-channel-"):
                    waveform = torch.from_numpy(audio[:samples, int(signal.rsplit("-", 1)[1])].copy())
                elif signal == "impulse":
                    waveform = torch.zeros(samples)
                    waveform[0] = 0.5
                elif signal == "sine":
                    waveform = 0.2 * torch.sin(2 * torch.pi * 440 * time_axis)
                else:
                    waveform = 0.05 * torch.randn(samples, generator=rng)
                directory = args.output / f"{signal}-{samples}"
                directory.mkdir(parents=True, exist_ok=True)
                case = {"name": directory.name, "source": "direct-latent" if signal == "synthetic" else "waveform", "tensors": {}}
                if signal != "synthetic":
                    case["tensors"]["waveform"] = save(directory, "waveform", waveform[None, None])
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                synchronize()
                start = time.perf_counter()
                if signal == "synthetic":
                    latent = (1.8 * torch.rand(1, 128, max(1, samples // 1920), generator=rng) - 0.9)
                    # Include exact FP32 halfway inputs and values outside tanh's range.
                    halfway = torch.arange(-8.5, 9, 1) / 9
                    latent.view(-1)[:len(halfway)] = halfway
                    latent.view(-1)[-2:] = torch.tensor([-1.25, 1.25])
                    latent = latent.to(device)
                else:
                    encoder_stages = {}
                    encoder_hooks = [layer.register_forward_hook(
                        lambda _m, _a, output, name=f"encoder.{i}": encoder_stages.update({name: output.detach().cpu()}))
                        for i, layer in enumerate(model.encoder)] if args.encoder_stages else []
                    latent = model.encode(waveform[None, None].to(device))
                    for hook in encoder_hooks:
                        hook.remove()
                    if args.encoder_stages:
                        encoder_stages["encoder.7"] = latent.detach().cpu() # final stage includes tanh
                        for name, tensor in encoder_stages.items():
                            case["tensors"][name] = save(directory, name, tensor)
                quantized = torch.round(9 * latent) / 9
                case["tensors"]["sq.latent"] = save(directory, "sq.latent", latent)
                case["tensors"]["sq.quantized"] = save(directory, "sq.quantized", quantized)
                # Serialize inputs before decoding: no dependence on cross-device RNG behavior.
                stages = {}
                hooks = [layer.register_forward_hook(
                    lambda _m, _a, output, name=f"decoder.{i}": stages.update({name: output.detach().cpu()}))
                    for i, layer in enumerate(model.decoder)]
                decoded = model.decode(latent)
                for hook in hooks:
                    hook.remove()
                synchronize()
                case["elapsed_seconds"] = time.perf_counter() - start
                case["tensors"]["sq.decoded"] = save(directory, "sq.decoded", decoded)
                for name, tensor in stages.items():
                    case["tensors"][name] = save(directory, name, tensor)
                if device.type == "cuda":
                    case["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                    case["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                if independent is not None:
                    other_latent = independent.encode(waveform[None, None].to(device)) if signal != "synthetic" else latent
                    other_decoded = independent.decode(latent)
                    if signal != "synthetic":
                        case["spansynth_encode"] = difference(other_latent, latent)
                    case["spansynth_decode"] = difference(other_decoded, decoded)
                    torch.testing.assert_close(other_latent, latent, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(other_decoded, decoded, rtol=1e-5, atol=1e-6)
                manifest["cases"].append(case)
                (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                print(f"{case['name']}: {tuple(latent.shape)} -> {tuple(decoded.shape)}, "
                      f"{case['elapsed_seconds']:.3f}s", flush=True)
    print(f"Wrote {args.output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
