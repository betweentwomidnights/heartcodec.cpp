#!/usr/bin/env python3
"""Validate stereo layout, token padding/context, overlap and full decode."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import numpy as np
from flow_metrics import metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--flow-cli", type=Path, required=True)
    p.add_argument("--sq-cli", type=Path, required=True)
    p.add_argument("--flow-model", type=Path, required=True)
    p.add_argument("--sq-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", default="cuda", choices=("cpu", "cuda", "vulkan", "metal"))
    p.add_argument("--reuse", action="store_true", help="Reuse already computed outputs for additional diagnostic comparisons")
    p.add_argument("--cases", default="")
    a = p.parse_args()
    manifest = json.loads((a.reference / "manifest.json").read_text())
    previous = json.loads((a.output / "report.json").read_text()) if a.reuse and (a.output / "report.json").exists() else {}
    report = dict(backend=a.backend, cases=[], gates=dict(flow_max_abs=1e-3, flow_relative_l2=1e-4,
        waveform_max_abs=1e-2, waveform_relative_l2=1e-3, scalar_level_mismatch_fraction=1e-4))
    for case in manifest["cases"]:
        if a.cases and case["name"] not in a.cases.split(","): continue
        root = a.reference / case["name"]; output = a.output / case["name"]; output.mkdir(parents=True, exist_ok=True)
        result = dict(name=case["name"], windows=[])
        def ref(name): return np.fromfile(root / case["files"][name]["file"], np.float32)
        def run(cli, op, model, inp, out, extra=()):
            if a.reuse and out.exists(): return np.fromfile(out, np.float32)
            command = [str(cli.resolve()), op, "--model", str(model.resolve()), "--backend", a.backend,
                       "--input", str(inp.resolve()), "--output", str(out.resolve()), *map(str, extra)]
            run_result = subprocess.run(command, capture_output=True, text=True)
            out.with_suffix(".stderr.txt").write_text(run_result.stderr)
            if run_result.returncode: raise RuntimeError(run_result.stdout + run_result.stderr)
            return np.fromfile(out, np.float32)
        assembled = [None, None]
        native_assembled = [None, None]
        window = case["segment_codes"]; hop = window // 93 * 80 * 3840; overlap = (window - window // 93 * 80) * 3840
        for i, details in enumerate(case["windows"]):
            prefix = f"window.{i}."
            native = run(a.flow_cli, "flow", a.flow_model, root / (prefix + "codes.i32"), output / (prefix + "latent.f32"),
                ["--steps", case["steps"], "--guidance", case["guidance"], "--noise", (root / (prefix + "noise.f32")).resolve(),
                 "--context", (root / (prefix + "context.f32")).resolve(), "--prefix", details["prefix_frames"],
                 "--schedule", (root / (prefix + "schedule.f32")).resolve()])
            expected = ref(prefix + "latent")
            levels_expected = np.round(expected * np.float32(9))
            levels_native = np.round(native * np.float32(9))
            mismatch = int(np.count_nonzero(levels_expected != levels_native))
            w = dict(flow=metrics(expected, native), scalar_level_mismatches=mismatch,
                     scalar_level_mismatch_fraction=mismatch / native.size, channels=[])
            w["changed_levels"] = [dict(index=int(j), frame=int(j // 256), channel=int(j % 256 // 128),
                feature=int(j % 128), reference=float(expected[j]), native=float(native[j]),
                reference_level=int(levels_expected[j]), native_level=int(levels_native[j]))
                for j in np.flatnonzero(levels_expected != levels_native)]
            teacher = ref(prefix + "sq_input").reshape(2, 128, window * 2)
            for channel in range(2):
                inp = output / f"{prefix}channel.{channel}.input.f32"; teacher[channel].tofile(inp)
                audio = run(a.sq_cli, "sq-decode", a.sq_model, inp, output / f"{prefix}channel.{channel}.audio.f32")
                expected_audio = ref(prefix + "audio").reshape(2, -1)[channel]
                w["channels"].append(metrics(expected_audio, audio))
                if assembled[channel] is None: assembled[channel] = audio
                else:
                    weight = np.linspace(0, 1, overlap)
                    assembled[channel][-overlap:] = (assembled[channel][-overlap:].astype(np.float64) * (1 - weight) + audio[:overlap].astype(np.float64) * weight).astype(np.float32)
                    assembled[channel] = np.concatenate([assembled[channel], audio[overlap:]])
                native_input = np.ascontiguousarray(native.reshape(window * 2, 256)[:, channel * 128:(channel + 1) * 128].T)
                native_path = output / f"{prefix}channel.{channel}.native-input.f32"; native_input.tofile(native_path)
                native_audio = run(a.sq_cli, "sq-decode", a.sq_model, native_path, output / f"{prefix}channel.{channel}.native-audio.f32")
                if native_assembled[channel] is None: native_assembled[channel] = native_audio
                else:
                    weight = np.linspace(0, 1, overlap)
                    native_assembled[channel][-overlap:] = (native_assembled[channel][-overlap:].astype(np.float64) * (1 - weight) + native_audio[:overlap].astype(np.float64) * weight).astype(np.float32)
                    native_assembled[channel] = np.concatenate([native_assembled[channel], native_audio[overlap:]])
            w["passed"] = bool(w["flow"]["passed"] and w["scalar_level_mismatch_fraction"] <= 1e-4 and all(c["passed"] for c in w["channels"]))
            result["windows"].append(w)
        # Independent assembly of teacher-forced native SQ windows.
        target = case["frames"] * 3840
        teacher_stereo = np.stack([v[:target] for v in assembled]).reshape(-1)
        result["teacher_forced_assembly"] = metrics(ref("audio"), teacher_stereo)
        sampled = []; stop = threading.Event()
        def monitor():
            while not stop.is_set():
                try:
                    value = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    sampled.append(int(value.strip().splitlines()[0]))
                except (OSError, ValueError, subprocess.SubprocessError): pass
                stop.wait(0.25)
        thread = threading.Thread(target=monitor, daemon=True)
        if a.backend == "cuda": thread.start()
        started = time.monotonic()
        try:
            full = run(a.flow_cli, "decode", a.flow_model, root / "codes.i32", output / "stereo.f32",
                ["--sq-model", a.sq_model.resolve(), "--steps", case["steps"], "--guidance", case["guidance"],
                 "--segment-codes", window, "--noise", (root / "noises.f32").resolve(), "--stats", (output / "stats.json").resolve()])
        finally:
            stop.set()
            if thread.is_alive(): thread.join()
        result["wall_seconds_including_model_loads"] = time.monotonic() - started
        result["end_to_end"] = metrics(ref("audio"), full, absolute=1e-2, relative=1e-3)
        assembled_native = np.stack([v[:target] for v in native_assembled]).reshape(-1)
        result["end_to_end_vs_native_flow_assembly"] = metrics(assembled_native, full)
        changed_levels = sum(w["scalar_level_mismatches"] for w in result["windows"])
        result["end_to_end_waveform_gate_applies"] = changed_levels == 0
        result["end_to_end_gate_policy"] = "When SQ levels change, gate flow error and level-change rate separately; require full native decode to match independent assembly of native flow outputs. Preserve upstream waveform error as an evaluation. No tolerance changes."
        result["output_samples_per_channel"] = full.size // 2
        result["target_length_exact"] = full.size == 2 * target
        result["maximum_observed_device_used_mib"] = max(sampled, default=None)
        if a.reuse:
            old = next((c for c in previous.get("cases", []) if c["name"] == case["name"]), {})
            result["maximum_observed_device_used_mib"] = old.get("maximum_observed_device_used_mib")
            result["wall_seconds_including_model_loads"] = old.get("wall_seconds_including_model_loads")
        result["memory_sampling"] = "nvidia-smi total GPU memory, 250ms polling; includes desktop/driver; may miss transients"
        result["stats"] = json.loads((output / "stats.json").read_text())
        result["passed"] = bool(result["target_length_exact"] and result["teacher_forced_assembly"]["passed"] and
                                 result["end_to_end_vs_native_flow_assembly"]["passed"] and
                                 (changed_levels > 0 or result["end_to_end"]["passed"]) and all(w["passed"] for w in result["windows"]))
        report["cases"].append(result)
        print(case["name"], "COMPONENTS/ASSEMBLY PASS" if result["passed"] else "FAIL",
              "upstream waveform evaluation:", result["end_to_end"], flush=True)
        (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["passed"] = all(c["passed"] for c in report["cases"])
    report["strict_upstream_waveform_gate_passed"] = all(c["end_to_end"]["passed"] for c in report["cases"])
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__": main()
