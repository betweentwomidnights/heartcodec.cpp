#!/usr/bin/env python3
"""Compare native RVQ/estimator/Euler outputs against frozen Spark tensors."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from flow_metrics import metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--cli", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=("cpu", "cuda", "vulkan", "metal"), default="cuda")
    p.add_argument("--cases", default="")
    a = p.parse_args()
    manifest = json.loads((a.reference / "manifest.json").read_text())
    report = dict(backend=a.backend, model=str(a.model), reference_revision=manifest["heartlib_revision"],
                  gates=dict(max_abs=1e-3, relative_l2=1e-4), cases=[])
    for case in manifest["cases"]:
        if a.cases and case["name"] not in a.cases.split(","): continue
        root = a.reference / case["name"]
        output = a.output / case["name"]; output.mkdir(parents=True, exist_ok=True)
        def reference(name): return np.fromfile(root / case["files"][name]["file"], dtype=np.float32)
        def run(op, inp, extra):
            result = subprocess.run([str(a.cli.resolve()), op, "--model", str(a.model.resolve()), "--backend", a.backend,
                "--input", str(inp.resolve()), "--output", str((output / (op + ".f32")).resolve()),
                "--stats", str((output / (op + ".stats.json")).resolve()), *map(str, extra)], capture_output=True, text=True)
            (output / (op + ".stderr.txt")).write_text(result.stderr)
            if result.returncode: raise RuntimeError(result.stdout + result.stderr)
            return np.fromfile(output / (op + ".f32"), dtype=np.float32)
        codes = root / case["files"]["codes"]["file"]
        result = dict(name=case["name"], tensors={})
        result["tensors"]["rvq"] = metrics(reference("rvq"), run("rvq", codes, []))
        frames = case["frames"] * 2
        features = np.concatenate([reference("noise").reshape(frames, 256), reference("context").reshape(frames, 256),
                                   reference("mu").reshape(frames, 512)], axis=1)
        features.tofile(output / "input.f32")
        actual = run("step", output / "input.f32", ["--time", case["velocity_time"], "--dump-dir", output.resolve()])
        result["tensors"]["velocity"] = metrics(reference("velocity"), actual)
        for name in case["files"]:
            if name.startswith("stage."):
                result["tensors"][name] = metrics(reference(name), np.fromfile(output / (name + ".f32"), dtype=np.float32))
        actual = run("flow", codes, ["--steps", case["steps"], "--guidance", case["guidance"],
            "--noise", (root / "noise.f32").resolve(), "--context", (root / "context.f32").resolve(),
            "--prefix", case["prefix_frames"], "--schedule", (root / "schedule.f32").resolve(), "--dump-dir", output.resolve()])
        result["tensors"]["flow_output"] = metrics(reference("flow_output"), actual)
        trajectory = reference("trajectory").reshape(case["steps"] + 1, -1)
        for i in range(case["steps"] + 1):
            result["tensors"][f"euler.{i}"] = metrics(trajectory[i], np.fromfile(output / f"euler.{i}.f32", dtype=np.float32))
        result["tensors"]["mu"] = metrics(reference("mu"), np.fromfile(output / "mu.f32", dtype=np.float32))
        # Also exercise the public default schedule, without oracle injection.
        if case["steps"] != 4:
            generated = run("flow", codes, ["--steps", case["steps"], "--guidance", case["guidance"],
                "--noise", (root / "noise.f32").resolve(), "--context", (root / "context.f32").resolve(),
                "--prefix", case["prefix_frames"]])
            result["tensors"]["flow_default_schedule"] = metrics(reference("flow_output"), generated)
        result["prefix_exact"] = np.array_equal(actual.reshape(frames, 256)[:case["prefix_frames"]], reference("context").reshape(frames, 256)[:case["prefix_frames"]])
        result["passed"] = bool(result["prefix_exact"] and all(t["passed"] for t in result["tensors"].values()))
        result["stats"] = {op: json.loads((output / (op + ".stats.json")).read_text()) for op in ("rvq", "step", "flow")}
        report["cases"].append(result)
        print(case["name"], "PASS" if result["passed"] else "FAIL", result["tensors"]["velocity"], flush=True)
    report["passed"] = bool(report["cases"] and all(c["passed"] for c in report["cases"]))
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__": main()
