#!/usr/bin/env python3
"""Freeze RVQ, teacher-forced flow stages, CFG and Euler trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import numpy as np
from flow_reference import load_flow, get_snapshot
from sq_reference import configure_torch, revision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heartlib", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", default="4,9,372")
    a = p.parse_args()
    configure_torch()
    torch.manual_seed(1701)
    model, config = load_flow(a.heartlib, get_snapshot(a.source))
    model = model.cuda()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(heartlib_revision=revision(a.heartlib), checkpoint_revision=get_snapshot(a.source).name,
                    torch=torch.__version__, numpy=np.__version__, dtype="float32", cases=[])
    with torch.inference_mode():
        cases = [(frames, 4) for frames in map(int, a.frames.split(","))]
        cases += [(4, steps) for steps in (1, 3, 10)]
        for frames, steps in cases:
            root = a.output / (f"frames-{frames}" + (f"-steps-{steps}" if steps != 4 else ""))
            root.mkdir(exist_ok=True)
            files = {}
            def save(name, value):
                value = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
                value = np.ascontiguousarray(value)
                path = root / (name + (".i32" if value.dtype == np.int32 else ".f32"))
                path.write_bytes(value.tobytes())
                files[name] = dict(file=path.name, shape=list(value.shape), dtype=str(value.dtype),
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            codes = torch.randint(0, 8192, (1, frames, 8), device="cuda")
            codes[:, 0] = torch.tensor([0, 8191, 1, 8190, 42, 314, 4096, 7], device="cuda")
            save("codes", codes[0].T.contiguous().to(torch.int32))
            rvq = model.vq_embed.get_output_from_indices(codes)
            save("rvq", rvq)
            mu = model.cond_feature_emb(rvq).repeat_interleave(2, dim=1)
            save("mu", mu)
            x = torch.randn(1, frames * 2, 256, device="cuda")
            context = torch.zeros_like(x)
            prefix = min(3, frames)
            context[:, :prefix] = torch.randn_like(context[:, :prefix])
            save("noise", x)
            save("context", context)
            handles = []
            for name, module in model.estimator.named_modules():
                if name in ("proj_in", "connection_proj", "proj_out", "transformer_blocks.0",
                            "transformer_blocks.23", "transformer_blocks_2.0", "transformer_blocks_2.5"):
                    handles.append(module.register_forward_hook(lambda m, i, o, n=name: save("stage." + n, o)))
            t = torch.tensor([0.375], device="cuda")
            save("timestep_embedding", model.estimator.adaln_single.emb.timestep_embedding(t))
            velocity = model.estimator(torch.cat([x, context, mu], dim=-1), t)
            save("velocity", velocity)
            for h in handles:
                h.remove()
            # Freeze the schedule and each state. The native oracle must consume
            # serialized noise, rather than assuming identical RNG algorithms.
            schedule = torch.linspace(0, 1, steps + 1, device="cuda")
            save("schedule", schedule)
            trajectory = [x.clone()]
            noise = x.clone()
            time = schedule[0]
            dt = schedule[1] - time
            for i in range(steps):
                t = time.reshape(1)
                x[:, :prefix] = (1 - (1 - 1e-6) * t) * noise[:, :prefix] + t * context[:, :prefix]
                inp = torch.cat([torch.cat([x, context, torch.zeros_like(mu)], -1),
                                 torch.cat([x, context, mu], -1)], 0)
                uncond, cond = model.estimator(inp, t.repeat(2)).chunk(2)
                velocity = uncond + 1.25 * (cond - uncond)
                x = x + dt * velocity
                trajectory.append(x.clone())
                time = time + dt
                if i + 1 < steps:
                    dt = schedule[i + 2] - time
            oracle = model.solve_euler(noise.clone(), context, prefix, schedule, mu, 1.25)
            if not torch.equal(oracle, x):
                raise RuntimeError("Frozen Euler trace differs from the upstream solver")
            x[:, :prefix] = context[:, :prefix]
            save("trajectory", torch.cat(trajectory))
            save("flow_output", x)
            save("sq_left", x[0, :, :128].T.contiguous())
            save("sq_right", x[0, :, 128:].T.contiguous())
            case = dict(name=root.name, frames=frames, prefix_frames=prefix, steps=steps, guidance=1.25,
                        velocity_time=0.375, files=files, peak_allocated_bytes=torch.cuda.max_memory_allocated())
            (root / "case.json").write_text(json.dumps(case, indent=2) + "\n")
            manifest["cases"].append(case)
            print(f"Frozen {root.name}", flush=True)
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
