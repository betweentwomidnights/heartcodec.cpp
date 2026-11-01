#!/usr/bin/env python3
"""Freeze upstream detokenize, serialized window noises and stereo assembly."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from flow_reference import load_flow, get_snapshot
from sq_reference import load_heartlib, load_module, configure_torch, revision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heartlib", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--sq-weights", type=Path, required=True)
    p.add_argument("--sq-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    configure_torch(); torch.manual_seed(1701)
    flow, _ = load_flow(a.heartlib, get_snapshot(a.source))
    sq = load_heartlib(a.heartlib, a.sq_weights, a.sq_config)
    module = load_module("heartcodec_flow_reference.decoder", a.heartlib / "src/heartlib/heartcodec/models/decoder.py")
    # Reuse the two strictly loaded components while executing the actual
    # upstream detokenize method, without another initialized weight copy.
    decoder = module.HeartCodecDecoder.__new__(module.HeartCodecDecoder)
    torch.nn.Module.__init__(decoder)
    decoder.flow_matching = flow; decoder.scalar_model = sq; decoder.sample_rate = 48000
    decoder.eval().cuda()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(heartlib_revision=revision(a.heartlib), checkpoint_revision=get_snapshot(a.source).name,
                    torch=torch.__version__, cases=[])
    original_solve, original_infer, original_decode = flow.solve_euler, flow.inference_codes, sq.decode
    with torch.inference_mode():
        for frames, window, steps in ((4, 93, 4), (100, 93, 4), (372, 372, 4), (372, 372, 10)):
            root = a.output / (f"frames-{frames}-window-{window}" + (f"-steps-{steps}" if steps != 4 else "")); root.mkdir(exist_ok=True)
            files, noises, windows = {}, [], []
            index = [0]; audio_index = [0]
            def save(name, tensor):
                value = np.ascontiguousarray(tensor.detach().cpu().numpy())
                path = root / (name + (".i32" if value.dtype == np.int32 else ".f32"))
                payload = value.tobytes(); path.write_bytes(payload)
                files[name] = dict(file=path.name, shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(payload).hexdigest())
            def solve(x, context, prefix, schedule, mu, guidance):
                i = index[0]
                save(f"window.{i}.noise", x); save(f"window.{i}.context", context)
                save(f"window.{i}.schedule", schedule)
                noises.append(x.detach().cpu().numpy())
                windows.append(dict(prefix_frames=int(prefix), steps=steps, guidance=1.25))
                return original_solve(x, context, prefix, schedule, mu, guidance)
            def infer(codes, *args, **kwargs):
                i = index[0]; save(f"window.{i}.codes", codes[0][0].to(torch.int32))
                value = original_infer(codes, *args, **kwargs)
                save(f"window.{i}.latent", value)
                index[0] += 1
                return value
            def decode(latent):
                i = audio_index[0]; audio_index[0] += 1
                save(f"window.{i}.sq_input", latent)
                value = original_decode(latent)
                save(f"window.{i}.audio", value)
                return value
            flow.solve_euler = solve; flow.inference_codes = infer; sq.decode = decode
            codes = torch.randint(0, 8192, (8, frames), device="cuda")
            save("codes", codes.to(torch.int32))
            audio = decoder.detokenize(codes, duration=window / 12.5, num_steps=steps, guidance_scale=1.25, disable_progress=True)
            save("audio", audio)
            all_noise = np.ascontiguousarray(np.concatenate(noises))
            save("noises", torch.from_numpy(all_noise))
            case = dict(name=root.name, frames=frames, segment_codes=window, steps=steps, guidance=1.25,
                        windows=windows, files=files, peak_allocated_bytes=torch.cuda.max_memory_allocated())
            (root / "case.json").write_text(json.dumps(case, indent=2) + "\n")
            manifest["cases"].append(case)
            print(f"Frozen stereo {root.name}: {len(windows)} windows", flush=True)
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__": main()
