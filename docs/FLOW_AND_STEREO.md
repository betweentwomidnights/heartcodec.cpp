# RVQ, flow and stereo decoder validation

November 1, 2026. The RTX 5070 Laptop is now the primary native development
environment for **complete RVQ-token-to-48kHz-stereo decoding**, including full
FP32 flow weights. Spark remains the Docker-only upstream oracle environment.
Audio-to-RVQ feature/query encoders are not implemented.

## Implemented path

- Eight 8192-entry, 32-dimensional RVQ codebook lookups, summed before one shared
  32-to-512 projection. Conditional projection and 2x nearest time expansion.
- Full 24+6-block flow transformer: convolution/linear projections, timestep
  embeddings, AdaLN modulation/gates, RMSNorm, adjacent-pair RoPE, noncausal
  attention, SwiGLU and output normalization.
- Fixed Euler integration, conditional or classifier-free guidance, saved-noise
  injection, optional saved schedules, and exact prefix replacement.
- Stereo latent splitting, repeated/padded token windows, preceding-window
  latent context, SQ waveform decoding, float64 overlap mixing, and exact trimming.
- Component registration/loading/unloading and the negotiated C ABI/CMake package.

All tensor work uses standard operations in `betweentwomidnights/ggml` at
`07b77b0112cc30861b49116b057bac33e819ee06`. No model-local generic kernels were
added. Guidance branches execute sequentially to keep graph memory bounded.
Every input, including positions and scalar constants, is restored on each graph
invocation because gallocr can recycle its storage after its final consumer.

Flow GGUF architecture is `heartcodec-flow`, version 1, strict float32, 339 runtime
tensors. Conversion loads all 365 upstream state entries strictly and omits only
RVQ encoder projection/training buffers. SQ GGUFs remain separate; the flow model
is unloaded before SQ waveform graphs are allocated. Flow quantization is deferred.

## Oracle and gates

Source: heartlib `5858ca8f1ffe58d62be7c007ea55f1033c5ca456` and HeartCodec checkpoint
`f889dab0532cfa4bf459f2a3367eb6d346b8eeda`. Docker reference uses Torch
2.7.1+cu128, float32, deterministic settings and TF32 disabled. Native CUDA also
uses `GGML_CUDA_TF32=0`. No seed equivalence across runtimes is assumed: actual
noise arrays and schedules are frozen.

Flow fixtures cover 4, 9 and 372 token frames; estimator intermediate stages;
guided Euler trajectories; exact context prefixes; and 1-, 3-, 4- and 10-step
integration. The exported trace is checked against upstream `solve_euler` itself.
Flow gates are maximum absolute error <=1e-3 and relative L2 <=1e-4.
The full-window estimator maximum error is about 1.02e-5 on laptop CUDA.
All six flow cases pass CUDA; five short cases pass CPU. The original 33 SQ
fixtures also pass the post-integration regression with zero changed levels on
both laptop backends. Static/DLL C consumers, shared/imported GGML parent builds,
and the ARM64 Linux installed package are checked separately.

Stereo fixtures exercise four-code repetition/trimming, a two-window 100-code
input with 93-code windows, and full 372-code windows. Independently decoding saved
upstream latents validates both SQ channels and overlap assembly. Independently
assembling native-generated latents must also match the complete native API.

Small FP32 flow differences can cross the discontinuous SQ rounding boundary.
The four-step full-window case changes **3 of 190,464 scalar levels** (0.00158%).
Its upstream waveform comparison is **0.010835 maximum absolute error and 0.001636
relative L2**, exceeding the strict end-to-end waveform evaluation gate. With
identical upstream SQ inputs, maximum waveform error is only 9.82e-6; independent
assembly of native flow outputs matches the full native decode exactly.

Reports retain that failed strict waveform evaluation. Component acceptance gates
flow error and changed-level rate (<=1e-4) separately, then requires exact pipeline
layout/length and close agreement with independent native assembly. When levels
are unchanged, the upstream waveform gate also applies. This is numerical parity
with an explicit quantizer-boundary limitation, not a claim of identical full
decoder waveforms. No quantizer snapping or fixture-specific correction is used.

The separate default **ten-step** full-window fixture changes one scalar level.
Its upstream waveform comparison is 0.005443 maximum error and 0.0008193 relative
L2 (0.0819%), passing the strict evaluation gate. Its native complete decode and
independent native-flow assembly match exactly; flow latent maximum error is
6.29e-6. Both four- and ten-step cases return exactly 1,428,480 samples per channel.

## Laptop memory and timing

| Measurement | Result |
| --- | ---: |
| Resident flow model buffers | 5,957,189,632 bytes / 5.55 GiB |
| Full-window flow compute graph, four-step sampling | 101,903,488 bytes / 97.2 MiB |
| Resident combined SQ model buffers | 672,182,784 bytes / 641 MiB |
| Full-window SQ compute graph | 3,291,221,248 bytes / 3.07 GiB |
| Maximum sampled total GPU use during four-step full stereo decode | 5,906 MiB |
| Four-step full-window flow, excluding model loading | about 2.75 seconds |
| Four-step full stereo decode, including component switch but excluding initial flow load | about 4.92 seconds |
| Ten-step full stereo decode, including component switch but excluding initial flow load | about 9.64 seconds |

Total device samples include the desktop/driver and use 250ms polling; they can
miss short-lived peaks. Native buffer counts exclude backend context/workspace.
Timing is a single validation run, not a steady-state performance claim.

The laptop does not need Spark for routine RVQ/flow/SQ C++ work. MuLaCover's full
FP32 guided upstream reference peaked around 27.5 GiB allocated on Spark; its
unquantized oracle remains a Spark workload. A native Q4_K_M generator's memory
and quality still require implementation and measurement.

## Reproduction and artifacts

```powershell
.\scripts\build-windows.ps1 -Backend cuda
.\scripts\evaluate-windows.ps1 -Backend cuda -Suite flow
.\scripts\evaluate-windows.ps1 -Backend cuda -Suite stereo
```

`heartcodec-flow` exposes `rvq`, `step`, `flow` and `decode` CLI operations. Token
files are raw `[8,T]` int32, flow tensors are frame-major float32, stereo outputs
are planar float32. `--noise`, `--context`, `--prefix`, `--schedule`, `--dump-dir`
and `--stats` support oracle comparison. See `docs/C_ABI_V1.md` for embedding.

Raw weights and fixtures remain ignored under `models/` and `refdata/`. Compact
comparison reports are under `docs/validation/`. Spark reproduction commands are
`scripts/spark.sh prepare-decoder`, `convert-flow`, `reference-flow` and
`reference-stereo`, after building `image-reference`. MuLaCover's separate sibling
repository contains the FP32/BF16 symbolic/prompt/style/logit/KV fixture harness.
