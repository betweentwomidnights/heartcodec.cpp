# Laptop development and SQ quantization

November 1, 2026. Primary workspace: `C:\dev\heartcodec.cpp`.

## Environment decision

The RTX 5070 Laptop GPU can run the current complete Base SQ encoder/decoder
primitives and their quantization tests. Routine native development now happens
here, using the frozen Spark fixtures. It requires no running Spark service or
local PyTorch installation. Spark remains the Docker-based reference environment
when new, larger model components need new upstream tensors.

Validated local toolchain: Windows x64, MSVC 19.44, CUDA 12.8.61, `120a-real`,
NVIDIA driver 616.56, and the existing shared GGML pin `07b77b0112cc30861b49116b057bac33e819ee06`.
The laptop had about 7.7 GiB free GPU memory before testing.

## FP32 parity and memory

Both local CPU and CUDA passed:

- The 27 original decoder fixtures, now also checking native encoding and roundtrip.
- Six additional fixtures: impulse/sine/noise at 2 and 1,922 input samples.
- All eight encoder stages for impulse/sine/noise at 48,000 samples.
- Native C ABI/quantizer CTests and seven Python tooling tests.

All **33 fixtures** have **zero changed FP32 scalar quantizer levels** on both
backends. The largest relative encoder latent error in the original suite is
2.63e-6 on CPU and 4.99e-6 on CUDA. Comparisons use identical saved upstream inputs;
roundtrip tests also exercise encode/decode on a single native context.

The 1,922-sample test is intentional: pre-pooling floors to 961 samples, then
causal strided convolutions ceil at each stage, yielding **two** latent frames.
Decode returns 3,840 samples. The primitive does not silently trim to the original
input length. The public C header documents this policy.

Sampled total device memory reached **3,908 MiB** during CUDA parity, including
the long 29.76-second fixtures. This is an observed 4 Hz device-wide sample,
not an exact allocator peak. The native CLI separately records model and graph
buffer allocations; those exclude backend contexts and temporary workspaces.

Small reports are stored in `docs/validation/laptop-{cpu,cuda}-*.json`, with the
sampled memory trace and model checksums alongside them. Raw tensors, weights,
and builds remain ignored local artifacts.

## Native surface and packaging

`include/heartcodec.h` now exposes unquantized SQ encoding, scalar quantization,
SQ decoding, explicit result release, and buffer statistics. Inputs are mono
48 kHz audio; SQ latents are `[128, frames]` with time contiguous. The CLI supports
`sq-encode`, `sq-decode`, and `sq-roundtrip` using little-endian raw FP32 files.

GGUF format version 2 can contain `encoder`, `decoder`, or `codec` components.
The strict converter folds upstream weight normalization and stores convolution
weights as matrices:

- Conv1d: Torch `[OC, IC, K]` -> GGML `[IC*K, OC]`.
- ConvTranspose1d: Torch `[IC, OC, K]` -> GGML `[IC, OC*K]` for `col2im_1d`.

This allows row-aligned matrix quantization without new convolution kernels.
IM2COL geometry uses a small shape tensor; model weights remain matrix-packed.
Non-overlapping average pooling is expressed through existing view/mean/reshape
operators, so it works on CUDA as well as CPU. Legacy version 1 decoder GGUFs
remain accepted.

Debug snapshots materialize retained views before dumping, so a view's source
buffer cannot be reused before it is read back. The stage tests cover this path.

## Quantization policy and observed quality

`heartcodec-quantize` calls the shared GGML quantizers locally, starting from the
original FP32 GGUF. It rejects requantization, protects block alignment, and can
target `encoder`, `decoder`, or `codec` independently.

The experimental `sq-conv-v1` policy keeps slopes, biases, boundary projections,
and ineligible rows in FP32. Q4_K_M/Q5_K_M promote pointwise residual projections
and sampling convolutions to Q6_K; other eligible matrices use Q4_K/Q5_K. Q6_K
and Q8_0 variants use their respective types. The FP16 experiment converts all
matrix weights, including boundary weights. These are distinct policies, not
uniform bits-per-parameter comparisons.

Results below are from the **RTX 5070 CUDA backend**, on the same three-second
real-music channel, compared with upstream FP32:

| Model | File size | Changed encoder SQ levels | Decoder relative L2 error | Roundtrip relative L2 error |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 641.06 MiB | 0% | ~5.5e-7 | ~5.5e-7 |
| FP16 | 320.63 MiB | 0.0208% | 0.0129% | 0.182% |
| Q8_0 mixed | 179.41 MiB | 5.34% | 0.939% | 4.00% |
| Q6_K mixed | 146.61 MiB | 1.19% | 1.44% | 2.84% |
| Q5_K_M mixed | 134.41 MiB | 1.43% | 1.98% | 3.48% |
| Q4_K_M mixed | 122.93 MiB | 2.08% | 2.96% | 4.17% |
| Q4_K_M encoder / FP32 decoder | 381.99 MiB | 2.08% | ~5.5e-7 | 3.07% |
| FP32 encoder / Q4_K_M decoder | 381.99 MiB | 0% | 2.96% | 2.96% |

Decoder error uses the **reference latent**, isolating decoder quantization.
Roundtrip error includes native encoding and its changed quantized levels.
Sine, noise, and synthetic latent fixtures were evaluated too; all results are
recorded in `docs/validation/quant-{cpu,cuda}-*.json`. CPU FP16 behaves differently
from CUDA because backend arithmetic differs; these measurements must not be
treated as backend-independent model quality or monotonic precision rankings.

The reports also contain STFT magnitude convergence and log-magnitude error
(1,024-sample symmetric Hann window, hop 256, no centering, log floor 1e-7).
Full Q4_K_M roundtrip spectral convergence on the music fixture is 0.0284.
These numerical metrics do not establish perceptual transparency.

**Keep SQ FP32 as the validated default.** Q4_K_M works operationally, but this
policy has not preserved SQ level identity or FP32 waveform parity. It remains a
research option. These results do not decide quantization of the much larger flow
transformer, query encoder, or feature frontends; those require separate parity
and quantization measurements.

## Local workflow

Initialize tooling and build sequentially (Windows cannot relink an executable
while a test process is running it):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\scripts\build-windows.ps1 cpu
.\scripts\build-windows.ps1 cuda
.\scripts\evaluate-windows.ps1 -Backend cuda -Suite parity
.\scripts\evaluate-windows.ps1 -Backend cuda -Suite quantization
$env:HEARTCODEC_QUANTIZER = (Resolve-Path build-windows-cpu/bin/Release/heartcodec-quantize.exe).Path
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The frozen local inputs are `models/heartcodec-sq-codec-f32.gguf` and
`refdata/laptop-goldens/`, plus `refdata/sq-pooling-boundary/` and
`refdata/sq-encoder-stages/` for additional checks. The model checksums and source
checkpoint provenance are recorded in `docs/validation/`.

Explicit conversion/quantization and a single native roundtrip:

```powershell
.\build-windows-cpu\bin\Release\heartcodec-quantize.exe `
  --input models/heartcodec-sq-codec-f32.gguf `
  --output models/heartcodec-sq-codec-q4_k_m.gguf --mix q4_k_m
$env:GGML_CUDA_TF32 = '0'
.\build-windows-cuda\bin\Release\heartcodec-cli.exe sq-roundtrip `
  --model models/heartcodec-sq-codec-f32.gguf `
  --input refdata/laptop-goldens/music-channel-0-144000/waveform.f32 `
  --output refdata/local-roundtrip.f32 --backend cuda --stats refdata/local-stats.json
```

Reference-accurate CUDA tests explicitly disable TF32. Quantization evaluation
records failed FP32 gates rather than treating approximate results as parity.
Build and model data are local; no Spark access is needed for these commands.
