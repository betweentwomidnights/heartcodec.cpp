# Base SQ decoder milestone

Validated November 1, 2026, on DGX Spark GB10 inside isolated Docker containers.

This records the first decoder milestone. The subsequent native encoder,
laptop CPU/CUDA roundtrip validation, and local quantization experiments are in
[Laptop development](LAPTOP_DEVELOPMENT.md).

## Implemented

- Trained-weight Python SQ encoder/quantizer/decoder fixture generation.
- Exact HeartLib versus SpanSynth cross-checks using the same extracted weights.
- Strict FP32 decoder conversion to GGUF, evaluating PyTorch's actual weight-norm
  parametrization rather than assuming normalization axes.
- Native GGML SQ decoder with causal convolutions, asymmetric transposed-convolution
  cropping, residual units, scalar PReLU, and sample repetition.
- PyTorch-compatible scalar quantization, including halfway rounding and signed zero.
- Size-prefixed versioned C ABI, library-owned results, explicit errors, and CLI dumps.
- CPU and CUDA builds against the shared GGML fork. No private generic kernels.

## Results

Each native backend passed **27 fixtures**, comparing all eight decoder stages
and the final waveform from identical serialized latent inputs.

| Fixture set | Cases |
| --- | ---: |
| Impulse, sine, seeded noise: 1,920 and 48,000 samples | 6 |
| Same signals: 1,919, 1,921, 1,428,479, 1,428,480, 1,428,481 samples | 15 |
| Three short signals plus arbitrary seeded latents, including half-step inputs | 4 |
| Three seconds of bundled real music, L/R tested independently | 2 |

HeartLib and SpanSynth produced **identical SQ outputs** on all tested inputs.
Native differences from the CUDA PyTorch reference:

| Metric | Native CPU | Native CUDA |
| --- | ---: | ---: |
| Largest final-waveform absolute sample error | 3.29e-5 | 5.25e-5 |
| Largest final-waveform relative L2 error | 2.81e-5 | 3.21e-5 |
| Largest absolute error at any saved internal stage | 4.70e-4 | 7.56e-4 |

The comparison gates were maximum absolute error <= 1e-3 **and** relative L2
error <= 1e-3 at every saved tensor. Gates were unchanged when diagnosing CUDA
precision. Near-silent impulse tails produce the largest relative waveform errors.
Detailed manifests and comparisons are in [validation/](validation/).

The Python validation process peaked at **3.18 GiB allocated / 3.71 GiB reserved**
CUDA memory over the boundary suite. That process held **two complete SQ reference
models**, plus fixtures and framework workspace. This is measured process memory
on GB10, not a native decoder-only requirement or a laptop measurement.

## Shared GGML fix

The precision regression lives in the shared fork at
[`07b77b0112cc30861b49116b057bac33e819ee06`](https://github.com/betweentwomidnights/ggml/commit/07b77b0112cc30861b49116b057bac33e819ee06),
on `feature/heartcodec-cuda-f32-precision`. The project submodule pins this commit.

With TF32 explicitly disabled, a small F32 matrix multiply at K=512, M=64, N=5
still used MMF and lost operand precision. Its relative L2 error against a
double-precision accumulation reference was **7.45e-4** before the fix and
**7.70e-8** after. N=1, 25, and 64 also pass, as does CPU. The regression was run
against the original source and the fixed source with the same TF32 opt-out.

The fix routes explicitly requested F32/F32 matrix multiplies past the reduced
precision MMF path to the existing cuBLAS F32 path, retaining the F32 vector path.
The fork's existing `GGML_CUDA_TF32=0` opt-out is still required for strict reference
work. Default-precision matrix dispatch and quantized matrices retain their paths.

## Remaining work

The native SQ encoder and native roundtrip are next. RVQ, flow, query, full feature
frontends, Q4_K_M, and tokenizer parity are not implemented here yet. Vulkan/Metal
options exist, but those backends and Windows builds have not been validated.
No existing dependent repositories were bumped to this unmerged shared-fork branch.

The tests establish decoder numerical parity for these fixtures. They do not yet
establish streaming/chunked native SQ decoding, long-form perceptual quality,
quantization quality, or a production-ready full HeartCodec API.
