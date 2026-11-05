# Native waveform-to-RVQ encoder (experimental)

The complete encoder now runs in C++: borrowed PCM -> resampling and chunking
-> MuEncoder, Whisper and WavLM features -> fusion/query transformer -> the
original decoder's eight RVQ codebooks. It uses `heartcodec::heartcodec` and the
existing shared GGML target. No Python, Torch, audio subprocess or dependency
installation is involved in native encoding.

**Validation is incomplete.** Native CPU output matches all 8,000 reference
codes across the three raw-audio fixtures below. CUDA matches two fixtures but
changes four codes in one frame of the 50-second fixture. Strict intermediate
activation gates also remain open on both backends. Keep FP32 weights and CPU
as the reference-compatible baseline for these fixtures; CUDA is experimental.
Do not interpret final-code agreement as a complete numerical-parity pass.

## Models and fixed reference

- HeartLib: `5858ca8f1ffe58d62be7c007ea55f1033c5ca456`.
- Encoder: `HeartMuLa/HeartCodec-oss-encoder`, revision
  `e6020510612d6653b73d6cd54393ef80929e3c81`, safetensors SHA256
  `19b3c0284fecac70983731433a213ad20eb08d096be3b9a287078171de0d4e79`.
- Shared RVQ: original decoder revision
  `f889dab0532cfa4bf459f2a3367eb6d346b8eeda`. The encoder GGUF contains its
  original project-in weights and eight codebooks, not a new trained quantizer.
- Native GGUF: architecture `heartcodec-encoder`, format 1, 983 strict F32
  tensors, 2,157,157,152 bytes, SHA256
  `b7cd97fb1cc48764070851777ed00c5b56540730ae7372ad39e2d8c92486c297`.
  Weight-normalized linear/convolution weights are materialized from the
  strictly loaded checkpoint. Training counters and decoder-only RVQ output
  projection are omitted. No encoder quantization is implemented.

Python is restricted to conversion, offline comparison and pinned CPU oracles
in `heartcodec-reference:20261101` on Spark. That image is
`sha256:70947194330272d2aed83dfa51923165ef013a32b0a03f9be8afb2d65957524b`.
The native laptop fixtures require only the frozen GGUF, PCM and code files.
Model weights, audio and tensor dumps are excluded from Git.

The coordinated GGML submodule remains `b77ae79`. Windows CPU validation uses
that baseline. CUDA validation reuses MuLaCover's already-authorized single
GGML target at isolated fix `0f2ed045`, not a changed HeartCodec pin. Standalone
CUDA builds on unmodified `b77ae79` retain its known F32 source-precision issue.
No new GGML changes, sibling bumps or ggml-bump-check runs are made here.

## Audio and representation contract

The separate negotiated capability is in `include/heartcodec_encoder.h`:

```c
const heartcodec_encoder_api_v1 * api = heartcodec_get_encoder_api(1);
```

Frozen core, SQ, flow and RNG layouts remain unchanged. Use the existing core
config initializer to select the encoder GGUF, explicit backend and CPU thread
count, then `api->context_create`. Results are library-owned frame-major
`int32_t[frames,8]`; SQ's 128-channel scalar latents are a different format.

`heartcodec_waveform_v1` borrows finite F32 mono/stereo PCM with explicit element
strides. For interleaved stereo, channel/sample strides are 1/2. For planar
stereo, they are num_samples/1. Mono is duplicated. Supported sample rates are
16,000; 22,050; 24,000; 32,000; 44,100; 48,000; 88,200; 96,000 and 192,000 Hz.
The API does not clip, normalize or time-stretch the source.

The full available waveform is resampled to 24 kHz before splitting into
714,240-sample (29.76-second) chunks. Each partial final chunk is zero-padded;
each encoder chunk also receives the reference's 240-sample alignment tail.
The input conversion uses Torch-functional-style F32-built sinc kernels. The
internal cached 24k->16k transform preserves its separately CPU/F64-built,
F32-stored kernel. Frozen Hann windows and mel filters are used. MuEncoder's
AmplitudeToDB has no `top_db` floor; Whisper retains its separate log-mel floor.

The capability returns **complete original-clock 80-ms frames only**:

```text
frames = floor(num_samples * 25 / (2 * sample_rate))
```

At least one complete frame is required. A partial tail is discarded. This
matches the continuation harness's guide policy; the official wrapper's extra
partial-frame rule is deliberately not exposed as another complete frame.
`resampled_samples` reports the 24-kHz count before padding. Do not infer this
framing policy from SQ sample/frame counts.

Calls on one encoder context are serialized and run outside realtime audio
callbacks. Initialize/free results through this same capability. Results
survive context destruction and must be freed before unloading the library.
Invalid PCM, rate/stride overflow, short structs or occupied results are rejected
without mutating result buffers. Source PCM is never modified. A requested
backend remains explicit; there is no silent GPU-to-CPU inference fallback.

Create/encode/release the encoder, retain its owned code buffer while priming
MuLaCover, then release it when no longer needed. Stage generator and decoder
models separately as in the accepted continuation path. Host duration policy
such as yue2's `natural_max_seconds` belongs to the generation job: a ceiling
that permits early EOS, mapped to MuLaCover's integer new-frame/cache budget.
It is separate from encoder chunk duration and guide length.

## Native CLI

Build standalone CPU on the unchanged coordinated pin:

```powershell
./scripts/build-windows.ps1 -Backend cpu
./build-windows-cpu/bin/Release/heartcodec-encode.exe encode `
  --model models/heartcodec-encoder-f32-v2.gguf `
  --input input.wav --output refdata/my-native-encoding `
  --backend cpu --threads 8
```

The CLI accepts mono/stereo PCM16/24/32 or IEEE F32 WAV, including the supported
WAVEFORMATEXTENSIBLE forms. It writes `codes.i32` and `run.json`. Use a fresh
output directory; `--dump 1` additionally retains per-chunk intermediate tensors.
The output codes can be passed directly to native `mulacover-generate` as
`--audio-prefix`. The CLI exercises result ownership after encoder-context
destruction. `query`, `rvq`, `preprocess`, `features`, `mu`, `whisper`, and `wavlm`
modes are diagnostics for prepared reference fixtures in static tool builds.
The DLL tool's `encode` command uses only the negotiated public C capability;
internal diagnostic C++ symbols are not exported as another ABI.

For CUDA, use the existing MuLaCover development-target build; its
`build-windows-development-cuda/bin/Release/heartcodec-encode.exe` links this
same HeartCodec implementation and the single allowed GGML target. Keep
`GGML_CUDA_TF32=0`, `GGML_CUDA_CUBLAS_COMPUTE_TYPE=auto`, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` for the recorded comparisons.

## Measured fixtures and open gates

| Raw input | RVQ frames | CPU changed codes | CUDA changed codes |
| --- | ---: | ---: | ---: |
| Source prefix, 29.76 s, stereo 44.1 kHz | 372 | 0 / 2,976 | 0 / 2,976 |
| Source prefix, 50 s, stereo 44.1 kHz, two chunks | 625 | 0 / 5,000 | 4 / 5,000 |
| Derived 0.24 s excerpt, mono F32 48 kHz | 3 | 0 / 24 | 0 / 24 |

The first two CUDA runs take about 6 / 9 seconds including model loading
and debug dumping. CPU takes about 26 / 49 seconds. These are diagnostic
run timings, not isolated host-worker latency benchmarks. The full F32 encoder
fits the 8GB laptop: library-owned weight and maximum graph buffers are about
2.16 GB each. Those counters exclude backend/context/workspace and desktop
memory; they are not a measured total VRAM peak.

Before waveform integration, native query/RVQ from frozen synthetic and real
feature streams matches all 6,080 IDs on CPU and CUDA. Full feature streams and
raw-audio runs are evaluated separately at unchanged maximum absolute error
1e-3, relative L2 1e-4, and exact code IDs. Several Whisper activations exceed
the flat absolute threshold. Using the checkpoint's serialized rotary frequencies
closes the earlier MuEncoder failures without changing the gates.
Reports retain `passed: false`; final-code agreement does not waive these gates.

In the CUDA 50-second run, frame 259 first changes at residual level 4. Its
reference winning score margin is `5.960464477539063e-8`, with subsequent code
changes at levels 5--7. Cross tests isolate the query backend: reference
features -> CUDA query changes the same four codes; CUDA features -> CPU query
matches. CPU query -> CUDA RVQ also matches. This is measured boundary evidence,
not a rounding exception or a passed CUDA parity claim. Keep the CUDA mismatch
open and do not quantize or optimize away this diagnostic.

A separate stock Torch CUDA neural-stage oracle on Spark produces the same
four changes against the CPU oracle. Native CUDA matches all 5,000 of that
oracle's IDs. This comparison uses frozen CPU frontend tensors and effective
weights materialized on CPU before moving the neural modules to CUDA. It is
not a stock end-to-end CUDA tokenize run: the reference image's complex STFT
Jiterator path does not support GB10. Strict Whisper activation failures remain
even against the CUDA-stage oracle; neither backend agreement nor the measured
CPU/CUDA differences waive a gate. See the boundary diagnostics report.

The [Whisper investigation](WHISPER_PARITY.md) adds per-layer/projection tests,
independent F64 controls and a separately labeled candidate accuracy contract.
It preserves the original failed limits and exact-code checks. No numerical
runtime experiment or new GGML patch from that investigation is adopted.

The new CPU encoder's 372-frame guide was also passed into the native Q4
continuation/decoder. All 8,760 output codes, full 87.6-second WAV and 57.84-second
new-only WAV match the user's accepted native listening baseline byte-for-byte.
Text/Qwen style remains a cached preset. This is not a native custom-style
frontend, nor does it resolve the existing strict upstream decoder waveform gap.

Reports live under `docs/validation/encoder-*.json`. `tools/compare-encoder.py`
checks golden file hashes and emits nonzero status for failed tensor/code gates.
CPU static tests, CPU DLL tests, CUDA integration tests and Spark Docker native
build/tests cover negotiation, malformed input, output ownership and unchanged
core APIs. See the compact validation summary for exact build/check counts.

## Reference workflow

On Spark, these actions remain container-only and network-disabled:

```sh
bash scripts/spark.sh reference-encoder refdata/encoder-cpu-v3
bash scripts/spark.sh reference-encoder50 refdata/encoder50-cpu-v1
bash scripts/spark.sh convert-encoder models/heartcodec-encoder-f32-v2.gguf
bash scripts/spark.sh reference-encoder-cuda-stages refdata/encoder50-cpu-v1 refdata/encoder50-cuda-stages-v1
```

The helper mounts encoder checkpoint cache and reference sources read-only.
`HEARTCODEC_ENCODER_INPUT`, `HEARTCODEC_ENCODER_AUDIO`, and
`HEARTCODEC_ENCODER_EXPECTED_CODES` support additional frozen inputs; an empty
expected-code path requests a fresh independent oracle. No host Python or
production container is used. Keep complete available input boundaries frozen:
the 50-second encoding changes five earlier codes compared with the shorter
encoding, so arbitrary prefixes cannot be assumed invariant to more context.
