# Shared HeartCodec C++ / GGML Port — Implementation Brief

## Purpose

Build a **shared native HeartCodec implementation** intended to become reusable infrastructure across the Patch's `.cpp` music-model projects.

The immediate goal is **not** to implement MuLaCover or SpanSynth-Edit in full. The goal is to create a clean, well-tested HeartCodec foundation that makes both of those future projects substantially easier:

- `mulacover.cpp`
- `synthedit.cpp` / `spansynth.cpp` (name can be decided later)

The implementation should use the Patch's **shared GGML fork**:

- https://github.com/betweentwomidnights/ggml

This is a hard requirement.

If HeartCodec requires a GGML capability that is missing, awkward, slow, or backend-incomplete, prefer to:

1. implement or improve that capability in `betweentwomidnights/ggml`,
2. validate it across the relevant GGML backends,
3. version/tag the shared GGML fork appropriately,
4. bump the GGML dependency across our other `.cpp` projects as needed,
5. avoid project-local copies of generic kernels or runtime behavior that belong in GGML.

We are increasingly treating our GGML fork as common model infrastructure. HeartCodec should reinforce that architecture rather than become another isolated implementation.

---

# Why HeartCodec First

HeartCodec is now a useful shared dependency for at least two musician-facing model families we care about.

## MuLaCover / HeartMuLa path

HeartMuLa's released HeartCodec stack exposes a complete encoder/decoder path.

The public encoder:

- accepts mono or stereo floating-point waveform input,
- resamples internally to **24 kHz**,
- uses fixed **29.76-second / 714,240-sample chunks** at 24 kHz,
- extracts several aligned pretrained feature streams,
- fuses/downsamples them through a query transformer,
- produces **12.5 Hz** features,
- quantizes them with the decoder's shared residual vector quantizer,
- produces **8 RVQ token streams**.

The decoder:

- consumes the RVQ streams,
- obtains RVQ embeddings,
- upsamples the 12.5 Hz conditioning to **25 Hz**,
- runs a conditional flow-matching model that generates a **256-dimensional 25 Hz latent**,
- reshapes that latent as two 128-channel stereo latent streams,
- decodes each channel through the Base SQ waveform codec,
- emits **48 kHz stereo audio**.

This is the representation needed by HeartMuLa / MuLaCover.

## SpanSynth-Edit path

SpanSynth-Edit vendors the **Base SQ portion of HeartCodec** as a standalone frozen codec.

Its released configuration is:

- **48 kHz**
- causal
- mono
- **128 latent channels**
- effectively **25 latent frames/sec**
- scalar quantization using `round(9 * x) / 9`
- latent values bounded by `tanh`, therefore effectively 19 scalar levels over `[-1, 1]`

SpanSynth-Edit operates directly on these SQ latents rather than on the 12.5 Hz 8-codebook HeartMuLa representation.

This distinction is critical:

> Do not design `heartcodec.cpp` as only an RVQ-tokenizer wrapper.

The shared implementation should expose **both representation layers**.

---

# Primary Upstream References

The agent should clone/vendor these as sibling reference repositories during implementation.

## Official HeartMuLa / HeartCodec implementation

### heartlib

https://github.com/HeartMuLa/heartlib

Most important files:

- `src/heartlib/heartcodec/modeling_heartcodec.py`
- `src/heartlib/heartcodec/configuration_heartcodec.py`
- `src/heartlib/heartcodec/models/encoder.py`
- `src/heartlib/heartcodec/models/decoder.py`
- `src/heartlib/heartcodec/models/feature_extractors.py`
- `src/heartlib/heartcodec/models/query_transformer.py`
- `src/heartlib/heartcodec/models/flow_matching.py`
- `src/heartlib/heartcodec/models/transformer.py`
- `src/heartlib/heartcodec/models/sq_codec.py`

Treat current upstream code as authoritative for tensor shapes, chunking, padding, resampling, model math, and weight naming.

## SpanSynth-Edit

https://github.com/mimbres/spansynth-edit

Most important files:

- `spansynth/codec/config.py`
- `spansynth/codec/loader.py`
- `spansynth/codec/sq_codec.py`
- `spansynth/audio.py`
- `spansynth/sampling.py`

SpanSynth is especially valuable as an independent, cleaner description of the Base SQ codec used inside HeartCodec.

## MuLaCover

https://github.com/HeartMuLa/MuLaCover

Use this to understand how HeartCodec is consumed by the larger model and what API boundaries will make a future `mulacover.cpp` easy.

## Other native ports worth studying

Before reimplementing every operation from scratch, inspect existing native/non-PyTorch ports of HeartCodec components when available.

Examples worth locating and evaluating:

- HeartCodec / HeartMuLa MLX implementations
- native/Rust HeartCodec decoding implementations
- any community GGML ports that appear after this document was written

Do **not** blindly copy an implementation. Use them to cross-check layouts, padding, weight transforms, and edge cases.

---

# Repository Strategy

Preferred structure:

```text
heartcodec.cpp/
├── CMakeLists.txt
├── README.md
├── LICENSE / THIRD_PARTY_NOTICES as appropriate
├── cmake/
├── include/
│   └── heartcodec.h
├── src/
│   ├── heartcodec.cpp
│   ├── heartcodec-model.cpp
│   ├── heartcodec-sq.cpp
│   ├── heartcodec-rvq.cpp
│   ├── heartcodec-flow.cpp
│   ├── heartcodec-encoder.cpp
│   ├── heartcodec-features.cpp
│   ├── heartcodec-query.cpp
│   ├── heartcodec-audio.cpp
│   └── heartcodec-gguf.cpp
├── tools/
│   ├── heartcodec-cli.cpp
│   ├── convert-heartcodec-to-gguf.py
│   ├── inspect-heartcodec.py
│   └── compare-pytorch.py
├── tests/
│   ├── test-sq.cpp
│   ├── test-rvq.cpp
│   ├── test-flow.cpp
│   ├── test-query.cpp
│   └── test-end-to-end.cpp
├── reference/
│   └── scripts used to emit PyTorch golden tensors
└── ggml/
    └── submodule or dependency pointing at betweentwomidnights/ggml
```

A monolithic file should be avoided.

The codec has enough independent stages that we want narrow validation and future reuse.

---

# Dependency Rule: Shared GGML Fork

Use:

https://github.com/betweentwomidnights/ggml

Do not default to upstream `ggml-org/ggml` for the project dependency.

## When a missing GGML primitive is found

Before implementing a bespoke HeartCodec-only kernel, decide whether the operation is generic.

Examples:

- padding behavior,
- causal 1D convolution,
- transposed 1D convolution,
- grouped convolution,
- rotary embedding variants,
- attention/QK normalization,
- weight-normalized linear or convolution handling,
- PReLU,
- Snake activation,
- efficient interpolation / nearest-repeat,
- STFT or mel frontends,
- tensor rearrangement,
- batched overlap operations,
- backend-safe resampling helpers.

If the primitive is generally useful:

1. implement it in `betweentwomidnights/ggml`,
2. add CPU correctness coverage,
3. add/verify CUDA, Vulkan, Metal behavior where applicable,
4. avoid introducing a backend-specific assumption into HeartCodec,
5. bump the shared GGML version/submodule reference in:
   - `sa3.cpp`
   - `acestep.cpp`
   - `audiocraft.cpp`
   - and other dependent Patch `.cpp` repos as appropriate.

Do not let fear of dependency bumps force bad local architecture.

---

# Architectural Goal

The native library should expose HeartCodec at **three useful abstraction levels**.

## Level 1 — Base SQ codec

Needed by SpanSynth-Edit.

Suggested API shape:

```cpp
struct heartcodec_sq_latents {
    int64_t frames;
    int64_t channels; // 128
    std::vector<float> data;
};

heartcodec_sq_latents heartcodec_sq_encode(
    heartcodec_context * ctx,
    const float * mono_audio,
    int64_t num_samples,
    int sample_rate);

std::vector<float> heartcodec_sq_decode(
    heartcodec_context * ctx,
    const heartcodec_sq_latents & latents);

heartcodec_sq_latents heartcodec_sq_quantize(
    const heartcodec_sq_latents & latents);
```

Also consider exposing quantized integer levels directly:

```cpp
struct heartcodec_sq_codes {
    int64_t frames;
    int64_t channels; // 128
    std::vector<int8_t> levels; // e.g. -9 .. +9
};
```

That may be cleaner for a future SpanSynth model than storing repeated float values.

## Level 2 — HeartCodec RVQ tokenizer

Needed by HeartMuLa / MuLaCover.

Suggested API:

```cpp
struct heartcodec_tokens {
    int num_quantizers; // 8
    int64_t frames;     // 12.5 Hz
    std::vector<int32_t> codes; // [num_quantizers][frames]
};

heartcodec_tokens heartcodec_tokenize(
    heartcodec_context * ctx,
    const float * audio,
    int64_t num_samples,
    int sample_rate,
    int channels);

std::vector<float> heartcodec_detokenize(
    heartcodec_context * ctx,
    const heartcodec_tokens & tokens,
    const heartcodec_decode_params & params);
```

## Level 3 — Intermediate representations

Expose these intentionally for future research.

Examples:

```cpp
heartcodec_feature_bundle heartcodec_extract_features(...);
heartcodec_query_features heartcodec_query_encode(...);
heartcodec_rvq_embeddings heartcodec_dequantize_rvq(...);
heartcodec_flow_latents heartcodec_generate_flow_latents(...);
```

This may look unnecessary for a codec library, but it is strategically valuable.

Future work may want to:

- teacher-force HeartMuLa,
- use source HeartCodec representations as conditioning,
- seed generation with known token prefixes,
- perform latent editing,
- train alternative adapters,
- bypass expensive upstream stages,
- inspect semantic/acoustic representations,
- reuse the 25 Hz SQ latent independently.

Do not lock these representations behind a single `encode()` black box.

---

# Phase 0 — Reproduce Upstream Exactly in Python

Before the C++ port, create a small deterministic reference harness.

The harness should:

1. load the official HeartCodec models,
2. create deterministic input waveforms,
3. save intermediate tensors,
4. save final RVQ codes,
5. save Base SQ latents,
6. save flow outputs with fixed random seed,
7. save reconstructed audio,
8. record exact tensor shapes and dtypes.

Use test signals including:

- impulse,
- sine wave,
- stereo sine with different L/R frequencies,
- white noise with fixed seed,
- 1–5 second real music clip,
- 29.76 second clip,
- clip slightly shorter/longer than a chunk boundary.

Write outputs into something simple like `.npy` + JSON metadata.

These become golden references for every native stage.

---

# Phase 1 — Port the Base SQ Codec First

This is the highest-leverage initial milestone.

SpanSynth's implementation gives us a relatively clean isolated codec with:

- 48 kHz input/output
- causal Conv1d
- causal ConvTranspose1d
- PReLU
- residual units
- downsampling
- upsampling
- `tanh` final encoder latent
- scalar quantizer
- decoder

## SQ configuration

Current SpanSynth Base SQ configuration:

```text
sample_rate: 48000
num_bands: 1
init_channel: 64
downsample_factors: [3, 4, 4, 4, 5]
downsample_kernel_sizes: [6, 8, 8, 8, 10]
upsample_factors: [5, 4, 4, 4, 3]
upsample_kernel_sizes: [10, 8, 8, 8, 6]
latent_hidden_dim: 128
default_kernel_size: 7
delay_kernel_size: 5
num_samples: 2
causal: true
res_kernel_size: 7
mode: pre_proj
```

The effective latent rate is 25 Hz.

Reason:

```text
48,000 Hz
÷ 2 preprocessor pooling
÷ (3×4×4×4×5 = 960)
= 25 frames/sec
```

## Scalar quantizer

The reference quantizer is:

```python
round(9 * x) / 9
```

Because encoder output passes through `tanh`, values live approximately in:

```text
[-1, 1]
```

giving levels:

```text
-9/9, -8/9, ... 0 ... 8/9, 9/9
```

i.e. 19 scalar levels.

### Required milestone

Native:

```text
wav -> SQ encode -> SQ quantize -> SQ decode -> wav
```

must agree closely with Python.

Do this before touching HeartCodec's large feature encoder.

This milestone alone makes the shared library directly useful for a future SpanSynth port.

---

# Weight Norm Handling

Upstream uses PyTorch weight normalization extensively.

Do not attempt to reproduce PyTorch's parametrization framework at inference.

During conversion:

```text
effective_weight = g * v / ||v||
```

or whatever exact axis-specific parameterization applies for each layer.

Bake the effective weights into GGUF.

The C++ runtime should load normal inference weights whenever possible.

The conversion script should be responsible for:

- reconstructing weight-normalized tensors,
- verifying their shapes,
- optionally comparing forward outputs before/after baking.

---

# Phase 2 — Port RVQ

HeartCodec's decoder owns the shared residual vector quantizer.

Current configuration seen in upstream:

```text
dim: 512
codebook_size: 8192
codebook_dim: 32
num_quantizers: 8
```

The encoder deliberately receives the decoder's quantizer so the weights exist only once.

Mirror that concept in native code.

Suggested ownership:

```cpp
heartcodec_model
├── sq_codec
├── rvq
├── flow_decoder
├── query_encoder
└── feature_extractors
```

Do not duplicate RVQ codebooks across encoder and decoder objects.

## Tests

Given a known `[T, 512]` input:

- compare quantizer indices exactly,
- compare dequantized embeddings numerically,
- confirm code ordering,
- confirm `[8, T]` external token layout.

---

# Phase 3 — Port HeartCodec Flow Decoder

The RVQ tokens do not decode directly to waveform.

The public decoder:

1. dequantizes the RVQ indices,
2. projects the 512-d feature,
3. nearest-neighbor upsamples from 12.5 Hz to 25 Hz,
4. runs a conditional flow transformer,
5. produces 256-dimensional 25 Hz latents,
6. splits them into two 128-channel channel latents,
7. runs the Base SQ decoder independently for L/R,
8. crossfades long segments.

This stage is useful independently of the full encoder.

That means a valuable intermediate milestone is:

```text
official RVQ tokens
        ↓
native HeartCodec RVQ + flow + SQ decoder
        ↓
48 kHz stereo waveform
```

This lets us validate decoding before tackling the much more complex encoder frontend.

## Flow solver

Upstream uses simple fixed-step Euler integration.

Implement the exact reference solver first.

Only optimize or add alternate solvers after parity.

Parameters to expose:

```cpp
struct heartcodec_decode_params {
    int num_steps = 10;
    float guidance_scale = 1.25f;
    float segment_duration = 29.76f;
    uint64_t seed = ...;
};
```

Make random seeds deterministic and externally controllable.

---

# Phase 4 — Port the 12.5 Hz Query Encoder

The HeartCodec encoder produces fused 25 Hz features and downsamples them to 12.5 Hz.

The query encoder:

- dimension: 512
- layers: 6
- head dimension: 128
- interval: 2
- uses self attention
- Q/K LayerNorm
- partial RoPE
- LayerScale
- sigmoid GLU feed-forward blocks
- weight-normalized linear layers

Feature dimensions entering query fusion:

```text
acoustic: 2048
semantic: 2048
whisper: 1536
wavlm: 1536
```

The query encoder groups two 25 Hz frames plus a learned query token and emits one 512-d output every two frames.

Validate this independently using Python-exported feature tensors.

---

# Phase 5 — Port Feature Extraction

This is likely the largest and most annoying part of the encoder.

Current upstream combines aligned features from:

1. MusicFM-derived MuEncoder acoustic representation,
2. MusicFM-derived MuEncoder semantic representation,
3. Whisper encoder features,
4. WavLM features.

The upstream implementation is already stripped to inference-only versions.

The agent should inspect exact layer selections, resampling paths, padding, and output alignment in `feature_extractors.py`.

## Important

Do not guess at feature alignment.

Export deterministic upstream tensors for each stream and match:

- frame count,
- temporal offset,
- dtype,
- normalization,
- padding behavior,
- selected hidden layer.

## Potential GGML work

This phase may expose useful missing generic operators.

If so, implement them in the shared GGML fork.

Examples may include:

- STFT/mel operations,
- Conformer blocks,
- particular normalization paths,
- 2D convolution/subsampling,
- attention features needed by WavLM/Whisper,
- resampling utilities.

Before porting a whole pretrained frontend manually, also inspect whether existing GGML code from other projects can be reused.

Our ecosystem may already contain pieces for:

- Whisper,
- WavLM-like transformers,
- conformers,
- mel frontends.

Do not duplicate working common components.

---

# Phase 6 — Full Tokenizer Parity

Target:

```text
arbitrary mono/stereo audio
        ↓
native resample to 24 kHz
        ↓
feature extraction
        ↓
12.5 Hz query encoder
        ↓
RVQ
        ↓
8 token streams
```

The native code must reproduce the official tokenization behavior.

Important upstream behavior:

- mono input is duplicated to stereo,
- input is converted to CPU float32 before chunking in Python,
- 24 kHz internal encoder sample rate,
- 29.76-second chunks,
- final partial chunks are zero padded,
- upstream computes:
  `output_frames = int(duration_seconds * 12.5) + 1`
- there is a 240-sample alignment tail added before feature extraction,
- returned codes are shaped `[num_quantizers, frames]`.

Test exact token identity whenever numerically feasible.

If tiny floating point/backend differences cause boundary-level RVQ changes, quantify the rate and trace it to the first divergent tensor.

---

# Phase 7 — GGUF Format

Use GGUF for model packaging.

Do not preserve PyTorch checkpoint structure merely for convenience.

Suggested metadata:

```text
general.architecture = "heartcodec"
heartcodec.version = "vqv12"

heartcodec.sq.sample_rate = 48000
heartcodec.sq.latent_dim = 128
heartcodec.sq.frame_rate = 25

heartcodec.rvq.num_quantizers = 8
heartcodec.rvq.codebook_size = 8192
heartcodec.rvq.codebook_dim = 32
heartcodec.rvq.feature_dim = 512
heartcodec.rvq.frame_rate = 12.5

heartcodec.encoder.sample_rate = 24000
heartcodec.encoder.chunk_samples = 714240
heartcodec.encoder.chunk_seconds = 29.76
```

Consider either:

### Option A — one GGUF

```text
heartcodec-vqv12.gguf
```

Contains:

- Base SQ
- RVQ
- flow decoder
- query encoder
- all feature extractors

Pros: simple user experience.

Cons: larger load and less flexible.

### Option B — modular GGUFs

```text
heartcodec-sq.gguf
heartcodec-decoder.gguf
heartcodec-encoder.gguf
```

Where:

- `heartcodec-sq.gguf` = Base SQ codec
- `heartcodec-decoder.gguf` = RVQ + flow + SQ dependency metadata
- `heartcodec-encoder.gguf` = feature extractors + query encoder

This is attractive because:

- SpanSynth only needs SQ.
- MuLaCover needs full HeartCodec.
- future experiments may only need tokens or SQ latents.

Recommendation: **design the loader to support modular components**, even if the first converter also offers an all-in-one GGUF.

---

# Phase 8 — Backend Targets

At minimum:

- CPU
- CUDA
- Vulkan
- Metal

These align with our broader GGML strategy.

Do not allow the initial CUDA implementation to make the model architecture CUDA-specific.

## Backend validation matrix

For each backend, validate:

### SQ codec
- encode
- quantize
- decode

### RVQ
- code selection
- embedding reconstruction

### flow decoder
- one step
- deterministic fixed-seed multi-step decode

### full tokenizer
- feature extraction
- query encoder
- token output

### end to end
- audio -> tokens -> audio

Collect:

- wall-clock runtime
- peak memory
- numerical difference from CPU
- token mismatch rate
- reconstructed-audio metrics

---

# Phase 9 — CLI

The first native CLI should make HeartCodec independently useful.

Example:

```bash
heartcodec-cli encode input.wav --model heartcodec-vqv12.gguf --output tokens.bin
```

```bash
heartcodec-cli decode tokens.bin --model heartcodec-vqv12.gguf --output reconstructed.wav
```

```bash
heartcodec-cli roundtrip input.wav --model heartcodec-vqv12.gguf --output roundtrip.wav
```

```bash
heartcodec-cli sq-encode input.wav --model heartcodec-sq.gguf --output latent.bin
```

```bash
heartcodec-cli sq-decode latent.bin --model heartcodec-sq.gguf --output sq-roundtrip.wav
```

```bash
heartcodec-cli inspect input.wav --dump-intermediates ./debug/
```

Useful flags:

```text
--backend
--device
--threads
--seed
--steps
--guidance
--chunk-size
--no-quantize
--dump-tensors
```

---

# Phase 10 — API Design for Future `synthedit.cpp`

SpanSynth should not need to know about HeartMuLa's RVQ tokenizer.

It should depend only on the Base SQ surface:

```cpp
auto latent = heartcodec_sq_encode(...);
auto codes = heartcodec_sq_quantize(latent);

// SpanSynth DiT edits codes / latents here.

auto output = heartcodec_sq_decode(...);
```

A future SpanSynth port will likely need:

- 128-channel latents
- 25 Hz timing
- scalar quantization
- clean conversion between float latents and integer scalar levels
- efficient batch decode
- ability to keep source latents resident on GPU

Avoid unnecessary host round trips.

Potential future interface:

```cpp
heartcodec_tensor_view heartcodec_sq_encode_device(...);
void heartcodec_sq_decode_device(...);
```

so SpanSynth can pass GGML tensors directly without CPU copies.

---

# Phase 11 — API Design for Future `mulacover.cpp`

MuLaCover will want:

```text
source audio
    ↓
HeartCodec tokenizer
    ↓
8 RVQ code streams @ 12.5 Hz
```

and later:

```text
generated HeartCodec tokens
    ↓
HeartCodec detokenizer
    ↓
48 kHz stereo audio
```

Expose token structures in a way compatible with an autoregressive model:

```cpp
struct heartcodec_tokens_view {
    const int32_t * data;
    int quantizers;
    int64_t frames;
    int64_t stride_q;
    int64_t stride_t;
};
```

Future MuLaCover work may also want:

- token prefix injection,
- direct dequantized RVQ embeddings,
- teacher-forced hidden-state experiments,
- partial token replacement.

Do not bury token layout in private code.

---

# Phase 12 — Conversion Tooling

Write a Python conversion script.

Responsibilities:

1. load official safetensors / HF model,
2. validate expected architecture/config,
3. fold weight norm where safe,
4. transpose/reshape weights into native convention,
5. write GGUF,
6. include architecture metadata,
7. optionally quantize supported tensors,
8. produce a conversion report,
9. fail loudly on unknown/missing/unexpected tensors.

Example:

```bash
python tools/convert-heartcodec-to-gguf.py \
  --encoder HeartMuLa/HeartCodec-Encoder \
  --decoder HeartMuLa/HeartCodec-Decoder \
  --output heartcodec-vqv12-f16.gguf
```

The exact HF repo names should be read from current official documentation rather than hardcoded from this document.

---

# Quantization Strategy

Do not start by quantizing aggressively.

Recommended order:

1. F32 / exact reference where needed
2. F16
3. BF16 where backend supports it
4. Q8 / Q6 / Q5 / Q4 selectively

The Base SQ convolutional codec may respond differently to quantization than transformers.

The feature extractors may also be more or less sensitive than the flow model.

Quantize per subsystem and measure:

- RVQ token agreement
- reconstruction quality
- spectral error
- latency
- memory

Tokenization is especially sensitive because small representation errors can cross codebook decision boundaries.

A decoder can tolerate approximate audio.

A tokenizer whose discrete codes change massively may break compatibility with HeartMuLa.

Therefore prioritize **token identity / distribution stability** over size during encoder quantization work.

---

# Testing Philosophy

Every subsystem should have a deterministic Python golden test.

Do not debug the entire model by listening to final audio.

## Suggested tolerance hierarchy

### Exact or near-exact
- tensor shapes
- padding lengths
- codebook index layout
- scalar quantizer integer levels
- metadata

### Tight numerical tolerance
- SQ latent values
- query encoder outputs
- dequantized RVQ vectors
- individual flow steps on CPU/F32

### Perceptual/statistical tolerance
- final decoded waveform across reduced-precision GPU backends

When a mismatch occurs:

```text
waveform
  ↓
preprocessing
  ↓
feature extractor
  ↓
query encoder
  ↓
RVQ
  ↓
flow
  ↓
SQ decode
```

Find the **first** divergent stage.

---

# Performance Work

Do not optimize before parity, but structure the port so obvious optimizations remain possible.

Important targets:

## SQ codec
- fuse causal padding where possible
- avoid repeated allocations
- keep persistent convolution buffers
- reuse graphs for fixed frame lengths
- device-resident encode/decode

## Query transformer
- graph reuse
- KV temporary reuse
- efficient RoPE
- fused GLU if GGML supports it

## Flow decoder
- prebuild/reuse step graph
- avoid rebuilding constant RVQ-conditioning graph each Euler step
- reuse CFG buffers
- allow CFG disable
- support batch decode

## Full encoder
- chunk streaming
- avoid placing the entire recording on GPU
- reuse feature-extractor graphs for the fixed 29.76 s chunk size
- pipeline CPU resampling and GPU inference where useful

---

# Shared GGML Opportunities

HeartCodec may identify improvements that benefit our whole `.cpp` stack.

Potential areas:

- reusable fixed-shape graph caching
- host arena reduction
- convolution workspace reuse
- transposed-convolution performance
- Metal/Vulkan convolution kernels
- generic audio resampling
- STFT/mel primitives
- backend-safe causal padding
- PReLU / Snake fusion
- model-component load/unload API
- memory mapping for modular GGUF files

If an improvement is generic, upstream it into `betweentwomidnights/ggml`.

Then bump our dependent model projects.

This port should be treated as another stress test of our GGML-native music stack.

---

# Licensing Audit

Before redistributing:

1. record the license of every official HeartCodec checkpoint,
2. record the license of heartlib source,
3. preserve required notices from MusicFM-derived frontend code,
4. inspect Whisper/WavLM model licensing,
5. inspect SpanSynth's copied codec code and training/checkpoint terms,
6. keep model-weight licensing separate from our native runtime code.

Do not assume that an Apache-2.0 source repository means all model weights are commercially usable.

Create:

```text
THIRD_PARTY_NOTICES.md
```

early rather than after the port is complete.

---

# Recommended Implementation Order

## Milestone 1 — Repository + converter skeleton

Deliver:

- CMake
- shared GGML fork wired in
- GGUF loader
- Python tensor-dump harness
- converter skeleton
- deterministic test assets

## Milestone 2 — Base SQ decoder

Deliver:

```text
known 128xT SQ latent -> native 48 kHz waveform
```

Match SpanSynth/HeartCodec Python.

## Milestone 3 — Base SQ encoder + scalar quantizer

Deliver:

```text
48 kHz waveform -> 128xT latent -> quantized latent -> decode
```

This is the first major useful milestone for future SpanSynth work.

## Milestone 4 — RVQ + HeartCodec flow decoder

Deliver:

```text
known 8xT HeartCodec tokens -> 48 kHz stereo waveform
```

At this point a future MuLaCover generator can use the native decoder.

## Milestone 5 — Query encoder

Deliver parity from exported aligned features to 12.5 Hz 512-d representation.

## Milestone 6 — Full feature frontend

Port:

- MusicFM-derived MuEncoder
- Whisper path
- WavLM path
- alignment logic

## Milestone 7 — Native tokenizer

Deliver:

```text
audio -> 8 HeartCodec token streams
```

## Milestone 8 — Multi-backend validation

CPU / CUDA / Vulkan / Metal.

## Milestone 9 — Quantization and performance

Only after parity.

---

# Stop Conditions / Things Not to Do

Do not:

- start by implementing MuLaCover itself,
- start by implementing SpanSynth's DiT,
- make HeartCodec dependent on either downstream model,
- create a private project-specific GGML fork,
- copy generic GGML kernels into this repo if they belong in shared GGML,
- collapse RVQ and SQ representations into one vague "latent" type,
- assume 12.5 Hz HeartMuLa tokens and 25 Hz SpanSynth latents are interchangeable,
- optimize away reference behavior before parity,
- treat final listening tests as sufficient validation,
- hardcode CUDA-only logic into model architecture,
- quantize the tokenizer aggressively before measuring token stability.

---

# First Agent Tasks

The PC agent should begin with these concrete actions:

1. Clone:
   - `betweentwomidnights/ggml`
   - `HeartMuLa/heartlib`
   - `mimbres/spansynth-edit`
   - `HeartMuLa/MuLaCover`

2. Create `heartcodec.cpp` as a sibling repo/project.

3. Inspect the exact current model/checkpoint names and download requirements.

4. Build a Python reference script that exports:
   - SQ encode latent
   - SQ quantized latent
   - SQ reconstruction
   - RVQ indices
   - RVQ embeddings
   - one flow-decoder step
   - final HeartCodec decode

5. Port only the Base SQ decoder first.

6. Compare its output numerically against Python.

7. Port the Base SQ encoder and scalar quantizer.

8. Once Base SQ roundtrip parity is solid, commit/tag that milestone before moving on.

9. Only then begin RVQ/flow work.

10. Whenever a missing operator appears, evaluate whether it belongs in the shared `betweentwomidnights/ggml` fork before implementing locally.

---

# Definition of Success

The project is successful when we have one shared native codec library that can support both of these future architectures without duplication:

```text
                  ┌─────────────────────────┐
                  │     heartcodec.cpp      │
                  │                         │
audio ───────────►│ Base SQ 25 Hz / 128-d  │◄──────── SpanSynth-Edit
                  │                         │
audio ───────────►│ RVQ 12.5 Hz / 8 streams│◄──────── MuLaCover / HeartMuLa
                  │                         │
tokens ──────────►│ flow + SQ decoder       │────────► 48 kHz stereo
                  └─────────────────────────┘
                              │
                              ▼
                  betweentwomidnights/ggml
```

The best outcome is not merely "HeartCodec works."

The best outcome is:

> HeartCodec becomes another reusable native building block in the Patch's GGML music-model stack, and any generic runtime work required to make it fast improves our other `.cpp` projects at the same time.

