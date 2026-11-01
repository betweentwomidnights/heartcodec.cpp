# Development environment decision

Assessment date: October 31, 2026 (America/Phoenix).

This document records the initial assessment. For implementation results completed
November 1, see [SQ decoder parity](SQ_PARITY.md) and the [Docker workflow](SPARK_WORKFLOW.md).
The [laptop development report](LAPTOP_DEVELOPMENT.md) supersedes the development
location recommendation for the completed SQ primitives. The later
[RVQ/flow and stereo measurements](FLOW_AND_STEREO.md) also establish complete
token-to-audio native decoding on the laptop through staged component loading.
The full audio tokenizer and unquantized MuLaCover reference remain larger-memory
Spark workloads; the initial resident-weight estimate below is historical.

## Decision

Use **DGX Spark `spark-c7ca` as the reference/parity development environment** for
the complete HeartCodec pipeline. Keep Windows/RTX 5070 Laptop for isolated SQ
experiments, CPU/CUDA/Vulkan primitive validation, and eventual deployment testing.

Full resident FP32 HeartCodec weights exceed the laptop GPU's physical memory.
CPU execution or deliberate component offload can avoid that limit, but the Spark
is the practical choice for generating complete reference fixtures without changing
upstream's numerical behavior. This is a capacity decision, not a measured OOM or
an inference benchmark. No model forward pass was run during this assessment.

## Observed hardware and software

Read-only local and SSH probes succeeded:

| Machine | Observation |
| --- | --- |
| Laptop | RTX 5070 Laptop GPU, 8,151 MiB total, 5,100 MiB free at probe time; NVIDIA driver 616.56 |
| Laptop host | About 63.4 GiB RAM total, 32.1 GiB free at probe time; Python 3.11.9 |
| Existing SA3 Python environment | `C:\dev\sa3.cpp\.venv`: PyTorch `2.9.1+cpu`, CUDA unavailable |
| Spark SSH | `ssh -o BatchMode=yes spark-c7ca` succeeds; aarch64 / NVIDIA GB10 |
| Spark memory | 121 GiB system/unified memory total, 71 GiB available at probe time |
| Spark storage | About 1.1 TiB available in the home filesystem |
| Spark toolchain | CMake available; `/usr/local/cuda/bin/nvcc` is CUDA 13.0; driver 580.178.04 |
| Spark Python | System `python3` has no torch, safetensors, or huggingface_hub |

GB10 reports GPU memory as N/A through the queried `nvidia-smi` fields; the memory
number above is Linux's available unified/system memory, not free dedicated VRAM.
Existing Docker music services use the Spark. No service was stopped, no container
was modified, and no reference environment was installed there. A service's
`/opt/sa3-venv` process path is inside its container, not a host Python environment.

## Exact checkpoint storage inventory

The [machine-readable audit](checkpoint-audit.json) was generated from pinned
official HF metadata and safetensors headers, reading 371,168 header bytes total
across all four models. Model tensor payloads were not downloaded.

All numbers below are **GiB (2^30 bytes)** and exclude activations and workspace:

| Component | FP32 tensor bytes |
| --- | ---: |
| Base SQ decoder | 0.312 |
| Base SQ encoder | 0.314 |
| Flow estimator | 5.539 |
| Shared RVQ | 0.016 |
| Remaining flow conditioning | 0.001 |
| Complete decoder checkpoint, including both SQ halves | **6.182** |
| MusicFM-derived MuEncoder frontend | 1.210 |
| WavLM frontend | 0.352 |
| Whisper frontend | 0.328 |
| Query encoder | 0.111 |
| Complete separate encoder checkpoint | **2.001** |
| Complete HeartCodec encoder + decoder | **8.183** |

The laptop's physical VRAM is 7.960 GiB, with 4.980 GiB free at the observed moment.
Thus the upstream complete FP32 load cannot reside entirely on that GPU even if
all currently used VRAM were released. The decoder alone exceeds the observed
free VRAM. SQ weights leave ample room for short isolated tests; actual peak
memory for SQ activations/convolutions still needs measurement.

The full MuLaCover checkpoint independently stores **8.148 GiB of BF16 tensors**.
It too exceeds this GPU's physical VRAM before generation caches and buffers.
The style model is Qwen3-Embedding-0.6B: 1.110 GiB in its stored BF16 form, or
2.219 GiB when loaded in the FP32 dtype shown in MuLaCover's example.

Sources: [HeartCodec reconstruction example](https://github.com/HeartMuLa/heartlib/blob/5858ca8f1ffe58d62be7c007ea55f1033c5ca456/examples/run_music_reconstruction.py),
[decoder weights](https://huggingface.co/HeartMuLa/HeartCodec-oss-20260123/tree/f889dab0532cfa4bf459f2a3367eb6d346b8eeda),
[encoder weights](https://huggingface.co/HeartMuLa/HeartCodec-oss-encoder/tree/e6020510612d6653b73d6cd54393ef80929e3c81),
[MuLaCover weights](https://huggingface.co/HeartMuLa/MuLaCover/tree/bbbaef2b31835c2ef5172ff528dbe325230d46fb).

## What Q4_K_M can reasonably mean here

Q4_K_M is a mixed tensor policy using Q4_K and Q6_K, not a PyTorch inference dtype
or a uniform four bits per parameter. SA3's `tools/sa3-quantize.cpp` selects
eligible two-dimensional weights, promotes selected critical tensors to Q6_K,
and leaves ineligible tensors in their original form. HeartCodec will need its
own validated tensor mapping and promotion policy.

The inspector provides two **hypothetical storage endpoints**: all eligible
rank-2 weights in Q4_K versus all in Q6_K, with everything else FP32. It counts
GGML block overhead (144 or 210 bytes per 256 elements), requires 256-element row
alignment, and does not quantize rank-3 convolutions. These are pre-conversion
scenarios, not converted model sizes or validated eligibility guarantees.

| Model | Q4_K rank-2 / rest FP32 | Q6_K rank-2 / rest FP32 |
| --- | ---: | ---: |
| HeartCodec decoder checkpoint | 1.521 GiB | 1.871 GiB |
| HeartCodec encoder checkpoint | 0.498 GiB | 0.611 GiB |
| MuLaCover generator | 2.858 GiB | 3.866 GiB |
| Qwen style model | 0.312 GiB | 0.455 GiB |

These sizes support investigating **staged component residency** on 8 GB. They
do not prove complete generation fits or preserves quality. Keep SQ, small
normalizations, and RVQ precision under independent control. Establish FP32 and
F16 behavior first; measure feature error, token disagreement, and reconstruction
error before choosing the tokenizer's quantization. A lower-bit tokenizer can
change discrete codes even when its continuous feature error looks small.

Upstream's tokenizer pads short audio to a 29.76-second chunk. The detokenizer
likewise uses its default 29.76-second window for short token sequences. A one
second public-API test therefore does not necessarily mean one second of working
memory. Short isolated SQ tests are a different workload.

## SA3 patterns and reference pins

Read the local SA3 CMake setup, embedding/C ABI guidance, shared-fork notes,
PyTorch dump/feed parity workflow, and native quantizer before defining new APIs.
Continue the same patterns:

- One reusable native library beneath CLI and downstream hosts; explicit context
  ownership, status returns, and library-owned result release across DLLs.
- Versioned, size-prefixed C ABI when exposing a stable host interface.
- GGUF metadata and a strict converter; bake PyTorch weight normalization.
- Explicit backend toggles; CPU reference plus CUDA, Vulkan, and Metal coverage.
- Dump intermediate tensors; feed identical inputs into each native stage so
  stochastic flow divergence does not hide the first incorrect operation.
- Serialize actual random input tensors. A seed alone is insufficient to assume
  identical random draws across devices, library versions, or CPU/CUDA paths.
- Use the shared GGML fork, with generic operator changes and tests there.
- Quantization is a separately measured stage after numerical parity.

Source checkouts created as siblings under `C:\dev\_heartcodec_refs`:

| Reference | Revision |
| --- | --- |
| heartlib | `5858ca8f1ffe58d62be7c007ea55f1033c5ca456` |
| spansynth-edit | `f593bf21e48cc4aa836002a3834cfae42e8d4254` |
| MuLaCover | `e27fa7fa0da7c9817879cc6210cc5693f9327946` |
| Local sa3.cpp HEAD | `c078e01e0e694408e0434d39c31777892d972948` |

The existing shared `C:\dev\ggml` checkout has the required fork origin and a
clean working tree at `4ad3b30bfe452efc5f4f9c0e111e017d9970292a`. SA3's actual
`ggml/` checkout is at that same commit, while its recorded HEAD gitlink is
`b77ae790e628270af46a8af1eca38441c23f1252`. Record this distinction when selecting
the native project's pin; do not silently rewrite either existing checkout.
No GGML dependency has been added to this assessment-only project yet.

## Next parity milestone on Spark

1. Create a dedicated project workspace and isolated ARM64 Python environment.
   Verify a CUDA-enabled PyTorch build can execute on GB10 before installing the
   pinned heartlib dependencies. Preserve existing production service environments.
2. Clone the references at the revisions above and fetch the pinned encoder and
   decoder checkpoint revisions in `checkpoint-audit.json`. Initially only the
   decoder checkpoint is needed for Base SQ. Avoid downloading MuLaCover itself
   until the shared codec milestones are ready.
3. Strictly extract `scalar_model.*` tensors, preserving effective weight-norm
   axes. Verify against both heartlib and SpanSynth before using either as a
   substitute reference. Decode applies `round(9*x)/9` internally too.
4. Export deterministic input/output and intermediate `.npy` tensors plus JSON
   metadata for SQ encode, quantize, and decode. Cover impulse, sine, seeded
   noise, stereo channel separation, short clips, and chunk boundary lengths.
5. Measure CUDA allocated/reserved peaks and wall time for load, SQ decode,
   SQ encode, RVQ, one teacher-forced flow step, ten-step flow, each feature
   frontend, and the full 29.76-second tokenizer. Record memory already used
   by other processes separately from this test's peak.
6. Wire CMake and GGUF to the shared fork, implement only the SQ decoder first,
   and match the saved Python inputs/outputs. Next add SQ encoder and scalar
   quantizer. Only then advance to RVQ/flow, query, and features.
7. Transfer reference fixtures to Windows for CPU/CUDA/Vulkan checks. Validate
   Q4_K_M and staged residency against the unquantized baseline on the laptop.

This assessment establishes where to do parity work. It does not claim any codec
stage, quantization quality, backend parity, or peak-runtime-memory test has passed.
