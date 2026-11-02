# Torch CUDA Philox capability

`include/heartcodec_rng.h` adds a separately negotiated v1 RNG capability. The
frozen core/flow structs, tables and legacy seed behavior are unchanged. A CUDA
build provides contiguous F32 normal and exponential(lambda=1) draws matching
the pinned Torch CUDA distribution kernels. A CPU-only build reports unavailable;
there is no silent RNG fallback or PyTorch runtime dependency.

The Philox4x32-10 rounds follow sibling sa3.cpp `src/rng.h` at
`95508283b303164aa6a0b5a5140dfe79d4c489b3`. ACE-Step's BF16-oriented implementation
was consulted; this capability instead matches the F32 kernels used by MuLaCover.
CUDA math intrinsics reproduce the normal transform and Torch's fast exponential
log/clamp. See `THIRD_PARTY_NOTICES.md` for source attribution and license text.
No cuRAND headers or library are needed, and no shared GGML source/pin changes.

## Ownership and stream semantics

Negotiate `heartcodec_get_rng_api(1)`, check the table's size/version, initialize
both config and result using their size fields, then call `generate`. The result
owns host F32 memory in `[draws,count]` order; release it with that capability's
`result_free`. It survives model/context unloading and can be borrowed by existing
model APIs. Calls are independent: the caller owns seed/offset state. Errors and
occupied-output rejection leave the result unchanged; free before reuse.

`count` is elements **per original Torch kernel call**, not the flattened total.
`draws` repeats that shape with consecutive offsets. Offsets count Philox 32-bit
outputs and must be divisible by four. For grid `g`, stride `s=256*g`, a call
advances by `4*ceil(count/(4*s))`. Empty shapes consume no offset. F32 is the only
supported dtype; arbitrary TensorIterator layouts and BF16 kernels are not implied.

`device` selects the actual CUDA ordinal. `max_blocks=0` queries that GPU's SM
count and maximum threads per SM. A nonzero cap emulates the **reference GPU's**
grid mapping, independently of the execution device. Torch seeds alone are not
sufficient across GPU models for large tensors. Spark GB10 uses cap 288; the
RTX 5070 Laptop reports 216. Matching Spark draws on the laptop requires 288.

## MuLaCover / HeartCodec use

For each MuLaCover frame, generate eight exponential calls of 8,197 elements.
They consume four offsets per head on the tested devices, or 32 per frame. Pass
the result through `mulacover_sampling_config_v1.exponential_noise`, keeping it
alive for that sample call. Continue with `result.next_offset`, or precompute the
requested frame budget once and index each `[8,8197]` frame. Reset/retry can reuse
the same buffer and offset without advancing an implicit global generator.

For codec windows of 372 codes, generate consecutive normal calls of
`2*372*256=190464` elements, resetting the seed/offset independently for the
decoder. Pass the buffer through existing `heartcodec_flow_config_v1.noise` and
set `noise_segments` to the exact padded window count. Do not reseed each window.
The upstream codec's CPU random context fillers are masked out; the effective
flow noise comes from these CUDA calls. A model-free oracle exercises the real
upstream `detokenize` and `inference_codes` methods to verify this call graph.

## Validation

MuLaCover's `reference/export-rng.py` freezes Torch 2.7.1+cu128 oracles in Spark
Docker. `tools/compare-rng.py` requires bit-identical values and offsets. All 31
cases pass on Spark (CUDA 13 native build) and the laptop (CUDA 12.8), including
high-bit seeds/offsets, tails, capped multi-trip loops and all 7,000 head draws
for a 70-second request: 85,205,452 values per device. The actual three-window
codec call graph adds 571,392 bit-exact normal values on the laptop.

C/C++ ABI tests cover negotiation, validation, ownership, output preservation and
offset overflow. Model arithmetic and quantization remain separate: identical
random tensors do not guarantee identical logits or decoded waveform samples.
See sibling MuLaCover's `docs/PHILOX_PARITY.md` for full-song token/audio results
and the remaining strict waveform gate failure.
