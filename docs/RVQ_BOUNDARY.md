# RVQ boundary and backend-matched code results

Native CPU matches all 9,840 upstream CPU code IDs across five raw-audio inputs.
Native CUDA matches all 9,840 upstream CUDA neural-stage IDs on those same inputs.
The original CPU/CUDA four-code discrepancy remains recorded as a failed
cross-backend comparison. No remapping, tie epsilon, CPU fallback or new GGML
kernel is introduced. These are exact code results; original activation and
decoder-waveform limits are separate and remain unchanged.

## The disputed frame

In the 50-second input, the first difference is frame 259 (20.72 seconds),
residual level 4. CPU selects 6144; CUDA selects 7217. Levels 5--7 then differ
because a different level-4 embedding was subtracted. This is one residual
cascade, not four independent frames.

Direct F64 distances retain each choice when given its own projected latent and
F32 subtraction history. The Torch CPU latent is about 2.48e-8 from the nearest
code boundary, while native CUDA is about 9.44e-8 away on the other side.
Higher-precision distance re-ranking cannot make these distinct points select
the same embedding. See `docs/validation/rvq-decision-boundary.json`.

Independent query controls use identical frozen CPU feature inputs and effective
F32 weights materialized on CPU before selecting an arithmetic mode:

| Query control | Level-4 code at frame 259 |
| --- | ---: |
| Original Torch CPU F32 | 6144 |
| Original Torch CUDA F32 | 7217 |
| F64 arithmetic retaining the source's forced-F32 rotary operations | 7217 |
| Diagnostic F64 arithmetic and rotary operations | 7217 |

The F64 runs are diagnostics, not replacement goldens. The last control
deliberately removes the source's rotary casts; its manifest labels that change.
Pinned reference files remain read-only and unchanged.

One earlier inference was too narrow: feature and query roundoff jointly affect
the result. Native CPU query/RVQ on frozen upstream CPU features also selects
7217 and differs by four IDs. Whole native CPU encoding selects 6144 on this
input. This does not support attributing the discrepancy solely to a faulty
CUDA query backend or claiming universal component-level code identity.

Replaying each of four identical query tensors through native CPU and CUDA RVQ
produces identical codes on both backends and matches each tensor's originating
code stream: 2,976 IDs per replay, eight replays. The quantizer does not introduce
an additional discrepancy in those controls.

## Backend-matched validation

| Raw input | Frames | CPU vs CPU oracle | CUDA vs CUDA neural oracle |
| --- | ---: | ---: | ---: |
| 29.76-second source prefix, stereo 44.1 kHz | 372 | 2,976 exact | 2,976 exact |
| 50-second source prefix, stereo 44.1 kHz | 625 | 5,000 exact | 5,000 exact |
| 0.24-second excerpt, mono 48 kHz | 3 | 24 exact | 24 exact |
| Held-out 16-second song passage, stereo 44.1 kHz | 200 | 1,600 exact | 1,600 exact |
| Held-out 2.4-second stereo tone, 48 kHz | 30 | 240 exact | 240 exact |

The CUDA oracle runs stock MuEncoder, Whisper, WavLM, query and RVQ modules on
Spark CUDA, using frozen CPU frontend tensors and CPU-materialized effective
weights. This is **not** stock end-to-end CUDA `tokenize`: the reference image's
complex STFT Jiterator path does not support GB10. Native encoding consumes the
original WAV and performs the entire native PCM-to-code path.

The CLI now records its requested backend in `run.json`. Code-only comparison
can require matching backend labels and explicitly leaves activation parity
unevaluated (`null`):

```powershell
./.venv/Scripts/python.exe tools/compare-encoder.py `
  --reference refdata/encoder50-cuda-stages-v1 `
  --native refdata/rvq-profile-cuda-raw50 `
  --model models/heartcodec-encoder-f32-v2.gguf `
  --mode codes --require-matching-backend --output refdata/example-code-comparison.json
```

Without code-only mode, original tensor limits and failure behavior remain
unchanged. Backend matching never excuses a changed code or golden hash.
Six offline tests exercise those distinctions. `rvq-backend-parity.json` records
the matched results, source controls, and identical-query replays. Previous
cross-backend failure reports are preserved.

For query diagnostics on Spark, inside the established Docker workflow:

```sh
bash scripts/spark.sh reference-query-precision refdata/encoder50-cpu-v1 refdata/query-cpu source-f32 cpu
bash scripts/spark.sh reference-query-precision refdata/encoder50-cpu-v1 refdata/query-cuda source-f32 cuda
bash scripts/spark.sh reference-query-precision refdata/encoder50-cpu-v1 refdata/query-mixed64 source-mixed-f64 cpu
bash scripts/spark.sh reference-query-precision refdata/encoder50-cpu-v1 refdata/query-f64 diagnostic-f64 cpu
```

`tools/analyze-rvq-boundary.py` consumes explicit projected-latent and code files;
offline analysis cannot affect runtime selection. Centered query normalization
was tested with existing GGML operations and reverted after it did not remove
the difference. Shared GGML remains b77ae79, CUDA uses MuLaCover's single
authorized isolated target 0f2ed045, and no sibling bump or additional GGML PR
was needed. Native numerical execution is unchanged.
