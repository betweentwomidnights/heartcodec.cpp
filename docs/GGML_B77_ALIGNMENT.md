# Shared GGML baseline and precision review

November 1, 2026. User-selected baseline:
`b77ae790e628270af46a8af1eca38441c23f1252` in
`betweentwomidnights/ggml`, branch `feature/shared-sa3-acestep-v0.17`.
HeartCodec's gitlink and both laptop/Spark GGML checkouts use this clean commit.
The branch name is historical; exact SHA is authoritative. No sibling checkout
or production container was changed.

The consumer now uses `ggml_prec_set_acc(F32)` and, for F32 weights,
`ggml_prec_set_src(F32, 1)`. These are separate requests in the new public API:
FP32 accumulation does not prohibit reducing source precision. Quantized/F16 SQ
experiments retain their earlier accumulator policy.

## Baseline results

Fresh Windows CPU build: `build-windows-b77-cpu`. All 33 SQ fixtures pass with
zero changed scalar levels; all five short flow cases pass. Frozen references
are reused without regenerating the upstream oracle. Seven Python tooling tests
and the native API/quantizer/precision tests pass. Detailed reports remain ignored under
`refdata/b77-*`; compact results are recorded beside this report.

Fresh shared Spark CUDA build: sibling MuLaCover's `build-shared-b77-cuda`,
one in-tree GGML instance, GCC 11.4/CUDA 13.0, GB10 `121a-real`, TF32 disabled.
Builds and tests ran only in dedicated Docker images, with GGML mounted read-only.

The standalone K=512/M=64 F32 multiply regression fails on CUDA for N=5, 8, 16:
maximum absolute error about 0.0014, relative L2 about 0.00075. N=1, 17, 25, 64
pass; all seven shapes pass CPU. Explicit source and accumulator requests are set.
The two small HeartCodec flow cases fail their unchanged tensor gates. MuLaCover
fails 130/300 conditional and 124/300 CFG tensors; reset/replay is still exact and
unused cache tails stay zero. Historical `07b77b0` passing reports remain intact
and must not be treated as validation of `b77ae79`.

## Isolated proposal

Draft [GGML PR #21](https://github.com/betweentwomidnights/ggml/pull/21), head
`0f2ed045eaff6bf01bed074878b6988248efbc22`. Its first parent before the two proposal
commits is exactly `b77ae79`; target is `feature/shared-sa3-acestep-v0.17`.

The proposed branch `fix/cuda-f32-source-precision-b77` starts at the exact shared
baseline, in a separate GGML worktree. It routes ordinary F32/F32 `MUL_MAT` with
an explicit src1 F32 request to cuBLAS before MMF can downconvert its operands.
It retains the full-float MMVF paths and does not change default, quantized,
F16/BF16, or `MUL_MAT_ID` dispatch. The installed-package experiment remains
deferred and is not included.

This is scoped to the parity environment: `GGML_CUDA_TF32=0` and no forced
reduced cuBLAS compute type. It does not claim to repair every backend/operation's
precision handling or override the fork's TF32 environment policy.

The GGML regression covers N=1/5/8/16/17/25/64 against FP64 host dot products,
with explicit source precision and default/explicit F32 accumulation on CPU/CUDA.
Independent review confirmed the narrow dispatch change and prompted restriction
of default-accumulator test coverage to these backends, plus a supports-op check.

The review candidate is mounted over `/codec/ggml` only in isolated test
containers and uses separate build/output directories. Shared checkout and
consumer pins stay at `b77ae79`; adopting a reviewed fix is a later coordinated
decision. Candidate parity must report its actual commit and be marked unadopted.

## Candidate results (not adopted)

- New precision regression: CPU/CUDA pass all seven shapes with both accumulation
  settings; CUDA max absolute error 1.56e-6. Five combined native tests pass.
- Existing GGML CUDA `MUL_MAT` filter `m=16,n=8,k=256`: 78/78 pass.
- MuLaCover: 600/600 tensors pass unchanged gates, exact replay and zero cache tails.
  Worst guided-logit absolute error 3.624e-5, worst KV error 4.483e-5.
- HeartCodec flow: all six CUDA cases pass, including 372 frames and all step counts.
- SQ: all 33 existing numerical gates pass. Three long-noise cases each change
  one of 95,232 encoder levels. The `noise-1428479` baseline/candidate crosscheck
  has identical metrics and the same one-level change on unmodified `b77ae79`.
  This is not zero-level-change equivalence. The normal roundtrip uses frozen
  reference levels; complete native encoder-decoder transparency is not established.

Compact evidence: [baseline](validation/ggml-b77-baseline.json),
[unadopted candidate](validation/ggml-b77-review-candidate.json), and MuLaCover's
`docs/validation/native-f32-b77-{baseline,review-candidate}.json`.
All candidate model execution was inside dedicated Spark Docker containers.
The laptop's new baseline MuLaCover static and DLL C ABI smoke builds also pass.
