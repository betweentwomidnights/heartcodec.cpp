# Whisper parity investigation and proposed accuracy contract

The original encoder tensor limits remain unchanged: maximum absolute error
1e-3, relative L2 1e-4, and exact RVQ IDs. Whisper still fails the flat absolute
limit. This document does not turn those reports into passes.

The user considers small numerical differences acceptable if numerical fidelity
and downstream compatibility remain goals. The recommendation is to evaluate
Whisper accuracy against independently measured upstream arithmetic variation,
while retaining exact code comparisons and the original failures separately.
No encoder quantization or new shared-GGML change is adopted here.

## Measured cause

New traces expose every Whisper layer and the projections, normalizations and
activations around layers 0, 5, 6 and 11. The first large amplification occurs
in layer 6's feed-forward path. In the first 50-second-source chunk, an error
around 0.00025 entering that layer becomes about 0.015 in a hidden channel.
The reference hidden-state peak exceeds 800. This is not a missing layer,
incorrect GELU selection or misplaced query scaling found by this investigation.

With identical frozen layer-6 activated inputs and checkpoint weights:

| Projection result versus F64 calculation, rounded to F32 | Maximum absolute error | Relative L2 |
| --- | ---: | ---: |
| Upstream Torch CPU F32 | 0.001953125 | 8.63869e-7 |
| Native CPU F32 | 0.0001220703125 | 6.85615e-8 |

The frozen Torch CPU projection reproduces exactly when rerun. Computing its
matrix product and bias addition separately still differs from F64 by
0.001953125; bias fusion alone does not explain the gap. The Spark Torch build
reports `BLAS_INFO=nvpl`. NVIDIA documents NVPL as an
[Arm CPU library](https://docs.nvidia.com/nvpl/latest/), so reproducing that
backend's rounding is distinct from improving a portable F32 calculation.
These measurements do not establish that every native operation is more accurate.

The complete stock Torch Whisper F32/F64 comparison reaches 0.015258789 in the
last hidden layer and 0.010281563 after final normalization on this input. Merely
disabling Torch's oneDNN CPU backend changes the last hidden layer by 0.017333984.
Thus the original absolute gate can fail between upstream execution paths too.
F64 runs are diagnostic controls, not replacement compatibility goldens.

Centered normalization, matching the stereo batch dimensions, and disabling
CUDA fast math did not close the original gates. Those runtime experiments were
discarded. The fast-math experiment used an isolated worktree based on b77ae79
with the already-authorized CUDA source fix; the experimental patch is not adopted
or proposed as another GGML PR. The shared checkout, authorized review checkout,
and consumer pins remain unchanged.

## Candidate rule, declared before held-out evaluation

`tools/compare-whisper-accuracy.py` implements the review proposal
`whisper-upstream-envelope-v1-candidate`. For each tensor on an identical mel
input and checkpoint:

1. Freeze stock Torch CPU F32, CUDA F32 and CPU F64 outputs independently.
2. Let E be the larger maximum absolute CPU/CUDA F32 error against F64.
3. Require native error against F64 <= max(1e-3, 2*E), **and** relative L2 <=1e-4.

The multiplier is an explicit, reviewable allowance for ordinary backend variation;
it is not an analytical error bound. The independent relative-L2 limit remains
fixed. A candidate pass does not claim bit equality, universal audio quality,
quantizer compatibility for every input, or passage of the original gates.
The comparator preserves both original CPU/CUDA flat-gate verdicts and never
waives code mismatches. Missing tensors, changed control hashes and nonfinite
results cannot pass.

Two held-out inputs were selected before applying this rule: a 16-second song
excerpt starting at 31.5 seconds, and a 2.4-second synthetic stereo tone clip.
These supplement the original source fixture rather than replacing it. See
`docs/validation/whisper-validation.json` for measured outcomes and provenance.
All checkpoint, PCM and raw tensor payloads remain excluded from Git.

Both held-out cases pass the candidate accuracy rule on CPU and CUDA. Each
backend's full PCM-to-RVQ runs additionally match all 1,840 held-out code IDs.
CPU now matches 9,840 compared IDs across the five raw fixtures. The original
50-second CUDA case's four changes remain open. This is encouraging evidence
for the candidate numerical contract, not an adopted replacement for the
original gates or a guarantee for arbitrary future input.

## Reproduction

On Spark, inside the established isolated Docker workflow:

```sh
bash scripts/spark.sh reference-whisper refdata/encoder50-cpu-v1 refdata/whisper-cpu source-chunk-0 cpu
bash scripts/spark.sh reference-whisper refdata/encoder50-cpu-v1 refdata/whisper-cuda source-chunk-0 cuda
HEARTCODEC_WHISPER_DTYPE=f64 bash scripts/spark.sh reference-whisper refdata/encoder50-cpu-v1 refdata/whisper-f64 source-chunk-0 cpu
```

`HEARTCODEC_WHISPER_THREADS` defaults to 8. Setting
`HEARTCODEC_WHISPER_MKLDNN=0` requests a separate CPU-backend diagnostic.
`HEARTCODEC_WHISPER_LINEAR_CONTROL=1` additionally audits the isolated layer-6
FC2 projection against F64 using its exact frozen activated input.
Each manifest records compute/storage dtypes, backend configuration and input
hashes; control generation never overwrites existing fixtures.

With the static native tool, `whisper` consumes a frozen mel fixture directory.
`whisper-layer --layer 6 --frames 1500` consumes the reference layer-5 hidden-state
file; `whisper-linear --layer 6 --projection fc2 --frames 1500` consumes its
activated-input file. These diagnostic boundaries borrow reference tensors only
for testing. Production `encode` always computes the entire path from PCM.

`tools/compare-whisper.py` retains the original flat limits. The accuracy comparator
requires `--reference-cpu`, `--reference-cuda`, `--reference-f64`, `--native`, and
a fresh `--output`. It reports a candidate verdict explicitly, alongside the
unchanged legacy verdicts. Native inference still invokes neither Python nor Torch.
