# Spark Docker workflow

Workspace: `/home/kev/dev/heartcodec.cpp` on `spark-c7ca`.
Reference checkouts: `/home/kev/dev/_heartcodec_refs/{heartlib,spansynth-edit}`.
Run commands below from the project root on Spark. Python, model execution,
dependency installation, builds, and tests run **inside Docker**. Host directories
are persistent mounts. Reference repositories are mounted read-only during runs.

## Images

The reference uses the already-installed SA3 image, verified by immutable image ID:

```text
sha256:4e63792da2db6bd4de97a16666cc49278cd5a092c427738261847288af270437
```

It is locally tagged `heartcodec-reference-base:20261101`; it contains Python
3.10.12, PyTorch 2.7.1+cu128, NumPy 2.2.6, and safetensors 0.7.0. It successfully
executes on GB10. This is an isolated new container, not the production SA3 container.

`docker/Dockerfile.spark` derives `heartcodec-dev:20261101` from that image, adding
CMake and a CUDA 13.0 compiler/cuBLAS toolchain for GGML's **121a-real** target:

```text
sha256:150cde2b00b093fa40b6f107e10538342a2be108378e53cfd18fe6fac1995c2a
```

The reference remains on its validated CUDA 12.8 PyTorch stack. Native builds use
the development image's CUDA 13.0 toolkit. No host CUDA/Python installation is used.
The inherited base image contains `HF_HUB_OFFLINE=1`; the download command explicitly
overrides it. Inference, conversions, and tests run with networking disabled.

## Reproduce the first milestone

The reference repositories must be at the revisions in
[the environment assessment](DEVELOPMENT_ENVIRONMENT.md). Model downloads are pinned
in `tools/prepare-sq.py`. GGML is a submodule of the shared fork, pinned at
`07b77b0112cc30861b49116b057bac33e819ee06`.

```bash
bash scripts/spark.sh image
bash scripts/spark.sh bootstrap
bash scripts/spark.sh prepare
bash scripts/spark.sh reference
bash scripts/spark.sh convert
bash scripts/spark.sh build-cpu
bash scripts/spark.sh build-cuda
bash scripts/spark.sh precision-test
bash scripts/spark.sh compare cpu refdata/sq-cuda refdata/native-cpu
bash scripts/spark.sh compare cuda refdata/sq-cuda refdata/native-cuda
```

`bootstrap` installs the pinned pure-Python GGUF 0.19.0 wheel into the mounted
`.python-deps/` directory, verifying its PyPI SHA256. It does not modify the image
or host Python. `prepare` downloads only the shard containing all `scalar_model.*`
tensors and writes a strict SQ-only safetensors checkpoint with provenance.

Boundary reference set (48 kHz sample counts):

```bash
bash scripts/spark.sh reference refdata/sq-boundaries 1919,1921,1428479,1428480,1428481
bash scripts/spark.sh compare cpu refdata/sq-boundaries refdata/native-cpu-boundaries
bash scripts/spark.sh compare cuda refdata/sq-boundaries refdata/native-cuda-boundaries
```

For synthetic decoder inputs and real music, run `reference/export_sq.py` inside the
reference image using the same mounts and options as `scripts/spark.sh reference`.
Add `--synthetic-latents --samples 5760` or
`--audio /refs/heartlib/assets/reference.mp3 --samples 144000`, respectively, and
choose a separate output directory. Audio decoding/resampling uses the image's
ffmpeg; the two stereo channels become independent mono SQ fixtures.

## Native interface

`include/heartcodec.h` exposes a versioned C ABI with explicit status/error returns,
size-prefixed structs, serialized context use, and library-owned audio release.
The initial operation consumes mono SQ latents in `[128, frames]` order and emits
`frames * 1920` mono samples at 48 kHz. It applies the upstream scalar quantizer
inside decode, including ties-to-even rounding. No host audio trimming, resampling,
or stereo packing is implied by this narrow SQ primitive.

The CLI currently exchanges little-endian raw FP32 files, matching the reference
exporter's `.f32` companions to its `.npy` tensors:

```bash
GGML_CUDA_TF32=0 build-cuda13/bin/heartcodec-cli sq-decode \
  --model models/heartcodec-sq-decoder-f32.gguf \
  --input refdata/sq-cuda/sine-48000/sq.latent.f32 \
  --output refdata/example.f32 --backend cuda --dump-dir refdata/example-stages
```

Run that command inside the development image with its workspace mounted. The
`compare` wrapper already sets **`GGML_CUDA_TF32=0`** for reference accuracy.
GGML's default CUDA math permits TF32; requesting F32 matrix precision alone does
not disable that cuBLAS math mode. The shared-fork fix additionally prevents the
small-matrix MMF path from bypassing the full-precision cuBLAS path.

Weights, fixtures, cache, and build directories stay on the mounted workspace and
are excluded from Git. Small reports are copied to `docs/validation/` for review.
