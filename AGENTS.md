# Project instructions

- Follow `shared-heartcodec-cpp-implementation-brief.md` and sa3.cpp's library,
  GGUF, parity, ownership, and shared-GGML patterns.
- **Spark development runs inside Docker.** Run Python, inference, conversion,
  dependency installation, native builds, and tests in dedicated HeartCodec
  containers. The host holds mounted project files and runs Docker orchestration.
  Do not install a host Python environment or modify production service containers.
- Use `scripts/spark.sh` for the established container workflow. Reference code is
  mounted read-only from the sibling `_heartcodec_refs` directory.
- Use `betweentwomidnights/ggml`; keep generic kernels and backend fixes there.
  The coordinated pin is `e3db5210fe3ecdaa4d65b4c93d515f2cf15524b7` (GGML 0.29,
  ggml#22) on `feature/shared-sa3-acestep-v0.17`. The branch name is historical;
  use exact commit identity rather than inferring the upstream release from its
  name. Earlier `07b77b0` and `b77ae79` parity reports are historical until
  revalidated on this pin.
- GGML changes and pin updates require coordination across consumers. First try
  the existing in-tree target/API. If a change is essential, isolate it in a
  separate worktree/branch and open a focused draft PR for cross-agent review;
  keep consumer pins unchanged until adoption is coordinated after review.
  Any necessary new GGML PR must start from the coordinated `e3db521` baseline
  and target `feature/shared-sa3-acestep-v0.17`; never patch the shared checkout.
- The CUDA F32-source fix that MuLaCover developed against in isolation
  (`0f2ed045`, ggml#21) is part of the coordinated pin. ggml-bump-check covers
  this repository and MuLaCover on CUDA, Vulkan and Metal.
- Match unquantized intermediate tensors before quantization or optimization.
  SQ latents and RVQ codes are distinct representations.
- The laptop checkout at `C:\dev\heartcodec.cpp` is now the primary development
  environment for SQ and quantization. Use frozen Spark goldens under
  `refdata/laptop-goldens`; routine native work does not require Spark or PyTorch.
- Keep SQ FP32 as the validated default. `sq-conv-v1` quantization is experimental;
  report encoder scalar-level changes separately from decoder waveform error.
  Q4_K_M operation does not imply tokenizer compatibility or audio transparency.
- Keep weights, large tensor fixtures, caches, and builds out of Git. Keep small
  manifests/comparison reports in `docs/` so validation remains reviewable.
- Native RVQ/flow and token-to-stereo decoding now run on the laptop. Use
  `refdata/flow-cuda` and `refdata/stereo-cuda`; see `docs/FLOW_AND_STEREO.md` for
  rounding-boundary differences. Do not treat small FP32 flow error as proof of
  identical SQ levels. Flow GGUF is currently strict F32; its quantization is not
  implemented. MuLaCover generator fixtures live in sibling `mulacover.cpp`.
- Expose reusable functionality through the frozen negotiated C ABI and
  `heartcodec::heartcodec`. Reuse a downstream's GGML target, never add a second copy.
- `heartcodec_rng.h` adds a separate negotiated Torch CUDA Philox capability;
  frozen core/flow layouts and legacy RNG defaults remain unchanged. See
  `docs/TORCH_CUDA_RNG.md`. CUDA normal/exponential streams match Torch bit-for-bit
  on Spark and laptop when target grid cap 288 is explicit. CPU builds report
  this capability's generation unavailable, without RNG fallback.
  Sibling MuLaCover now matches all 861 upstream frames / 6,888 codes and EOS.
  Full-song decoded audio still fails unchanged strict waveform limits (max_abs
  0.0827555, relative L2 0.00299061); scalar-level diagnostics for that case remain
  outstanding. Do not call waveform parity passed or assume a rounding exception.
