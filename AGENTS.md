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
  The initial SQ milestone pins `07b77b0112cc30861b49116b057bac33e819ee06`
  (shared branch `feature/heartcodec-cuda-f32-precision`).
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
