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
- `heartcodec_work.h` separately negotiates controlled encode/token-decode calls
  with cooperative cancellation and worker progress. Keep legacy tables frozen;
  see `docs/CONTROLLED_WORK.md`. Never interrupt a running backend graph or retain
  a caller observer after return. MuLaCover owns orchestration and context staging.
- Full waveform-to-RVQ encoding is now an experimental native capability in
  `heartcodec_encoder.h`, separate from frozen core/flow/RNG layouts. It includes
  native resampling, mel/STFT, MuEncoder/Whisper/WavLM, query fusion and original
  decoder RVQ encoding. See `docs/AUDIO_ENCODER.md`. No native inference launches
  Python. Reference generation/conversion stays in `scripts/spark.sh` Docker.
  The encoder GGUF is strict F32 (983 tensors); no encoder quantization yet.
- Complete frames are floor(original_samples*25/(2*rate)); partial source tails
  are dropped explicitly. Mono/stereo borrowed views have element strides;
  library-owned frame-major eight-code outputs survive context destruction.
  Stage encoder/generator/decoder ownership and keep calls off realtime threads.
- CPU raw fixtures match all 8,000 IDs (29.76s and 50s stereo 44.1k, 0.24s mono
  48k). CUDA on MuLaCover's authorized shared development target matches two
  cases but changes four IDs at frame259 levels4--7 in the 50s case. Measured
  reference level4 margin is 5.96e-8. Further tests show feature and query
  roundoff jointly determine this result; do not isolate blame to CUDA query.
  A stock Torch CUDA neural-stage oracle (frozen CPU frontend/effective weights)
  produces those same four changes; native CUDA matches all its 5,000 codes.
  This is not an end-to-end CUDA tokenize claim or a CPU-gate waiver.
  Strict 1e-3 max_abs / 1e-4 relative-L2 intermediate gates retain failures in
  Whisper activations. Frozen checkpoint rotary frequencies closed the earlier
  MuEncoder gate failures. Do not call complete encoder parity
  passed, waive code mismatches, or optimize/quantize before resolving gates.
- Whisper's deeper investigation is in `docs/WHISPER_PARITY.md`. The user accepts
  small numerical drift in principle; a separate upstream-envelope accuracy rule
  is a review proposal, not an automatic waiver of existing flat gates or RVQ
  mismatches. Stock Torch CPU F32 itself differs from F64 by 0.001953125 on the
  isolated layer6 FC2 projection; native CPU differs by 0.0001220703125. Preserve
  these controls and original reports. Numerical runtime trials were discarded;
  no new shared-GGML patch is adopted. Python additions remain offline diagnostics.
  The proposed contract passes matched-mel tests on both backends for the pilot
  and two held-out inputs. Those held-out raw-audio encodings match all 1,840 IDs
  on both backends; CPU now matches 9,840 compared IDs. CUDA's earlier four-code
  difference remains open. See `docs/validation/whisper-validation.json`.
- The backend-matched RVQ audit is in `docs/RVQ_BOUNDARY.md` and
  `docs/validation/rvq-backend-parity.json`: native CPU matches 9,840 CPU codes;
  native CUDA matches 9,840 CUDA neural-stage codes across the same five inputs.
  CUDA controls use frozen CPU frontend tensors, not stock end-to-end CUDA
  tokenize. Original cross-backend four-code and activation failures remain.
  Native CPU query on frozen upstream CPU features also changes those four IDs;
  whole native CPU encoding matches CPU. Identical-query RVQ replay matches on
  both backends. No tie epsilon, remapping, hidden CPU fallback or GGML change
  is justified by this boundary. Code-only passes report activation parity null.
- The native CPU encoder guide -> native Q4 continuation/stereo chain matches
  all 8,760 accepted native codes and both WAVs exactly. This validates that
  listening baseline, not unchanged strict upstream waveform limits. Cached
  text/Qwen preparation is still separate. The user prefers a downstream
  `natural_max_seconds`-style generation ceiling that permits early EOS; encoder
  chunk duration and guide duration are independent of that job policy.
