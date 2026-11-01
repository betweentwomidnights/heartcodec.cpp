# heartcodec.cpp

Shared native HeartCodec primitives for future MuLaCover and SpanSynth consumers
in the gary4local / gary4juce ecosystem. Implementation follows sa3.cpp and uses
the shared [betweentwomidnights/ggml](https://github.com/betweentwomidnights/ggml) fork.

**Native RVQ tokens now decode to 48 kHz stereo on the 8GB RTX 5070 Laptop.**
RVQ lookup, the full FP32 flow transformer, guided Euler sampling, stereo splitting,
window context, overlap/crossfade and trimming run locally using the shared GGML
fork. Flow and SQ weights load in separate stages. The laptop is the primary
native development workspace; Spark Docker containers generate upstream fixtures.
The [RVQ/flow and stereo report](docs/FLOW_AND_STEREO.md) records parity, memory,
and the small number of SQ rounding-boundary changes in the long fixture.

Base SQ encode/quantize/decode continues to pass its frozen parity suite. Full
audio-to-RVQ tokenization (query/feature encoders) remains future work. MuLaCover
generator development lives in the sibling `mulacover.cpp` repository, where
FP32/BF16 prompt, symbolic, logits and KV-cache reference fixtures are established.

The [laptop development and quantization report](docs/LAPTOP_DEVELOPMENT.md) includes
local FP16, Q8, Q6, Q5_K_M, and Q4_K_M measurements. Q4_K_M reduces the combined SQ
weights from 641 MiB to 123 MiB, but changes encoder levels and reconstructed audio.
FP32 remains the SQ default; quantized SQ models are experimental.

Start local work with:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\scripts\build-windows.ps1 cpu
.\scripts\build-windows.ps1 cuda
.\scripts\evaluate-windows.ps1 -Backend cuda -Suite parity
.\scripts\evaluate-windows.ps1 -Backend cuda -Suite quantization
.\scripts\evaluate-windows.ps1 -Backend cuda -Suite flow
.\scripts\evaluate-windows.ps1 -Backend cuda -Suite stereo
```

The frozen GGUF and fixture files are already present in this workspace and remain
outside Git. The [Docker workflow](docs/SPARK_WORKFLOW.md) remains available for
reference generation on Spark. Spark work continues to run inside Docker.

Reproduce the checkpoint inventory without installing PyTorch or downloading weights:

```powershell
python tools/inspect-heartcodec.py --output docs/checkpoint-audit.json
python -m unittest discover -s tests -v
```

The inspector reads HF metadata and bounded safetensors header ranges. It resolves
each model to an immutable revision and checks shard names, tensor names, and byte
counts against the index when present. Storage scenarios exclude execution memory.
The checked-in [audit](docs/checkpoint-audit.json) records exact checkpoint revisions.

The [implementation brief](shared-heartcodec-cpp-implementation-brief.md) defines the
larger roadmap. The next generator milestone is native teacher-forced MuLaCover
parity, then its Q4_K_M policy. Query and feature extraction remain later work.
Quantization stays controlled per component; the flow GGUF currently requires F32.

Downstream builds consume `heartcodec::heartcodec` through `find_package` or
`add_subdirectory`, reusing an existing GGML target. The negotiated C ABI is
resolved through `heartcodec_get_api(HEARTCODEC_ABI_VERSION_1)`. See the
[integration contract](docs/C_ABI_V1.md) for ownership, layouts and build examples.

Keep 25 Hz / 128-channel SQ latents and 12.5 Hz / eight-stream RVQ tokens as distinct
representations. Generic missing operators belong in the shared GGML fork.
