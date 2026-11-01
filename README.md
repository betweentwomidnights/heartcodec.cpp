# heartcodec.cpp

Shared native HeartCodec primitives for future MuLaCover and SpanSynth consumers
in the gary4local / gary4juce ecosystem. Implementation follows sa3.cpp and uses
the shared [betweentwomidnights/ggml](https://github.com/betweentwomidnights/ggml) fork.

The first completed step is the [development environment assessment](docs/DEVELOPMENT_ENVIRONMENT.md).
Use **spark-c7ca for full FP32 reference generation and full-pipeline parity**.
The RTX 5070 Laptop remains useful for isolated SQ work and native backend tests.
**Base SQ encode, quantize, and decode now run locally on laptop CPU and CUDA.**
The Windows checkout is the main development environment for these primitives.
FP32 passes 33 frozen Spark fixtures with zero scalar-level mismatches; encoder
stage checks also pass. RVQ/flow and the full tokenizer remain future work.

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
larger roadmap. The next model milestone is RVQ and the flow decoder, followed by
query and feature extraction. Quantization stays controlled per component.

Keep 25 Hz / 128-channel SQ latents and 12.5 Hz / eight-stream RVQ tokens as distinct
representations. Generic missing operators belong in the shared GGML fork.
