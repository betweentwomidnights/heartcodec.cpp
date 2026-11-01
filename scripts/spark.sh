#!/usr/bin/env bash
# Run on Spark. Host actions are Docker orchestration; Python/builds stay in images.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
refs="${HEARTCODEC_REFS:-$(dirname "$project_root")/_heartcodec_refs}"
base_id=sha256:4e63792da2db6bd4de97a16666cc49278cd5a092c427738261847288af270437
base_tag=heartcodec-reference-base:20261101
image=heartcodec-dev:20261101
common=(--rm --user "$(id -u):$(id -g)" -v "$project_root:/workspace" -w /workspace)
python=/opt/sa3-venv/bin/python
case "${1:-}" in
    image)
        docker image inspect "$base_id" >/dev/null
        docker tag "$base_id" "$base_tag"
        docker build -t "$image" -f "$project_root/docker/Dockerfile.spark" "$project_root/docker"
        ;;
    bootstrap)
        docker run "${common[@]}" "$base_tag" "$python" tools/bootstrap-gguf.py
        ;;
    prepare)
        docker run "${common[@]}" -e HF_HUB_OFFLINE=0 -e HF_HUB_DISABLE_XET=1 \
            -e HF_HOME=/workspace/.cache/huggingface "$base_tag" "$python" \
            tools/prepare-sq.py --output models/sq
        ;;
    reference)
        docker run "${common[@]}" --gpus all --network none \
            -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v "$refs:/refs:ro" "$base_tag" "$python" \
            reference/export_sq.py --heartlib /refs/heartlib --spansynth /refs/spansynth-edit \
            --weights models/sq/sq.safetensors --config models/sq/config.json \
            --output "${2:-refdata/sq-cuda}" --samples "${3:-1920,48000}"
        ;;
    convert)
        docker run "${common[@]}" --network none -v "$refs:/refs:ro" "$base_tag" "$python" \
            tools/convert-sq-to-gguf.py --heartlib /refs/heartlib --weights models/sq/sq.safetensors \
            --config models/sq/config.json --output models/heartcodec-sq-decoder-f32.gguf
        ;;
    build-cpu)
        docker run "${common[@]}" --network none "$image" /bin/bash -lc \
            'cmake -S . -B build-cpu -G Ninja -DGGML_NATIVE=OFF -DGGML_BUILD_TESTS=OFF && cmake --build build-cpu -j 4 && ctest --test-dir build-cpu --output-on-failure'
        ;;
    build-cuda)
        docker run "${common[@]}" --network none --gpus all "$image" /bin/bash -lc \
            'cmake -S . -B build-cuda13 -G Ninja -DGGML_NATIVE=OFF -DGGML_BUILD_TESTS=OFF -DHEARTCODEC_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real && cmake --build build-cuda13 -j 4 && ctest --test-dir build-cuda13 --output-on-failure'
        ;;
    precision-test)
        docker run "${common[@]}" --network none --gpus all -e GGML_CUDA_TF32=0 "$image" /bin/bash -lc \
            'cmake -S . -B build-cuda13 -DGGML_BUILD_TESTS=ON && cmake --build build-cuda13 --target test-mul-mat-f32-precision -j 4 && ctest --test-dir build-cuda13 -R "^test-mul-mat-f32-precision$" --output-on-failure'
        ;;
    compare)
        backend="${2:-cpu}"
        build=build-cpu
        gpu=()
        if [[ "$backend" == cuda ]]; then build=build-cuda13; gpu=(--gpus all); fi
        docker run "${common[@]}" "${gpu[@]}" --network none -e GGML_CUDA_TF32=0 "$image" "$python" \
            tools/compare-sq.py --reference "${3:-refdata/sq-cuda}" --cli "$build/bin/heartcodec-cli" \
            --model models/heartcodec-sq-decoder-f32.gguf --output "${4:-refdata/native-$backend}" --backend "$backend"
        ;;
    *)
        echo 'Usage: bash scripts/spark.sh image|bootstrap|prepare|reference [OUT SAMPLES]|convert|build-cpu|build-cuda|precision-test|compare [cpu|cuda REF OUT]' >&2
        exit 2
        ;;
esac
