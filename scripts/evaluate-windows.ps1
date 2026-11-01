param(
    [ValidateSet('cpu','cuda')][string]$Backend = 'cuda',
    [ValidateSet('parity','quantization')][string]$Suite = 'parity',
    [string]$Reference = 'refdata/laptop-goldens'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
$previousTf32 = $env:GGML_CUDA_TF32
try {
    $env:GGML_CUDA_TF32 = '0'
    $python = Join-Path $projectRoot '.venv/Scripts/python.exe'
    $cli = Join-Path $projectRoot "build-windows-$Backend/bin/Release/heartcodec-cli.exe"
    $quantizer = Join-Path $projectRoot 'build-windows-cpu/bin/Release/heartcodec-quantize.exe'
    if (-not (Test-Path -LiteralPath $cli)) { throw 'Build the requested backend first.' }
    if ($Suite -eq 'parity') {
        & $python tools/compare-sq-roundtrip.py --reference $Reference --cli $cli `
            --model models/heartcodec-sq-codec-f32.gguf --backend $Backend --output "refdata/windows-$Backend-parity"
        if ($LASTEXITCODE) { throw 'FP32 parity failed.' }
    } else {
        foreach ($mix in @('f16','q8_0','q6_k','q5_k_m','q4_k_m','q4_k_m-encoder','q4_k_m-decoder')) {
            $model = "models/heartcodec-sq-codec-$mix.gguf"
            if (-not (Test-Path -LiteralPath $model)) {
                $scope = 'codec'; $tier = $mix
                if ($mix -eq 'q4_k_m-encoder') { $scope = 'encoder'; $tier = 'q4_k_m' }
                if ($mix -eq 'q4_k_m-decoder') { $scope = 'decoder'; $tier = 'q4_k_m' }
                & $quantizer --input models/heartcodec-sq-codec-f32.gguf --output $model --mix $tier --scope $scope
                if ($LASTEXITCODE) { throw "Quantization failed: $mix" }
            }
            & $python tools/compare-sq-roundtrip.py --reference $Reference --cli $cli --model $model `
                --backend $Backend --output "refdata/quant-$Backend-$mix" --evaluate `
                --case sine-48000 --case noise-48000 --case music-channel-0-144000 --case synthetic-5760
            if ($LASTEXITCODE) { throw "Quality evaluation failed to execute: $mix" }
        }
    }
} finally {
    $env:GGML_CUDA_TF32 = $previousTf32
    Pop-Location
}
