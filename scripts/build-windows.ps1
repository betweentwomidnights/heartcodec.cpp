param([ValidateSet('cpu','cuda','vulkan')][string]$Backend = 'cpu')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if ($cmake) { $cmakePath = $cmake.Source } else {
    $cmakePath = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
}
if (-not (Test-Path -LiteralPath $cmakePath)) { throw 'CMake was not found.' }
$buildPath = Join-Path $projectRoot "build-windows-$Backend"
$flags = @('-DHEARTCODEC_CUDA=OFF','-DHEARTCODEC_VULKAN=OFF','-DGGML_BUILD_TESTS=OFF')
if ($Backend -eq 'cuda') { $flags += @('-DHEARTCODEC_CUDA=ON','-DCMAKE_CUDA_ARCHITECTURES=120a-real') }
if ($Backend -eq 'vulkan') { $flags += '-DHEARTCODEC_VULKAN=ON' }
& $cmakePath -S $projectRoot -B $buildPath -G 'Visual Studio 17 2022' -A x64 @flags
if ($LASTEXITCODE) { throw 'CMake configuration failed.' }
& $cmakePath --build $buildPath --config Release --parallel 4 -- /verbosity:minimal /nologo
if ($LASTEXITCODE) { throw 'Build failed.' }
$ctestPath = Join-Path (Split-Path -Parent $cmakePath) 'ctest.exe'
& $ctestPath --test-dir $buildPath -C Release --output-on-failure
if ($LASTEXITCODE) { throw 'Tests failed.' }
