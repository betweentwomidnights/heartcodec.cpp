# Native integration contract

The public header is `include/heartcodec.h`. It contains C types only and does
not expose GGML or C++ ownership. Windows x64 static and DLL consumers have been
tested, along with an ARM64 Linux build inside Spark Docker.

## Negotiation and ownership

Resolve `heartcodec_get_api`, request `HEARTCODEC_ABI_VERSION_1`, check for a
non-null table, and require `api->size >= HEARTCODEC_API_V1_SIZE`. Unsupported
versions return null. The table has library lifetime. The v1 prefix ends at
`tokens_decode`; future tables may append entries without moving that boundary.
Existing v1 argument/result structures are frozen; incompatible changes require
new types/functions. Direct exports remain available for source integration.

Set each argument/result's `size = sizeof(value)` before calling its initializer.
Initializers leave a caller's larger tail untouched and reject smaller prefixes.
Initialize results, free a previous result before reusing it, and release buffers
through this library's corresponding free function. Freeing an initialized empty
result is safe. Results outlive component unload and context destruction, provided
the library remains loaded until their release. Inputs are borrowed for a call.

Calls on a context must be serialized. Error details are caller-owned. Failures
return a status rather than propagating C++ exceptions. Backend selection is exact:
`cpu`, `cuda`, `vulkan`, or `metal`; an unavailable backend returns an error.
Current parity evidence covers CPU/CUDA, not Vulkan/Metal.

## Layouts and component lifecycle

| Type | Layout / behavior |
| --- | --- |
| SQ latents | `[128,T]` float32, time contiguous; 25 Hz |
| RVQ token view | Eight codebooks, int32 indices 0..8191; explicit positive strides in elements; 12.5 Hz |
| RVQ result | Frame-major `[T,512]`, after the shared projection |
| Flow estimator input | Frame-major `[N,1024]`: x256, context256, mu512 |
| Flow result | Frame-major `[2*T,256]` |
| Stereo audio | Planar `[2,num_samples]`, float32 at 48 kHz |

`context_create` eagerly loads the initial SQ or flow GGUF. Register another
component with `component_set`; its path is copied and loading is lazy until first
use or `component_load`. Setting a new path unloads the old component. Unloading
retains its path for reload. Both unload and all result frees belong to this library.

`tokens_decode` unloads SQ, runs all flow windows, releases flow, then loads SQ and
decodes each stereo channel. This staging keeps the validated FP32 decoder within
the laptop's 8GB GPU. It leaves SQ resident; later calls reload flow as needed.
`get_memory` reports currently resident model buffers and graph buffer sizes;
backend contexts, driver allocation, caches and workspaces are excluded.

Defaults are 10 Euler steps, guidance 1.25 and a 372-code / 29.76-second window.
Window sizes must be positive multiples of 93. Hop is `window/93*80` codes, and
output is trimmed to exactly `input_token_frames*3840` samples per channel.
Short input repeats; longer input is repeated as needed to complete the final
window. A previous window's final latent overlap becomes the next prefix.
Crossfades include both endpoints and mix in float64 before float32 assignment,
matching upstream. SQ decoding applies upstream ties-to-even scalar quantization.

Noise uses a library-specific seeded RNG, not Torch's RNG. For reproducible
cross-runtime comparisons, supply saved float32 noise. `flow_decode` accepts one
noise window, optional prefix context, and an optional `steps+1` schedule. The
prefix is preserved exactly at the end. `tokens_decode` accepts a contiguous stack
of noise windows with the exact `noise_segments` count and computes context itself.
Caller noise/context/schedule buffers must contain the documented element counts.

## CMake integration

```cmake
find_package(heartcodec CONFIG REQUIRED)
target_link_libraries(my_model PRIVATE heartcodec::heartcodec)
```

Or, when the application already builds the shared GGML fork:

```cmake
add_subdirectory(path/to/shared-ggml ggml)
set(HEARTCODEC_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
add_subdirectory(path/to/heartcodec.cpp heartcodec)
target_link_libraries(my_model PRIVATE heartcodec::heartcodec)
```

Existing `ggml` or `ggml::ggml` targets are reused automatically. An explicit
`HEARTCODEC_GGML_TARGET` may name another existing target from the same fork.
HeartCodec does not override backend options when the parent owns GGML. The
installed package refers to `ggml::ggml`; it finds GGML only if the application has
not already provided that target. Its private `heartcodec::ggml_dependency` bridge
avoids exporting or bundling a second GGML build. CMake 3.20+ and C++17 are required.

Standalone builds support `HEARTCODEC_STATIC=OFF` for a DLL/shared library, and
install the library, public header, exported target and versioned package config.
For static consumers, the GGML package must come from the same shared fork/pin.
When combining multiple model libraries, either link them statically into one
application using one GGML target, or use a single shared GGML library across
model DLLs. Separate model DLLs each embedding static GGML would create separate
registries, even if they use the same source revision.

Static, DLL, installed-package, imported-GGML-parent and source-GGML-parent
integration checks live under `tests/package`.

Torch CUDA RNG is a separate optional capability negotiated with
`heartcodec_get_rng_api(1)` from `heartcodec_rng.h`; it does not extend or alter
the frozen core/flow table or structs. Its owned host tensor can be borrowed
through the existing explicit-noise fields. See [RNG semantics and validation](TORCH_CUDA_RNG.md).
