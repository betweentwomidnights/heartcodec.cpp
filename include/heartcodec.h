#ifndef HEARTCODEC_H
#define HEARTCODEC_H
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32) && defined(HEARTCODEC_SHARED)
# ifdef HEARTCODEC_BUILD
#  define HEARTCODEC_API __declspec(dllexport)
# else
#  define HEARTCODEC_API __declspec(dllimport)
# endif
#elif defined(__GNUC__) || defined(__clang__)
# define HEARTCODEC_API __attribute__((visibility("default")))
#else
# define HEARTCODEC_API
#endif
#ifdef __cplusplus
extern "C" {
#endif

typedef struct heartcodec_context heartcodec_context;
typedef enum heartcodec_status_v1 {
    HEARTCODEC_OK_V1 = 0,
    HEARTCODEC_INVALID_ARGUMENT_V1 = 1,
    HEARTCODEC_ERROR_V1 = 2
} heartcodec_status_v1;
typedef struct heartcodec_error_v1 {
    uint32_t size;
    heartcodec_status_v1 status;
    char message[512];
} heartcodec_error_v1;
typedef struct heartcodec_config_v1 {
    uint32_t size;
    const char * model_path;
    const char * backend; /* cpu, cuda, vulkan, metal; exact backend, no silent fallback */
    int32_t cpu_threads;
} heartcodec_config_v1;
typedef struct heartcodec_audio_v1 {
    uint32_t size;
    float * samples; /* mono, allocated by this library */
    int64_t num_samples;
    int32_t sample_rate;
} heartcodec_audio_v1;
typedef struct heartcodec_sq_latents_v1 {
    uint32_t size;
    float * data; /* [128, frames], time contiguous; allocated by this library */
    int64_t frames;
    int32_t channels;
} heartcodec_sq_latents_v1;
typedef struct heartcodec_memory_v1 {
    uint32_t size;
    uint64_t weight_buffer_bytes;
    uint64_t graph_buffer_bytes; /* most recent graph; excludes backend context/workspace */
} heartcodec_memory_v1;

/* RVQ token indices, distinct from the scalar-quantized SQ latent format.
 * Strides are in int32 elements. Both [8,T] and [T,8] views are supported.
 * Codes must be in [0,8191]; all eight codebooks are required. */
typedef struct heartcodec_tokens_v1 {
    uint32_t size;
    const int32_t * data;
    int64_t frames;
    int64_t codebook_stride;
    int64_t frame_stride;
} heartcodec_tokens_v1;

typedef struct heartcodec_tensor_v1 {
    uint32_t size;
    float * data; /* frame-major [frames, channels], library-owned */
    int64_t frames;
    int32_t channels;
} heartcodec_tensor_v1;

typedef struct heartcodec_stereo_v1 {
    uint32_t size;
    float * samples; /* planar [2, num_samples], library-owned */
    int64_t num_samples; /* per channel */
    int32_t sample_rate;
} heartcodec_stereo_v1;

typedef enum heartcodec_component_v1 {
    HEARTCODEC_SQ_V1 = 0,
    HEARTCODEC_FLOW_V1 = 1
} heartcodec_component_v1;

typedef struct heartcodec_flow_config_v1 {
    uint32_t size;
    int32_t steps; /* default 10 */
    float guidance; /* default 1.25; <=1 uses the conditional estimator */
    uint64_t seed; /* library RNG; not a PyTorch RNG seed */
    int64_t segment_codes; /* token-to-audio window, default 372, multiple of 93 */
    const float * noise; /* optional noise; see noise_segments */
    const float * context; /* optional same shape, flow_decode only */
    int64_t prefix_frames; /* prefix to preserve exactly from context */
    const float * schedule; /* optional steps+1 F32 values, from 0 to 1 */
    int64_t noise_segments; /* flow_decode: 0 or 1; token decode: exact window count.
                            * Each token-decode noise window is [2*segment_codes,256]. */
} heartcodec_flow_config_v1;

HEARTCODEC_API void heartcodec_config_init_v1(heartcodec_config_v1 * config);
HEARTCODEC_API void heartcodec_audio_init_v1(heartcodec_audio_v1 * audio);
HEARTCODEC_API void heartcodec_error_init_v1(heartcodec_error_v1 * error);
HEARTCODEC_API heartcodec_status_v1 heartcodec_context_create_v1(
    const heartcodec_config_v1 * config, heartcodec_context ** context, heartcodec_error_v1 * error);
HEARTCODEC_API void heartcodec_context_destroy_v1(heartcodec_context * context);
/* Input [128, frames], time contiguous. Applies upstream SQ quantization internally.
 * Context calls must be serialized. dump_directory optionally saves decoder stage .f32 files.
 * Initialize audio first; free its result before reusing it. */
HEARTCODEC_API heartcodec_status_v1 heartcodec_sq_decode_v1(
    heartcodec_context * context, const float * latents, int64_t frames,
    const char * dump_directory, heartcodec_audio_v1 * audio, heartcodec_error_v1 * error);
HEARTCODEC_API void heartcodec_audio_free_v1(heartcodec_audio_v1 * audio);
HEARTCODEC_API void heartcodec_sq_latents_init_v1(heartcodec_sq_latents_v1 * latents);
HEARTCODEC_API void heartcodec_sq_latents_free_v1(heartcodec_sq_latents_v1 * latents);
/* Input is finite mono audio at 48 kHz, at least two samples. Returns unquantized
 * tanh-bounded latents. Pre-pooling floors odd lengths; strided convolutions ceil.
 * Decode returns frames*1920 samples; callers decide padding/trimming policy. */
HEARTCODEC_API heartcodec_status_v1 heartcodec_sq_encode_v1(
    heartcodec_context * context, const float * waveform, int64_t num_samples,
    const char * dump_directory, heartcodec_sq_latents_v1 * latents, heartcodec_error_v1 * error);
HEARTCODEC_API heartcodec_status_v1 heartcodec_get_memory_v1(
    heartcodec_context * context, heartcodec_memory_v1 * memory, heartcodec_error_v1 * error);
/* PyTorch-compatible ties-to-even round(9*x)/9. Supports in-place operation.
 * Inputs are finite SQ latents; values whose multiplication by 9 overflows are rejected. */
HEARTCODEC_API heartcodec_status_v1 heartcodec_sq_quantize_v1(
    const float * input, float * output, size_t count, heartcodec_error_v1 * error);

/* Registering a component is lazy. Loading is explicit or occurs on first use.
 * Re-registering releases its current model; unload retains the registered path.
 * All context calls must be serialized. Results survive component/context unload. */
HEARTCODEC_API heartcodec_status_v1 heartcodec_component_set_v1(heartcodec_context *, heartcodec_component_v1,
    const char * model_path, heartcodec_error_v1 *);
HEARTCODEC_API heartcodec_status_v1 heartcodec_component_load_v1(heartcodec_context *, heartcodec_component_v1,
    heartcodec_error_v1 *);
HEARTCODEC_API heartcodec_status_v1 heartcodec_component_unload_v1(heartcodec_context *, heartcodec_component_v1,
    heartcodec_error_v1 *);
HEARTCODEC_API void heartcodec_tensor_init_v1(heartcodec_tensor_v1 *);
HEARTCODEC_API void heartcodec_tensor_free_v1(heartcodec_tensor_v1 *);
HEARTCODEC_API void heartcodec_stereo_init_v1(heartcodec_stereo_v1 *);
HEARTCODEC_API void heartcodec_stereo_free_v1(heartcodec_stereo_v1 *);
HEARTCODEC_API void heartcodec_flow_config_init_v1(heartcodec_flow_config_v1 *);
/* Returns shared RVQ projection [T,512], before conditional feature projection. */
HEARTCODEC_API heartcodec_status_v1 heartcodec_rvq_decode_v1(heartcodec_context *, const heartcodec_tokens_v1 *,
    heartcodec_tensor_v1 *, heartcodec_error_v1 *);
/* Estimator primitive: input [frames,1024] = [x256,context256,mu512]. */
HEARTCODEC_API heartcodec_status_v1 heartcodec_flow_step_v1(heartcodec_context *, const float * input,
    int64_t frames, float time, const char * dump_directory, heartcodec_tensor_v1 *, heartcodec_error_v1 *);
HEARTCODEC_API heartcodec_status_v1 heartcodec_flow_decode_v1(heartcodec_context *, const heartcodec_tokens_v1 *,
    const heartcodec_flow_config_v1 *, const char * dump_directory, heartcodec_tensor_v1 *, heartcodec_error_v1 *);
/* Complete token-to-48kHz stereo decode. Repeats/pads short input, carries latent
 * context between windows, crossfades and trims to token_frames*3840 samples.
 * Unloads SQ before flow, then flow before SQ. context/prefix must be empty.
 * Default windows match upstream 29.76 seconds; outputs use this library's RNG. */
HEARTCODEC_API heartcodec_status_v1 heartcodec_tokens_decode_v1(heartcodec_context *, const heartcodec_tokens_v1 *,
    const heartcodec_flow_config_v1 *, heartcodec_stereo_v1 *, heartcodec_error_v1 *);

#if defined(_WIN32)
# define HEARTCODEC_CALL __cdecl
#else
# define HEARTCODEC_CALL
#endif
#define HEARTCODEC_ABI_VERSION_1 1u
/* Frozen v1 prefix. Future extensions append fields without changing this size. */
typedef struct heartcodec_api_v1 {
    uint32_t size;
    uint32_t abi_version;
    void (HEARTCODEC_CALL * config_init)(heartcodec_config_v1 *);
    void (HEARTCODEC_CALL * error_init)(heartcodec_error_v1 *);
    heartcodec_status_v1 (HEARTCODEC_CALL * context_create)(const heartcodec_config_v1 *, heartcodec_context **, heartcodec_error_v1 *);
    void (HEARTCODEC_CALL * context_destroy)(heartcodec_context *);
    void (HEARTCODEC_CALL * audio_init)(heartcodec_audio_v1 *);
    void (HEARTCODEC_CALL * audio_free)(heartcodec_audio_v1 *);
    void (HEARTCODEC_CALL * sq_latents_init)(heartcodec_sq_latents_v1 *);
    void (HEARTCODEC_CALL * sq_latents_free)(heartcodec_sq_latents_v1 *);
    heartcodec_status_v1 (HEARTCODEC_CALL * sq_encode)(heartcodec_context *, const float *, int64_t, const char *, heartcodec_sq_latents_v1 *, heartcodec_error_v1 *);
    heartcodec_status_v1 (HEARTCODEC_CALL * sq_decode)(heartcodec_context *, const float *, int64_t, const char *, heartcodec_audio_v1 *, heartcodec_error_v1 *);
    heartcodec_status_v1 (HEARTCODEC_CALL * sq_quantize)(const float *, float *, size_t, heartcodec_error_v1 *);
    heartcodec_status_v1 (HEARTCODEC_CALL * get_memory)(heartcodec_context *, heartcodec_memory_v1 *, heartcodec_error_v1 *);
    heartcodec_status_v1 (HEARTCODEC_CALL * component_set)(heartcodec_context *, heartcodec_component_v1, const char *, heartcodec_error_v1 *);
    heartcodec_status_v1 (HEARTCODEC_CALL * component_load)(heartcodec_context *, heartcodec_component_v1, heartcodec_error_v1 *);
    heartcodec_status_v1 (HEARTCODEC_CALL * component_unload)(heartcodec_context *, heartcodec_component_v1, heartcodec_error_v1 *);
    void (HEARTCODEC_CALL * tensor_init)(heartcodec_tensor_v1 *);
    void (HEARTCODEC_CALL * tensor_free)(heartcodec_tensor_v1 *);
    void (HEARTCODEC_CALL * stereo_init)(heartcodec_stereo_v1 *);
    void (HEARTCODEC_CALL * stereo_free)(heartcodec_stereo_v1 *);
    void (HEARTCODEC_CALL * flow_config_init)(heartcodec_flow_config_v1 *);
    heartcodec_status_v1 (HEARTCODEC_CALL * rvq_decode)(heartcodec_context *, const heartcodec_tokens_v1 *, heartcodec_tensor_v1 *, heartcodec_error_v1 *);
    heartcodec_status_v1 (HEARTCODEC_CALL * flow_step)(heartcodec_context *, const float *, int64_t, float, const char *, heartcodec_tensor_v1 *, heartcodec_error_v1 *);
    heartcodec_status_v1 (HEARTCODEC_CALL * flow_decode)(heartcodec_context *, const heartcodec_tokens_v1 *, const heartcodec_flow_config_v1 *, const char *, heartcodec_tensor_v1 *, heartcodec_error_v1 *);
    heartcodec_status_v1 (HEARTCODEC_CALL * tokens_decode)(heartcodec_context *, const heartcodec_tokens_v1 *, const heartcodec_flow_config_v1 *, heartcodec_stereo_v1 *, heartcodec_error_v1 *);
} heartcodec_api_v1;
#define HEARTCODEC_API_V1_SIZE (offsetof(heartcodec_api_v1, tokens_decode) + sizeof(((heartcodec_api_v1 *)0)->tokens_decode))
/* Returns NULL for unsupported ABI; table and strings have library lifetime. */
HEARTCODEC_API const heartcodec_api_v1 * HEARTCODEC_CALL heartcodec_get_api(uint32_t abi_version);

#ifdef __cplusplus
}
#endif
#endif
