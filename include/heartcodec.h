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

#ifdef __cplusplus
}
#endif
#endif
