#ifndef HEARTCODEC_ENCODER_H
#define HEARTCODEC_ENCODER_H
#include "heartcodec.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Separate negotiated capability: core, SQ, flow and RNG layouts are unchanged.
 * A full FP32 waveform encoder GGUF is required; SQ latents are not RVQ codes. */
typedef struct heartcodec_encoder_context heartcodec_encoder_context;
typedef struct heartcodec_waveform_v1 {
    uint32_t size;
    const float * samples; /* borrowed finite PCM; normalized nominally to [-1,1] */
    int64_t num_samples; /* per channel */
    int32_t sample_rate;
    int32_t channels; /* 1 or 2; mono is duplicated */
    int64_t channel_stride; /* float elements; planar: num_samples, interleaved: 1 */
    int64_t sample_stride; /* float elements; planar: 1, interleaved: channels */
} heartcodec_waveform_v1;
typedef struct heartcodec_encoded_tokens_v1 {
    uint32_t size;
    int32_t * data; /* library-owned frame-major [frames,8], each code in [0,8191] */
    int64_t frames; /* complete original-clock 80-ms frames only; partial tail discarded */
    int64_t resampled_samples; /* per-channel 24-kHz sample count, before chunk padding */
} heartcodec_encoded_tokens_v1;

/* Supported input rates: 16000,22050,24000,32000,44100,48000,88200,96000,192000.
 * Input must contain at least one complete 80-ms frame. Resample the available
 * waveform before 29.76-second chunking; zero-pad each final partial chunk.
 * Calls on a context must be serialized and must run outside realtime callbacks.
 * Results survive context destruction; free before unloading the library.
 * Invalid input/occupied output leaves results and borrowed PCM unchanged. */
HEARTCODEC_API void HEARTCODEC_CALL heartcodec_encoded_tokens_init_v1(heartcodec_encoded_tokens_v1 *);
HEARTCODEC_API void HEARTCODEC_CALL heartcodec_encoded_tokens_free_v1(heartcodec_encoded_tokens_v1 *);
HEARTCODEC_API heartcodec_status_v1 HEARTCODEC_CALL heartcodec_encoder_context_create_v1(
    const heartcodec_config_v1 *,heartcodec_encoder_context **,heartcodec_error_v1 *);
HEARTCODEC_API void HEARTCODEC_CALL heartcodec_encoder_context_destroy_v1(heartcodec_encoder_context *);
HEARTCODEC_API heartcodec_status_v1 HEARTCODEC_CALL heartcodec_encode_v1(
    heartcodec_encoder_context *,const heartcodec_waveform_v1 *,const char * dump_directory,
    heartcodec_encoded_tokens_v1 *,heartcodec_error_v1 *);
HEARTCODEC_API heartcodec_status_v1 HEARTCODEC_CALL heartcodec_encoder_get_memory_v1(
    heartcodec_encoder_context *,heartcodec_memory_v1 *,heartcodec_error_v1 *);

typedef struct heartcodec_encoder_api_v1 {
    uint32_t size;
    uint32_t version;
    const heartcodec_api_v1 * core;
    void (HEARTCODEC_CALL * config_init)(heartcodec_config_v1 *);
    heartcodec_status_v1 (HEARTCODEC_CALL * context_create)(const heartcodec_config_v1 *,heartcodec_encoder_context **,heartcodec_error_v1 *);
    void (HEARTCODEC_CALL * context_destroy)(heartcodec_encoder_context *);
    void (HEARTCODEC_CALL * tokens_init)(heartcodec_encoded_tokens_v1 *);
    void (HEARTCODEC_CALL * tokens_free)(heartcodec_encoded_tokens_v1 *);
    heartcodec_status_v1 (HEARTCODEC_CALL * encode)(heartcodec_encoder_context *,const heartcodec_waveform_v1 *,const char *,heartcodec_encoded_tokens_v1 *,heartcodec_error_v1 *);
    heartcodec_status_v1 (HEARTCODEC_CALL * get_memory)(heartcodec_encoder_context *,heartcodec_memory_v1 *,heartcodec_error_v1 *);
} heartcodec_encoder_api_v1;
#define HEARTCODEC_ENCODER_API_V1_SIZE (offsetof(heartcodec_encoder_api_v1,get_memory)+sizeof(((heartcodec_encoder_api_v1 *)0)->get_memory))
HEARTCODEC_API const heartcodec_encoder_api_v1 * HEARTCODEC_CALL heartcodec_get_encoder_api(uint32_t version);
#ifdef __cplusplus
}
#endif
#endif
