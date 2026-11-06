#ifndef HEARTCODEC_WORK_H
#define HEARTCODEC_WORK_H
#include "heartcodec_encoder.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Separate capability. Existing core, flow, encoder and RNG layouts/defaults
 * are unchanged. Controlled calls use status 3 only for observed cancellation. */
enum { HEARTCODEC_WORK_CANCELLED_V1=3 };
enum { HEARTCODEC_WORK_ENCODING_V1=0, HEARTCODEC_WORK_FLOW_V1=1, HEARTCODEC_WORK_SQ_V1=2 };
typedef struct heartcodec_work_progress_v1 {
    uint32_t size;
    int32_t stage;
    int64_t completed;
    int64_t total;
} heartcodec_work_progress_v1;
typedef int32_t (HEARTCODEC_CALL * heartcodec_cancel_callback_v1)(void * user);
typedef void (HEARTCODEC_CALL * heartcodec_work_callback_v1)(void * user,const heartcodec_work_progress_v1 *);
typedef struct heartcodec_work_control_v1 {
    uint32_t size;
    heartcodec_cancel_callback_v1 should_cancel;
    heartcodec_work_callback_v1 on_progress;
    void * user;
} heartcodec_work_control_v1;
/* Synchronous, serialized context calls. Callbacks run on the calling worker,
 * must not throw or reenter the context, and are borrowed only for this call.
 * Cancellation is cooperative between graphs/chunks/flow steps/SQ channels;
 * an executing backend graph is not interrupted. Output stays empty/unchanged
 * on cancellation or failure. Contexts remain usable after the call. */
HEARTCODEC_API void HEARTCODEC_CALL heartcodec_work_control_init_v1(heartcodec_work_control_v1 *);
HEARTCODEC_API int32_t HEARTCODEC_CALL heartcodec_encode_controlled_v1(heartcodec_encoder_context *,const heartcodec_waveform_v1 *,const char *,const heartcodec_work_control_v1 *,heartcodec_encoded_tokens_v1 *,heartcodec_error_v1 *);
HEARTCODEC_API int32_t HEARTCODEC_CALL heartcodec_tokens_decode_controlled_v1(heartcodec_context *,const heartcodec_tokens_v1 *,const heartcodec_flow_config_v1 *,const heartcodec_work_control_v1 *,heartcodec_stereo_v1 *,heartcodec_error_v1 *);
typedef struct heartcodec_work_api_v1 {
    uint32_t size;
    uint32_t version;
    void (HEARTCODEC_CALL * control_init)(heartcodec_work_control_v1 *);
    int32_t (HEARTCODEC_CALL * encode)(heartcodec_encoder_context *,const heartcodec_waveform_v1 *,const char *,const heartcodec_work_control_v1 *,heartcodec_encoded_tokens_v1 *,heartcodec_error_v1 *);
    int32_t (HEARTCODEC_CALL * tokens_decode)(heartcodec_context *,const heartcodec_tokens_v1 *,const heartcodec_flow_config_v1 *,const heartcodec_work_control_v1 *,heartcodec_stereo_v1 *,heartcodec_error_v1 *);
} heartcodec_work_api_v1;
#define HEARTCODEC_WORK_API_V1_SIZE (offsetof(heartcodec_work_api_v1,tokens_decode)+sizeof(((heartcodec_work_api_v1 *)0)->tokens_decode))
HEARTCODEC_API const heartcodec_work_api_v1 * HEARTCODEC_CALL heartcodec_get_work_api(uint32_t version);
#ifdef __cplusplus
}
#endif
#endif
