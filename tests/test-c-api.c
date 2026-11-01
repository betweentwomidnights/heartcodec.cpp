#include "heartcodec.h"
#include <string.h>

int main(void) {
    const heartcodec_api_v1 * api = heartcodec_get_api(HEARTCODEC_ABI_VERSION_1);
    if (!api || api->size < HEARTCODEC_API_V1_SIZE || api->abi_version != HEARTCODEC_ABI_VERSION_1) return 1;
    if (heartcodec_get_api(0) || heartcodec_get_api(2) || !api->tokens_decode) return 1;
    struct extended_config { heartcodec_config_v1 config; unsigned char tail[16]; } extended;
    memset(&extended, 0xa5, sizeof(extended)); extended.config.size = sizeof(extended);
    api->config_init(&extended.config);
    if (extended.tail[0] != 0xa5 || extended.tail[15] != 0xa5 || extended.config.size != sizeof(extended)) return 1;
    heartcodec_config_v1 short_config;
    memset(&short_config, 0xa5, sizeof(short_config)); short_config.size = 4;
    api->config_init(&short_config);
    if (((const unsigned char *)&short_config)[4] != 0xa5) return 1;
    heartcodec_config_v1 config = {0}; config.size = sizeof(config);
    heartcodec_config_init_v1(&config);
    if (config.cpu_threads != 4 || strcmp(config.backend, "cpu")) return 1;
    heartcodec_error_v1 error = {0}; error.size = sizeof(error);
    heartcodec_error_init_v1(&error);
    heartcodec_audio_v1 audio = {0}; audio.size = sizeof(audio);
    heartcodec_audio_init_v1(&audio);
    heartcodec_context * context = NULL;
    if (heartcodec_context_create_v1(&config, &context, &error) != HEARTCODEC_INVALID_ARGUMENT_V1) return 1;
    if (!error.message[0] || context != NULL) return 1;
    if (heartcodec_sq_decode_v1(NULL, NULL, 0, NULL, &audio, &error) != HEARTCODEC_INVALID_ARGUMENT_V1) return 1;
    float values[] = {-1.0f, 0.0f, 1.0f};
    if (heartcodec_sq_quantize_v1(values, values, 3, &error) != HEARTCODEC_OK_V1) return 1;
    if (error.status != HEARTCODEC_OK_V1 || error.message[0]) return 1;
    if (values[0] != -1 || values[1] != 0 || values[2] != 1) return 1;
    heartcodec_audio_free_v1(&audio);
    heartcodec_sq_latents_v1 latent = {0}; latent.size = sizeof(latent);
    heartcodec_sq_latents_init_v1(&latent);
    if (heartcodec_sq_encode_v1(NULL, values, 3, NULL, &latent, &error) != HEARTCODEC_INVALID_ARGUMENT_V1) return 1;
    heartcodec_sq_latents_free_v1(&latent);
    heartcodec_sq_latents_free_v1(&latent);
    heartcodec_audio_free_v1(&audio);
    heartcodec_context_destroy_v1(NULL);
    heartcodec_flow_config_v1 flow = {0}; flow.size = sizeof(flow);
    api->flow_config_init(&flow);
    if (flow.steps != 10 || flow.guidance != 1.25f || flow.segment_codes != 372) return 1;
    heartcodec_tensor_v1 tensor = {0}; tensor.size = sizeof(tensor);
    api->tensor_init(&tensor);
    if (api->rvq_decode(NULL, NULL, &tensor, &error) != HEARTCODEC_INVALID_ARGUMENT_V1) return 1;
    if (api->component_unload(NULL, HEARTCODEC_SQ_V1, &error) != HEARTCODEC_INVALID_ARGUMENT_V1) return 1;
    api->tensor_free(&tensor); api->tensor_free(&tensor);
    heartcodec_stereo_v1 stereo = {0}; stereo.size = sizeof(stereo);
    api->stereo_init(&stereo); api->stereo_free(&stereo); api->stereo_free(&stereo);
    return 0;
}
