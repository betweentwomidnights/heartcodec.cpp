#include "heartcodec.h"
#include <string.h>

int main(void) {
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
    return 0;
}
