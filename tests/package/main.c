#include <heartcodec.h>
#include <heartcodec_work.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

int main(int argc, char ** argv) {
    const heartcodec_work_api_v1 * work=heartcodec_get_work_api(1);
    if(!work||work->size<HEARTCODEC_WORK_API_V1_SIZE||heartcodec_get_work_api(99))return 1;
    const heartcodec_api_v1 * api = heartcodec_get_api(HEARTCODEC_ABI_VERSION_1);
    if (!api || api->size < HEARTCODEC_API_V1_SIZE || heartcodec_get_api(99)) return 1;
    heartcodec_config_v1 config = {0}; config.size = sizeof(config); api->config_init(&config);
    heartcodec_error_v1 error = {0}; error.size = sizeof(error); api->error_init(&error);
    if (argc > 1) {
        heartcodec_context * context = NULL;
        config.model_path = argv[1];
        if (api->context_create(&config, &context, &error)) { puts(error.message); return 1; }
        heartcodec_memory_v1 memory = {0}; memory.size = sizeof(memory);
        if (api->get_memory(context, &memory, &error) || !memory.weight_buffer_bytes) return 1;
        if (api->component_unload(context, HEARTCODEC_SQ_V1, &error)) return 1;
        if (api->get_memory(context, &memory, &error) || memory.weight_buffer_bytes) return 1;
        if (api->component_load(context, HEARTCODEC_SQ_V1, &error)) return 1;
        if (api->get_memory(context, &memory, &error) || !memory.weight_buffer_bytes) return 1;
        if (argc > 2) {
            int32_t interleaved[] = {0,8191,1,8190,42,314,4096,7, 11,17,23,29,31,37,41,43};
            int32_t planar[16];
            int q, f;
            for (q = 0; q < 8; ++q) for (f = 0; f < 2; ++f) planar[q * 2 + f] = interleaved[f * 8 + q];
            heartcodec_tokens_v1 tokens = {sizeof(tokens), interleaved, 2, 1, 8};
            heartcodec_tensor_v1 first = {0}, second = {0}; first.size = sizeof(first); second.size = sizeof(second);
            api->tensor_init(&first); api->tensor_init(&second);
            if (api->component_unload(context, HEARTCODEC_SQ_V1, &error) ||
                api->component_set(context, HEARTCODEC_FLOW_V1, argv[2], &error)) return 1;
            if (api->rvq_decode(context, &tokens, &first, &error)) { puts(error.message); return 1; }
            tokens.data = planar; tokens.codebook_stride = 2; tokens.frame_stride = 1;
            if (api->rvq_decode(context, &tokens, &second, &error)) return 1;
            if (first.frames != 2 || first.channels != 512 || memcmp(first.data, second.data, 1024 * sizeof(float))) return 1;
            api->tensor_free(&second);
            if (api->component_unload(context, HEARTCODEC_FLOW_V1, &error)) return 1;
            if (api->get_memory(context, &memory, &error) || memory.weight_buffer_bytes) return 1;
            planar[0] = -1;
            if (api->rvq_decode(context, &tokens, &second, &error) != HEARTCODEC_INVALID_ARGUMENT_V1) return 1;
            api->context_destroy(context); context = NULL;
            if (!isfinite(first.data[0])) return 1; /* Result outlives model/context. */
            api->tensor_free(&first); api->tensor_free(&first);
        }
        api->context_destroy(context);
    }
    puts("C consumer and negotiated ABI passed");
    return 0;
}
