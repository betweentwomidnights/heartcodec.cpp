#ifndef HEARTCODEC_RNG_H
#define HEARTCODEC_RNG_H
#include "heartcodec.h"
#ifdef __cplusplus
extern "C" {
#endif
enum { HEARTCODEC_RNG_NORMAL_V1=0, HEARTCODEC_RNG_EXPONENTIAL_V1=1 };
typedef struct heartcodec_rng_config_v1 {
    uint32_t size;
    int32_t distribution; /* F32 torch CUDA normal or exponential(lambda=1) */
    uint64_t seed;
    uint64_t offset; /* Philox 32-bit output offset; multiple of four */
    int64_t count; /* contiguous elements per independent Torch kernel call */
    int64_t draws; /* consecutive calls of the same shape */
    uint32_t max_blocks; /* 0: query device; otherwise emulate target GPU cap */
    int32_t device; /* explicit CUDA ordinal; no CPU fallback */
} heartcodec_rng_config_v1;
typedef struct heartcodec_rng_result_v1 {
    uint32_t size;
    float * data; /* library-owned [draws,count], host F32 */
    int64_t count;
    int64_t draws;
    uint64_t next_offset;
    uint32_t grid_blocks;
    uint32_t device_max_blocks;
} heartcodec_rng_result_v1;
HEARTCODEC_API void heartcodec_rng_config_init_v1(heartcodec_rng_config_v1 *);
HEARTCODEC_API void heartcodec_rng_result_init_v1(heartcodec_rng_result_v1 *);
HEARTCODEC_API void heartcodec_rng_result_free_v1(heartcodec_rng_result_v1 *);
/* Opt-in capability. Preserves the frozen core/flow APIs and legacy seed policy.
 * Generates the pinned Torch 2.7.1 contiguous F32 CUDA distributions using
 * Philox4x32-10, cuRAND-compatible transforms, 256-thread grid-stride mapping
 * and generator offset advancement.
 * Host output survives context/model unloading. No model or GGML is required.
 * Invalid input/occupied output and runtime errors leave the result unchanged. */
HEARTCODEC_API heartcodec_status_v1 heartcodec_rng_generate_v1(
    const heartcodec_rng_config_v1 *,heartcodec_rng_result_v1 *,heartcodec_error_v1 *);
typedef struct heartcodec_rng_api_v1 {
    uint32_t size;
    uint32_t version;
    void (*config_init)(heartcodec_rng_config_v1 *);
    void (*result_init)(heartcodec_rng_result_v1 *);
    void (*result_free)(heartcodec_rng_result_v1 *);
    heartcodec_status_v1 (*generate)(const heartcodec_rng_config_v1 *,heartcodec_rng_result_v1 *,heartcodec_error_v1 *);
} heartcodec_rng_api_v1;
#define HEARTCODEC_RNG_API_V1_SIZE (offsetof(heartcodec_rng_api_v1,generate)+sizeof(((heartcodec_rng_api_v1 *)0)->generate))
HEARTCODEC_API const heartcodec_rng_api_v1 * heartcodec_get_rng_api(uint32_t version);
#ifdef __cplusplus
}
#endif
#endif
