#include "heartcodec_rng.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
int main() {
    if(heartcodec_get_rng_api(0)||heartcodec_get_rng_api(2))return 1;
    const auto * api=heartcodec_get_rng_api(1);if(!api||api->size<HEARTCODEC_RNG_API_V1_SIZE||api->version!=1)return 1;
    heartcodec_rng_config_v1 c{};c.size=sizeof(c);api->config_init(&c);
    heartcodec_rng_result_v1 r{};r.size=sizeof(r);api->result_init(&r);
    heartcodec_error_v1 e{};e.size=sizeof(e);
    for(int test=0;test<5;++test) {
        api->config_init(&c);c.count=16;
        if(test==0)c.offset=1;
        if(test==1)c.count=-1;
        if(test==2)c.distribution=2;
        if(test==3)c.draws=-1;
        if(test==4){c.count=INT32_MAX;c.draws=INT64_MAX;}
        r.next_offset=123;auto before=r;
        if(api->generate(&c,&r,&e)!=HEARTCODEC_INVALID_ARGUMENT_V1||std::memcmp(&before,&r,sizeof(r)))return 1;
    }
    api->config_init(&c);c.count=16;float occupied=1;r.data=&occupied;auto before=r;
    if(api->generate(&c,&r,&e)!=HEARTCODEC_INVALID_ARGUMENT_V1||std::memcmp(&before,&r,sizeof(r)))return 1;
    r.data=nullptr;api->result_free(&r);api->result_free(&r);
    api->config_init(&c);c.count=16;
    const auto status=api->generate(&c,&r,&e);
    if(status==HEARTCODEC_OK_V1) {
        if(!r.data||r.count!=16||r.draws!=1||r.next_offset!=4||r.grid_blocks!=1||!r.device_max_blocks)return 1;
        for(int i=0;i<16;++i)if(!std::isfinite(r.data[i]))return 1;
        api->result_free(&r);
        // A streaming job requests eight heads per frame, whereas the frozen
        // driver requests every head at once. Both must advance the same stream.
        for(int distribution=0;distribution<2;++distribution) {
            api->config_init(&c);c.distribution=distribution;c.count=distribution?8197:190464;
            c.draws=24;c.seed=UINT64_C(0xfedcba9812345678);c.offset=12;c.max_blocks=288;
            if(api->generate(&c,&r,&e))return 1;
            heartcodec_rng_result_v1 part{};part.size=sizeof(part);uint64_t offset=c.offset;
            for(int start=0;start<24;start+=8) {
                auto chunk=c;chunk.draws=8;chunk.offset=offset;
                if(api->generate(&chunk,&part,&e)||std::memcmp(r.data+int64_t(start)*c.count,part.data,size_t(c.count)*8*sizeof(float)))return 1;
                offset=part.next_offset;api->result_free(&part);
            }
            if(offset!=r.next_offset)return 1;
            api->result_free(&r);
        }
        std::puts("Philox bulk/chunked normal and exponential streams are bit-identical (full 64-bit seed, cap 288)");
        api->config_init(&c);c.count=16;
        api->result_free(&r);c.offset=UINT64_MAX-3;r.next_offset=123;before=r;
        if(api->generate(&c,&r,&e)!=HEARTCODEC_INVALID_ARGUMENT_V1||std::memcmp(&before,&r,sizeof(r)))return 1;
    } else {
#ifdef HEARTCODEC_TEST_REQUIRE_CUDA
        std::fprintf(stderr,"CUDA RNG generation failed: %s\n",e.message);return 1;
#else
        if(status!=HEARTCODEC_ERROR_V1||r.data||!e.message[0])return 1;
        std::puts("CPU build: Torch CUDA generation unavailable, no fallback");
#endif
    }
    api->result_free(&r);return 0;
}
