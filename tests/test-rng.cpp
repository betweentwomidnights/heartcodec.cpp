#include "heartcodec_rng.h"
#include <cstdint>
#include <cstring>
#include <cmath>
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
        api->result_free(&r);c.offset=UINT64_MAX-3;r.next_offset=123;before=r;
        if(api->generate(&c,&r,&e)!=HEARTCODEC_INVALID_ARGUMENT_V1||std::memcmp(&before,&r,sizeof(r)))return 1;
    } else if(status!=HEARTCODEC_ERROR_V1||r.data||!e.message[0])return 1;
    api->result_free(&r);return 0;
}
