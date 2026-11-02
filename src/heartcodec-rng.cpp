#include "heartcodec-internal.h"
#include "heartcodec_rng.h"
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#ifdef HEARTCODEC_RNG_CUDA
namespace heartcodec { void rng_cuda(const heartcodec_rng_config_v1 &,float *,uint64_t &,uint32_t &,uint32_t &); }
#endif
namespace {
template<class T> void initialize(T * p) {
    if(!p||p->size<sizeof(T))return;
    const auto size=p->size;std::memset(p,0,sizeof(T));p->size=size;
}
}
extern "C" {
void heartcodec_rng_config_init_v1(heartcodec_rng_config_v1 * p) {
    initialize(p);if(p&&p->size>=sizeof(*p)){p->seed=1701;p->draws=1;}
}
void heartcodec_rng_result_init_v1(heartcodec_rng_result_v1 * p){initialize(p);}
void heartcodec_rng_result_free_v1(heartcodec_rng_result_v1 * p) {
    if(!p||p->size<sizeof(*p))return;delete[] p->data;initialize(p);
}
heartcodec_status_v1 heartcodec_rng_generate_v1(const heartcodec_rng_config_v1 * c,heartcodec_rng_result_v1 * out,heartcodec_error_v1 * error) {
    try {
        if(!c||c->size<sizeof(*c)||!out||out->size<sizeof(*out)||out->data||c->distribution<0||c->distribution>1||
           c->count<0||c->count>INT32_MAX||c->draws<0||c->offset%4||c->device<0||c->max_blocks>1048576||
           (c->count&&uint64_t(c->draws)>std::numeric_limits<size_t>::max()/sizeof(float)/uint64_t(c->count)))
            throw std::invalid_argument("Invalid Torch CUDA RNG configuration or occupied result");
#ifdef HEARTCODEC_RNG_CUDA
        const size_t size=size_t(c->count)*size_t(c->draws);
        auto data=size?std::make_unique<float[]>(size):nullptr;
        uint64_t next=c->offset;uint32_t grid=0,cap=0;
        heartcodec::rng_cuda(*c,data.get(),next,grid,cap);
        out->data=data.release();out->count=c->count;out->draws=c->draws;out->next_offset=next;out->grid_blocks=grid;out->device_max_blocks=cap;
        return heartcodec::error(error,HEARTCODEC_OK_V1,"");
#else
        throw std::runtime_error("Torch CUDA RNG requires a CUDA-enabled HeartCodec build");
#endif
    }catch(const std::invalid_argument & e){return heartcodec::error(error,HEARTCODEC_INVALID_ARGUMENT_V1,e.what());}
    catch(const std::exception & e){return heartcodec::error(error,HEARTCODEC_ERROR_V1,e.what());}
}
const heartcodec_rng_api_v1 * heartcodec_get_rng_api(uint32_t version) {
    static const heartcodec_rng_api_v1 api={uint32_t(HEARTCODEC_RNG_API_V1_SIZE),1,heartcodec_rng_config_init_v1,
        heartcodec_rng_result_init_v1,heartcodec_rng_result_free_v1,heartcodec_rng_generate_v1};
    return version==1?&api:nullptr;
}
}
