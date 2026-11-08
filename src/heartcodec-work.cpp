#include "heartcodec-internal.h"
#include <cstring>
extern "C" {
void HEARTCODEC_CALL heartcodec_work_control_init_v1(heartcodec_work_control_v1 * c) {
    if(!c||c->size<sizeof(*c))return;const auto size=c->size;std::memset(c,0,sizeof(*c));c->size=size;
}
int32_t HEARTCODEC_CALL heartcodec_tokens_decode_controlled_v1(heartcodec_context * c,const heartcodec_tokens_v1 * tokens,
        const heartcodec_flow_config_v1 * config,const heartcodec_work_control_v1 * control,heartcodec_stereo_v1 * out,heartcodec_error_v1 * e) {
    if(!c||!control||control->size<sizeof(*control)||c->work)
        return heartcodec::error(e,HEARTCODEC_INVALID_ARGUMENT_V1,"Invalid controlled decode call or reentrant context");
    heartcodec::Work work{control};
    struct Scope{heartcodec_context * c;~Scope(){c->work=nullptr;for(auto & m:c->models)if(m)m->work=nullptr;}} scope{c};
    c->work=&work;for(auto & m:c->models)if(m)m->work=&work;
    const auto status=heartcodec_tokens_decode_v1(c,tokens,config,out,e);
    return work.cancelled?heartcodec::error(e,static_cast<heartcodec_status_v1>(HEARTCODEC_WORK_CANCELLED_V1),"HeartCodec decoding cancelled"):status;
}
const heartcodec_work_api_v1 * HEARTCODEC_CALL heartcodec_get_work_api(uint32_t version) {
    static const heartcodec_work_api_v1 api={HEARTCODEC_WORK_API_V1_SIZE,1,heartcodec_work_control_init_v1,
        heartcodec_encode_controlled_v1,heartcodec_tokens_decode_controlled_v1};
    return version==1?&api:nullptr;
}
}
