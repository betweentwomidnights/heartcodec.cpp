#include "heartcodec_work.h"
#include <cstdio>
#include <cstring>
#include <vector>
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"work check failed at line %d: %s (%s)\n",__LINE__,#x,e.message);return 1;}}while(0)
struct State {int polls=0,limit=0;bool cancel=false;};
static int32_t HEARTCODEC_CALL cancel(void * p){auto & s=*static_cast<State *>(p);return s.cancel||(s.limit&&++s.polls>=s.limit);}
static void HEARTCODEC_CALL progress(void * p,const heartcodec_work_progress_v1 *){static_cast<State *>(p)->cancel=true;}
int main(int argc,char ** argv){
    heartcodec_error_v1 e{};e.size=sizeof(e);const auto * api=heartcodec_get_work_api(1);CHECK(api&&api->size>=HEARTCODEC_WORK_API_V1_SIZE&&api->version==1&&!heartcodec_get_work_api(0)&&!heartcodec_get_work_api(2));
    struct {heartcodec_work_control_v1 c;uint64_t tail;} ext{};ext.c.size=sizeof(ext);ext.tail=123;api->control_init(&ext.c);CHECK(ext.c.size==sizeof(ext)&&ext.tail==123&&!ext.c.should_cancel&&!ext.c.on_progress);
    CHECK(api->encode(nullptr,nullptr,nullptr,&ext.c,nullptr,&e)==1&&api->tokens_decode(nullptr,nullptr,nullptr,&ext.c,nullptr,&e)==1);
    if(argc==1)return 0;CHECK(argc==5);
    heartcodec_config_v1 config{};config.size=sizeof(config);heartcodec_config_init_v1(&config);config.backend=argv[4];config.model_path=argv[1];config.cpu_threads=4;
    heartcodec_encoder_context * enc=nullptr;CHECK(!heartcodec_encoder_context_create_v1(&config,&enc,&e));
    std::vector<float> pcm(3840,0);heartcodec_waveform_v1 wave{sizeof(wave),pcm.data(),3840,48000,1,3840,1};heartcodec_encoded_tokens_v1 codes{};codes.size=sizeof(codes);heartcodec_encoded_tokens_init_v1(&codes);const auto empty=codes;
    State state;ext.c.should_cancel=cancel;ext.c.user=&state;state.cancel=true;CHECK(api->encode(enc,&wave,nullptr,&ext.c,&codes,&e)==3&&!std::memcmp(&empty,&codes,sizeof(codes)));
    state={};state.limit=8;CHECK(api->encode(enc,&wave,nullptr,&ext.c,&codes,&e)==3&&!std::memcmp(&empty,&codes,sizeof(codes)));
    state={};ext.c.on_progress=progress;CHECK(api->encode(enc,&wave,nullptr,&ext.c,&codes,&e)==3&&!std::memcmp(&empty,&codes,sizeof(codes)));
    // The observer pointer must have been cleared: legacy calls on this same
    // context still work even while the old observer requests cancellation.
    CHECK(!heartcodec_encode_v1(enc,&wave,nullptr,&codes,&e));std::vector<int32_t> expected(codes.data,codes.data+8);heartcodec_encoded_tokens_free_v1(&codes);
    state={};ext.c.on_progress=nullptr;CHECK(!api->encode(enc,&wave,nullptr,&ext.c,&codes,&e)&&!std::memcmp(expected.data(),codes.data,32));heartcodec_encoder_context_destroy_v1(enc);
    config.model_path=argv[2];heartcodec_context * dec=nullptr;CHECK(!heartcodec_context_create_v1(&config,&dec,&e));CHECK(!heartcodec_component_set_v1(dec,HEARTCODEC_SQ_V1,argv[3],&e));
    heartcodec_tokens_v1 tokens{sizeof(tokens),codes.data,1,1,8};heartcodec_flow_config_v1 flow{};flow.size=sizeof(flow);heartcodec_flow_config_init_v1(&flow);flow.steps=1;flow.segment_codes=93;
    heartcodec_stereo_v1 audio{};audio.size=sizeof(audio);heartcodec_stereo_init_v1(&audio);auto empty_audio=audio;
    state={};ext.c.on_progress=progress;CHECK(api->tokens_decode(dec,&tokens,&flow,&ext.c,&audio,&e)==3&&!std::memcmp(&empty_audio,&audio,sizeof(audio)));
    CHECK(!heartcodec_tokens_decode_v1(dec,&tokens,&flow,&audio,&e));std::vector<float> reference(audio.samples,audio.samples+audio.num_samples*2);heartcodec_stereo_free_v1(&audio);
    state={};ext.c.on_progress=nullptr;CHECK(!api->tokens_decode(dec,&tokens,&flow,&ext.c,&audio,&e)&&!std::memcmp(reference.data(),audio.samples,reference.size()*4));
    heartcodec_context_destroy_v1(dec);heartcodec_stereo_free_v1(&audio);heartcodec_encoded_tokens_free_v1(&codes);std::puts("controlled codec cancellation, context reuse and legacy replay passed");return 0;
}
