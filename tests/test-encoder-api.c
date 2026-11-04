#include "heartcodec_encoder.h"
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

int main(int argc,char ** argv) {
    const heartcodec_encoder_api_v1 * api=heartcodec_get_encoder_api(1);
    assert(api&&api->version==1&&api->size>=HEARTCODEC_ENCODER_API_V1_SIZE&&api->core==heartcodec_get_api(1));
    assert(!heartcodec_get_encoder_api(0)&&!heartcodec_get_encoder_api(2));
    struct {heartcodec_encoded_tokens_v1 value;uint64_t tail;} extended;
    memset(&extended,0,sizeof(extended));extended.value.size=sizeof(extended);extended.tail=UINT64_C(0x123456789abcdef0);
    api->tokens_init(&extended.value);assert(extended.value.size==sizeof(extended)&&extended.tail==UINT64_C(0x123456789abcdef0));
    api->tokens_free(&extended.value);api->tokens_free(&extended.value);assert(extended.tail==UINT64_C(0x123456789abcdef0));
    heartcodec_error_v1 error;error.size=sizeof(error);
    heartcodec_encoder_context * context=0;
    assert(api->context_create(0,&context,&error)==HEARTCODEC_INVALID_ARGUMENT_V1&&context==0);
    heartcodec_config_v1 config;config.size=sizeof(config);api->config_init(&config);
    config.model_path="unused.gguf";
    context=(heartcodec_encoder_context *)(uintptr_t)1;
    assert(api->context_create(&config,&context,&error)==HEARTCODEC_INVALID_ARGUMENT_V1&&context==(heartcodec_encoder_context *)(uintptr_t)1);
    context=0;
    heartcodec_encoded_tokens_v1 out;out.size=sizeof(out);api->tokens_init(&out);
    out.frames=123;
    assert(api->encode(0,0,0,&out,&error)==HEARTCODEC_INVALID_ARGUMENT_V1&&out.frames==123&&!out.data);
    api->tokens_init(&out);api->context_destroy(0);
    if(argc>1) {
        float pcm[3840]={0};
        config.model_path=argv[1];config.backend=argc>2?argv[2]:"cpu";config.cpu_threads=8;
        assert(api->context_create(&config,&context,&error)==HEARTCODEC_OK_V1&&context);
        heartcodec_waveform_v1 waveform={sizeof(waveform),pcm,3840,48000,1,3840,1};
        heartcodec_waveform_v1 bad=waveform;bad.sample_rate=12345;
        assert(api->encode(context,&bad,0,&out,&error)==HEARTCODEC_INVALID_ARGUMENT_V1&&!out.data);
        bad=waveform;bad.num_samples=1;
        assert(api->encode(context,&bad,0,&out,&error)==HEARTCODEC_INVALID_ARGUMENT_V1&&!out.data);
        bad=waveform;bad.sample_stride=INT64_MAX;
        assert(api->encode(context,&bad,0,&out,&error)==HEARTCODEC_INVALID_ARGUMENT_V1&&!out.data);
        bad=waveform;bad.channels=3;
        assert(api->encode(context,&bad,0,&out,&error)==HEARTCODEC_INVALID_ARGUMENT_V1&&!out.data);
        bad=waveform;bad.size=sizeof(bad)-1;
        assert(api->encode(context,&bad,0,&out,&error)==HEARTCODEC_INVALID_ARGUMENT_V1&&!out.data);
        pcm[0]=NAN;
        assert(api->encode(context,&waveform,0,&out,&error)==HEARTCODEC_INVALID_ARGUMENT_V1&&!out.data);
        pcm[0]=0;
        out.frames=1;
        assert(api->encode(context,&waveform,0,&out,&error)==HEARTCODEC_INVALID_ARGUMENT_V1&&out.frames==1);
        api->tokens_init(&out);
        assert(api->encode(context,&waveform,0,&out,&error)==HEARTCODEC_OK_V1&&out.data&&out.frames==1&&out.resampled_samples==1920);
        api->context_destroy(context);context=0;
        for(int i=0;i<8;++i)assert(out.data[i]>=0&&out.data[i]<8192);
        for(int i=0;i<3840;++i)assert(pcm[i]==0);
        api->tokens_free(&out);api->tokens_free(&out);
        assert(!out.data&&!out.frames&&!out.resampled_samples);
    }
    return 0;
}
