#include "heartcodec_encoder.h"
#include "encoder-features.h"
#include <limits>
#include <iomanip>
#include <sstream>

struct heartcodec_encoder_context {heartcodec::Model model;};
namespace {
template<class F> heartcodec_status_v1 guarded(heartcodec_error_v1 * error,F && f) {
    try {f();return heartcodec::error(error,HEARTCODEC_OK_V1,"");}
    catch(const std::invalid_argument & e){return heartcodec::error(error,HEARTCODEC_INVALID_ARGUMENT_V1,e.what());}
    catch(const std::exception & e){return heartcodec::error(error,HEARTCODEC_ERROR_V1,e.what());}
    catch(...){return heartcodec::error(error,HEARTCODEC_ERROR_V1,"Unknown waveform encoder failure");}
}
bool rate_supported(int rate) {
    for(int r:{16000,22050,24000,32000,44100,48000,88200,96000,192000})if(r==rate)return true;return false;
}
}
extern "C" {
void HEARTCODEC_CALL heartcodec_encoded_tokens_init_v1(heartcodec_encoded_tokens_v1 * out) {
    if(!out||out->size<sizeof(*out))return;out->data=nullptr;out->frames=0;out->resampled_samples=0;
}
void HEARTCODEC_CALL heartcodec_encoded_tokens_free_v1(heartcodec_encoded_tokens_v1 * out) {
    if(!out||out->size<sizeof(*out))return;delete[]out->data;heartcodec_encoded_tokens_init_v1(out);
}
heartcodec_status_v1 HEARTCODEC_CALL heartcodec_encoder_context_create_v1(const heartcodec_config_v1 * config,
        heartcodec_encoder_context ** out,heartcodec_error_v1 * error) {
    return guarded(error,[&]{
        if(!config||config->size<sizeof(*config)||!config->model_path||!config->backend||!out||*out||
           config->cpu_threads<1||config->cpu_threads>1024)throw std::invalid_argument("Invalid or occupied waveform encoder context");
        auto context=std::make_unique<heartcodec_encoder_context>();
        context->model.load(config->model_path,config->backend,config->cpu_threads);
        if(!context->model.audio_encoder)throw std::invalid_argument("Expected heartcodec-encoder GGUF");
        *out=context.release();
    });
}
void HEARTCODEC_CALL heartcodec_encoder_context_destroy_v1(heartcodec_encoder_context * context){delete context;}
heartcodec_status_v1 HEARTCODEC_CALL heartcodec_encode_v1(heartcodec_encoder_context * context,
        const heartcodec_waveform_v1 * input,const char * directory,heartcodec_encoded_tokens_v1 * out,heartcodec_error_v1 * error) {
    return guarded(error,[&]{
        if(!context||!input||input->size<sizeof(*input)||!input->samples||!out||out->size<sizeof(*out)||
           out->data||out->frames||out->resampled_samples||input->num_samples<1||input->num_samples>INT32_MAX||
           !rate_supported(input->sample_rate)||(input->channels!=1&&input->channels!=2)||input->sample_stride<1||input->channel_stride<1)
            throw std::invalid_argument("Invalid waveform or occupied token result");
        const int64_t samples=input->num_samples;
        const int64_t frames=samples*25/(int64_t(input->sample_rate)*2);
        if(frames<1)throw std::invalid_argument("Waveform must contain a complete 80-ms RVQ frame");
        const auto limit=int64_t(std::numeric_limits<ptrdiff_t>::max()/sizeof(float));
        if((samples-1)>limit/input->sample_stride||(input->channels-1)>limit/input->channel_stride)
            throw std::invalid_argument("Waveform strides overflow address range");
        const int64_t last=(samples-1)*input->sample_stride;
        if((input->channels-1)*input->channel_stride>limit-last)throw std::invalid_argument("Waveform strides overflow address range");
        context->model.checkpoint();
        std::vector<float> pcm(size_t(samples)*2);
        for(int ch=0;ch<2;++ch)for(int64_t i=0;i<samples;++i) {
            const float value=input->samples[i*input->sample_stride+(input->channels==1?0:ch*input->channel_stride)];
            if(!std::isfinite(value))throw std::invalid_argument("Nonfinite input waveform");
            pcm[size_t(ch*samples+i)]=value;
        }
        auto audio=heartcodec::encoder::resample(context->model,pcm.data(),samples,input->sample_rate,24000,2);
        const int64_t n24=int64_t(audio.size()/2),chunks=(n24+714239)/714240;
        const bool dumping=directory&&*directory;
        if(dumping){std::filesystem::create_directories(directory);heartcodec::encoder::dump(std::filesystem::path(directory)/"resampled24.f32",audio.data(),audio.size());}
        std::vector<int32_t> codes;codes.reserve(size_t(frames)*8);
        for(int64_t chunk=0;chunk<chunks;++chunk) {
            std::vector<float> data(714480*2,0);
            const int64_t first=chunk*714240,count=std::min<int64_t>(714240,n24-first);
            for(int ch=0;ch<2;++ch)std::copy_n(audio.data()+ch*n24+first,size_t(count),data.data()+ch*714480);
            std::ostringstream name;name<<"chunk-"<<std::setw(6)<<std::setfill('0')<<chunk;
            const auto root=(std::filesystem::path(dumping?directory:"")/name.str()).string();
            const char * debug=dumping?root.c_str():nullptr;
            if(dumping){std::filesystem::create_directories(root);heartcodec::encoder::dump(std::filesystem::path(root)/"chunk24.f32",data.data(),data.size());}
            auto f=heartcodec::encoder::features(context->model,data.data(),debug);
            auto query=heartcodec::encoder_query(context->model,f[0].data(),f[1].data(),f[2].data(),f[3].data(),744,debug);
            auto ids=heartcodec::encoder_rvq(context->model,query.data.data(),query.frames,debug);
            const int64_t remaining=frames-int64_t(codes.size()/8),take=std::min(remaining,query.frames);
            codes.insert(codes.end(),ids.begin(),ids.begin()+size_t(take)*8);
            if(context->model.work)context->model.work->progress(HEARTCODEC_WORK_ENCODING_V1,int64_t(codes.size()/8),frames);
            if(int64_t(codes.size()/8)==frames)break;
        }
        if(int64_t(codes.size()/8)!=frames)throw std::runtime_error("Waveform encoder returned too few RVQ frames");
        auto owned=std::make_unique<int32_t[]>(codes.size());std::copy(codes.begin(),codes.end(),owned.get());
        out->data=owned.release();out->frames=frames;out->resampled_samples=n24;
    });
}
heartcodec_status_v1 HEARTCODEC_CALL heartcodec_encoder_get_memory_v1(heartcodec_encoder_context * context,
        heartcodec_memory_v1 * memory,heartcodec_error_v1 * error) {
    return guarded(error,[&]{
        if(!context||!memory||memory->size<sizeof(*memory))throw std::invalid_argument("Invalid encoder memory result");
        memory->weight_buffer_bytes=ggml_backend_buffer_get_size(context->model.buffer);memory->graph_buffer_bytes=context->model.graph_bytes;
    });
}
const heartcodec_encoder_api_v1 * HEARTCODEC_CALL heartcodec_get_encoder_api(uint32_t version) {
    static const heartcodec_encoder_api_v1 api={HEARTCODEC_ENCODER_API_V1_SIZE,1,heartcodec_get_api(1),
        heartcodec_config_init_v1,heartcodec_encoder_context_create_v1,heartcodec_encoder_context_destroy_v1,
        heartcodec_encoded_tokens_init_v1,heartcodec_encoded_tokens_free_v1,heartcodec_encode_v1,heartcodec_encoder_get_memory_v1};
    return version==1?&api:nullptr;
}
int32_t HEARTCODEC_CALL heartcodec_encode_controlled_v1(heartcodec_encoder_context * context,const heartcodec_waveform_v1 * input,
        const char * directory,const heartcodec_work_control_v1 * control,heartcodec_encoded_tokens_v1 * out,heartcodec_error_v1 * e) {
    if(!context||!control||control->size<sizeof(*control)||context->model.work)
        return heartcodec::error(e,HEARTCODEC_INVALID_ARGUMENT_V1,"Invalid controlled encoder call or reentrant context");
    heartcodec::Work work{control};
    struct Scope{heartcodec::Model & model;~Scope(){model.work=nullptr;}} scope{context->model};
    context->model.work=&work;
    const auto status=heartcodec_encode_v1(context,input,directory,out,e);
    return work.cancelled?heartcodec::error(e,static_cast<heartcodec_status_v1>(HEARTCODEC_WORK_CANCELLED_V1),"HeartCodec encoding cancelled"):status;
}
}
