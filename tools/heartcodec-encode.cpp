#include "heartcodec_encoder.h"
#ifdef HEARTCODEC_ENCODER_DIAGNOSTICS
#include "heartcodec-internal.h"
#include "encoder-features.h"
#endif
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> static std::vector<T> read(const std::filesystem::path & p,size_t count) {
    if(std::filesystem::file_size(p)!=count*sizeof(T))throw std::runtime_error("Wrong encoder input length: "+p.string());
    std::vector<T> result(count);std::ifstream file(p,std::ios::binary);
    if(!file.read(reinterpret_cast<char *>(result.data()),std::streamsize(count*sizeof(T))))throw std::runtime_error("Cannot read encoder input");return result;
}
template<class T> static void write(const std::filesystem::path & p,const std::vector<T> & data) {
    std::ofstream file(p,std::ios::binary);
    if(!file.write(reinterpret_cast<const char *>(data.data()),std::streamsize(data.size()*sizeof(T))))throw std::runtime_error("Cannot write encoder output");
}
struct Wave {std::vector<float> samples;int rate=0,channels=0;};
static Wave read_wave(const std::filesystem::path & path) {
    const auto bytes=std::filesystem::file_size(path);
    if(bytes<44||bytes>UINT32_MAX)throw std::runtime_error("Invalid RIFF WAV length");
    auto raw=read<uint8_t>(path,size_t(bytes));
    auto u16=[&](size_t i){return uint16_t(raw[i]|uint16_t(raw[i+1])<<8);};
    auto u32=[&](size_t i){return uint32_t(raw[i])|uint32_t(raw[i+1])<<8|uint32_t(raw[i+2])<<16|uint32_t(raw[i+3])<<24;};
    if(std::memcmp(raw.data(),"RIFF",4)||std::memcmp(raw.data()+8,"WAVE",4))throw std::runtime_error("Expected little-endian RIFF WAV");
    size_t fmt=0,format_size=0,data=0,data_size=0;
    for(size_t i=12;i+8<=raw.size();) {
        const size_t n=u32(i+4),payload=i+8;if(n>raw.size()-payload)throw std::runtime_error("Truncated WAV chunk");
        if(!std::memcmp(raw.data()+i,"fmt ",4)){fmt=payload;format_size=n;}
        if(!std::memcmp(raw.data()+i,"data",4)){data=payload;data_size=n;}
        i=payload+n+(n%2);
    }
    if(!fmt||format_size<16||!data||!data_size)throw std::runtime_error("Missing WAV format/audio");
    int format=u16(fmt),channels=u16(fmt+2),bits=u16(fmt+14);const int rate=int(u32(fmt+4));
    if(format==65534&&format_size>=40)format=u16(fmt+24);
    if((channels!=1&&channels!=2)||!((format==1&&(bits==16||bits==24||bits==32))||(format==3&&bits==32))||
       u16(fmt+12)!=channels*bits/8||data_size%(channels*bits/8))throw std::runtime_error("Unsupported PCM WAV format");
    Wave result;result.rate=rate;result.channels=channels;result.samples.resize(data_size/(bits/8));
    for(size_t i=0;i<result.samples.size();++i) {
        const size_t p=data+i*size_t(bits/8);float value=0;
        if(format==3)std::memcpy(&value,raw.data()+p,4);
        else if(bits==16)value=float(int16_t(u16(p)))/32768.0f;
        else if(bits==24){int32_t v=int32_t(raw[p]|uint32_t(raw[p+1])<<8|uint32_t(raw[p+2])<<16);if(v&0x800000)v|=int32_t(0xff000000u);value=float(v)/8388608.0f;}
        else value=float(int32_t(u32(p)))/2147483648.0f;
        result.samples[i]=value;
    }
    return result;
}
int main(int argc,char ** argv) {
    try {
        if(argc<2)throw std::runtime_error("Usage: heartcodec-encode encode --model MODEL --input AUDIO.wav --output FRESH_DIR [--backend cpu|cuda --threads 8 --dump 0|1]; diagnostic query|rvq|preprocess|features|mu|whisper|wavlm modes additionally use --frames N and a fixture input directory");
        const std::string mode=argv[1];
        if(mode!="encode"&&mode!="query"&&mode!="rvq"&&mode!="preprocess"&&mode!="features"&&mode!="whisper"&&mode!="wavlm"&&mode!="mu")throw std::runtime_error("Unknown encoder mode");
        std::map<std::string,std::string> args;
        for(int i=2;i<argc;i+=2)if(i+1>=argc||!args.emplace(argv[i],argv[i+1]).second)throw std::runtime_error("Missing/duplicate encoder option");
        for(const auto & p:args)if(p.first!="--model"&&p.first!="--input"&&p.first!="--output"&&p.first!="--frames"&&p.first!="--backend"&&p.first!="--threads"&&p.first!="--sample-rate"&&p.first!="--dump")throw std::runtime_error("Unknown encoder option");
        const int64_t frames=mode=="encode"?0:std::stoll(args.at("--frames"));
        if(mode!="encode"&&(frames<1||frames>8192||(mode=="query"&&(frames%2||frames>744))))throw std::runtime_error("Invalid encoder frame count");
        const std::filesystem::path input(args.at("--input")),output(args.at("--output"));
        if(std::filesystem::exists(output))throw std::runtime_error("Use a fresh encoder output directory");
        const auto start=std::chrono::steady_clock::now();
        const auto backend=args.count("--backend")?args.at("--backend"):"cpu";
        const int threads=args.count("--threads")?std::stoi(args.at("--threads")):8;
        if(threads<1||threads>1024)throw std::runtime_error("Invalid CPU thread count");
        if(mode=="encode") {
            auto pcm=read_wave(input);const auto * api=heartcodec_get_encoder_api(1);
            heartcodec_config_v1 config{};config.size=sizeof(config);api->config_init(&config);
            config.model_path=args.at("--model").c_str();config.backend=backend.c_str();config.cpu_threads=threads;
            heartcodec_error_v1 error{};error.size=sizeof(error);heartcodec_encoder_context * context=nullptr;
            if(api->context_create(&config,&context,&error))throw std::runtime_error(error.message);
            std::unique_ptr<heartcodec_encoder_context,decltype(api->context_destroy)> owner(context,api->context_destroy);
            heartcodec_waveform_v1 wave{sizeof(wave),pcm.samples.data(),int64_t(pcm.samples.size()/pcm.channels),pcm.rate,pcm.channels,1,pcm.channels};
            heartcodec_encoded_tokens_v1 result{};result.size=sizeof(result);api->tokens_init(&result);
            struct TokenGuard {const heartcodec_encoder_api_v1 * api;heartcodec_encoded_tokens_v1 * value;~TokenGuard(){api->tokens_free(value);}} tokens_owner{api,&result};
            const auto flag=args.count("--dump")?args.at("--dump"):"0";if(flag!="0"&&flag!="1")throw std::runtime_error("Dump must be 0 or 1");
            const auto debug=(output/"stages").string();
            if(api->encode(context,&wave,flag=="1"?debug.c_str():nullptr,&result,&error))throw std::runtime_error(error.message);
            const int64_t count=result.frames,n24=result.resampled_samples;
            heartcodec_memory_v1 memory{};memory.size=sizeof(memory);if(api->get_memory(context,&memory,&error))throw std::runtime_error(error.message);
            // Exercise the C ABI result lifetime after all model storage is gone.
            owner.reset();std::vector<int32_t> codes(result.data,result.data+count*8);
            std::filesystem::create_directories(output);write(output/"codes.i32",codes);
            const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            std::ofstream report(output/"run.json");
            report<<"{\"mode\":\"encode\",\"frames\":"<<count<<",\"source_samples\":"<<wave.num_samples<<",\"source_rate\":"<<wave.sample_rate<<",\"source_channels\":"<<wave.channels
                  <<",\"resampled_samples\":"<<n24<<",\"seconds\":"<<seconds<<",\"weight_bytes\":"<<memory.weight_buffer_bytes<<",\"graph_bytes\":"<<memory.graph_buffer_bytes<<"}\n";
            if(!report)throw std::runtime_error("Cannot write encode report");std::cout<<"Encoded "<<count<<" RVQ frames in "<<seconds<<"s\n";return 0;
        }
#ifdef HEARTCODEC_ENCODER_DIAGNOSTICS
        heartcodec::Model model;
        model.load(args.at("--model").c_str(),backend.c_str(),threads);
        if(!model.audio_encoder)throw std::runtime_error("Expected full waveform encoder GGUF");
        std::filesystem::create_directories(output);
        std::vector<float> query;int64_t query_frames=frames;
        if(mode=="preprocess") {
            const auto count=std::filesystem::file_size(input/"waveform.f32")/4;
            if(count%2)throw std::runtime_error("Expected stereo planar waveform");
            auto audio=read<float>(input/"waveform.f32",size_t(count));
            auto resampled=heartcodec::encoder::resample(model,audio.data(),int64_t(count/2),std::stoi(args.at("--sample-rate")),24000,2);
            if(resampled.size()!=714240*2)throw std::runtime_error("Preprocessing fixture must be 29.76 seconds");
            write(output/"resampled24.f32",resampled);std::vector<float> chunk(714480*2,0);
            for(int ch=0;ch<2;++ch)std::copy_n(resampled.data()+ch*714240,714240,chunk.data()+ch*714480);
            write(output/"chunk24.f32",chunk);
            auto audio16=heartcodec::encoder::resample(model,chunk.data(),714480,24000,16000,2);
            write(output/"stage.feature.resample_24k_to_16k.f32",audio16);
            heartcodec::encoder::mel_features(model,chunk.data(),audio16.data(),output.string().c_str());
            query_frames=0;
        } else if(mode=="whisper"||mode=="mu"||mode=="wavlm") {
            std::array<std::vector<float>,2> features;
            const int width=mode=="mu"?128:80,time=mode=="mu"?2976:3000,dim=mode=="mu"?1024:768;
            std::vector<float> data;
            if(mode=="wavlm")data=read<float>(input/"stage.feature.resample_24k_to_16k.f32",476320*2);
            else data=read<float>(input/("input."+mode+"_mel.f32"),size_t(width)*time*2);
            std::vector<float> acoustic(744*2048),semantic(744*2048),packed(744*1536);
            for(int ch=0;ch<2;++ch) {
                const auto debug=(output/("channel-"+std::to_string(ch))).string();
                std::vector<float> mel(size_t(width)*time);
                if(mode!="wavlm")for(int t=0;t<time;++t)for(int d=0;d<width;++d)mel[size_t(t)*width+d]=data[(size_t(ch)*width+d)*time+t];
                if(mode=="mu") {
                    auto m=heartcodec::encoder::mu(model,mel.data(),debug.c_str());
                    for(int t=0;t<744;++t)for(int d=0;d<dim;++d) {
                        acoustic[size_t(t)*2048+ch*dim+d]=m[0][size_t(t)*dim+d];
                        semantic[size_t(t)*2048+ch*dim+d]=m[1][size_t(t)*dim+d];
                    }
                } else {
                    auto m=mode=="whisper"?heartcodec::encoder::whisper(model,mel.data(),debug.c_str()):heartcodec::encoder::wavlm(model,data.data()+ch*476320,476320,debug.c_str());
                    for(int t=0;t<744;++t)for(int d=0;d<dim;++d)
                        packed[size_t(t)*1536+ch*dim+d]=(m[size_t(t*2)*dim+d]+m[size_t(t*2+1)*dim+d])*.5f;
                }
            }
            if(mode=="mu"){write(output/"input.acoustic.f32",acoustic);write(output/"input.semantic.f32",semantic);}
            else write(output/("input."+mode+".f32"),packed);
            query_frames=0;
        } else if(mode=="features") {
            auto chunk=read<float>(input/"chunk24.f32",714480*2);
            auto f=heartcodec::encoder::features(model,chunk.data(),output.string().c_str());
            auto result=heartcodec::encoder_query(model,f[0].data(),f[1].data(),f[2].data(),f[3].data(),744,output.string().c_str());
            query=std::move(result.data);query_frames=result.frames;
        } else if(mode=="query") {
            auto a=read<float>(input/"input.acoustic.f32",size_t(frames)*2048),s=read<float>(input/"input.semantic.f32",size_t(frames)*2048);
            auto w=read<float>(input/"input.whisper.f32",size_t(frames)*1536),l=read<float>(input/"input.wavlm.f32",size_t(frames)*1536);
            auto result=heartcodec::encoder_query(model,a.data(),s.data(),w.data(),l.data(),frames,output.string().c_str());
            query=std::move(result.data);query_frames=result.frames;
        } else query=read<float>(input/"query.f32",size_t(frames)*512);
        if(query_frames){const auto codes=heartcodec::encoder_rvq(model,query.data(),query_frames,output.string().c_str());write(output/"codes.i32",codes);}
        const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::ofstream report(output/"run.json");
        report<<"{\"mode\":\""<<mode<<"\",\"frames\":"<<query_frames<<",\"seconds\":"<<seconds
              <<",\"weight_bytes\":"<<ggml_backend_buffer_get_size(model.buffer)<<",\"graph_bytes\":"<<model.graph_bytes<<"}\n";
        if(!report)throw std::runtime_error("Cannot write encoder report");
        std::cout<<"Encoded "<<query_frames<<" RVQ frames in "<<seconds<<"s\n";return 0;
#else
        throw std::runtime_error("Diagnostic modes require a static HeartCodec tool build");
#endif
    }catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
}
