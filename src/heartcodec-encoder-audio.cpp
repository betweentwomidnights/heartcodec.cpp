#include "encoder-features.h"
#include <complex>

namespace heartcodec { namespace encoder {
std::vector<float> resample(Model & model,const float * input,int64_t samples,int rate,int target,int channels) {
    if(rate==target)return std::vector<float>(input,input+samples*channels);
    if((target!=24000&&!(rate==24000&&target==16000))||samples<1||channels<1||channels>2)throw std::invalid_argument("Unsupported encoder sample rate");
    const auto name=target==16000?std::string("rs16.kernel"):"rsin."+std::to_string(rate)+".kernel";
    auto * kernel=model.tensor(name);
    const int gcd=std::gcd(rate,target),orig=rate/gcd,next=target/gcd;
    const int64_t width=(kernel->ne[0]-orig)/2,output_samples=(samples*next+orig-1)/orig;
    std::vector<float> output(size_t(output_samples)*channels);
    for(int channel=0;channel<channels;++channel) {
        Graph g(model);auto * x=g.input(input+samples*channel,1,samples);
        auto * y=g.keep("resampled",g.conv_kernel(kernel,x,orig,int(width),int(width)+orig),true);
        g.execute();auto result=values(y);
        if(int64_t(result.size())<output_samples)throw std::runtime_error("Encoder resampler returned too few samples");
        std::copy_n(result.data(),size_t(output_samples),output.data()+output_samples*channel);
    }
    return output;
}
namespace {
// Reference preprocessing uses CPU STFT. Double-complex butterflies followed
// by F32 magnitudes avoid an extra native dependency; windows/filterbanks are
// the exact frozen Torch/HF CPU buffers in the strict F32 GGUF.
struct FFT {
    int size;
    std::map<int,std::vector<std::complex<double>>> twiddles;
    std::vector<std::complex<double>> out,work;
    explicit FFT(int n):size(n),out(size_t(n)),work(size_t(n)) {
        for(int m=n;m>1;m/=m%2?5:2) {
            if(m%2&&m%5)throw std::runtime_error("Unsupported encoder FFT size");
            auto & row=twiddles[m];row.resize(size_t(m));
            for(int k=0;k<m;++k){const double a=-2*3.14159265358979323846*k/m;row[size_t(k)]={std::cos(a),std::sin(a)};}
        }
    }
    void run(const float * input,int stride,int n,std::complex<double> * dst) {
        if(n==1){dst[0]=input[0];return;}
        const int p=n%2?5:2,q=n/p;
        for(int r=0;r<p;++r)run(input+r*stride,stride*p,q,dst+r*q);
        const auto & tw=twiddles.at(n);
        for(int k=0;k<n;++k) {
            std::complex<double> sum=0;
            for(int r=0;r<p;++r)sum+=dst[r*q+k%q]*tw[size_t((r*k)%n)];
            work[size_t(k)]=sum;
        }
        std::copy_n(work.data(),size_t(n),dst);
    }
};
std::vector<float> spectrogram(const float * audio,int64_t samples,int fft,int hop,const std::vector<float> & window,bool drop_last) {
    const int64_t frames=samples/hop+1-int(drop_last);const int bins=fft/2+1;
    std::vector<float> result(size_t(frames)*bins),input(size_t(fft),0);FFT transform(fft);
    for(int64_t frame=0;frame<frames;++frame) {
        for(int j=0;j<fft;++j) {
            int64_t index=frame*hop+j-fft/2;
            if(index<0)index=-index;if(index>=samples)index=2*samples-2-index;
            input[size_t(j)]=audio[index]*window[size_t(j)];
        }
        transform.run(input.data(),1,fft,transform.out.data());
        for(int b=0;b<bins;++b) {
            const auto z=transform.out[size_t(b)];
            const float magnitude=std::hypot(float(z.real()),float(z.imag()));
            result[size_t(frame)*bins+b]=magnitude*magnitude;
        }
    }
    return result;
}
std::vector<float> filter(Model & model,const std::vector<float> & power,const std::string & name,int bins) {
    Graph g(model);auto * w=ggml_cont(g.ctx,ggml_transpose(g.ctx,model.tensor(name)));
    auto * x=g.input(power.data(),bins,int64_t(power.size()/bins));
    auto * y=g.keep("mel",g.mm(w,x),true);g.execute();return values(y);
}
void dump_channel_major(const std::filesystem::path & file,const std::vector<float> & data,int channels,int frames,int width) {
    std::vector<float> out(data.size());
    for(int b=0;b<channels;++b)for(int t=0;t<frames;++t)for(int d=0;d<width;++d)
        out[(size_t(b)*width+d)*frames+t]=data[(size_t(b)*frames+t)*width+d];
    dump(file,out.data(),out.size());
}
}
Mel mel_features(Model & model,const float * chunk,const float * audio16,const char * directory) {
    constexpr int mu_samples=714240,mu_frames=2977,wh_frames=3000;
    Mel result;result.mu.resize(2*2976*128);result.whisper.resize(2*wh_frames*80);
    const auto mu_window=values(model.tensor("mu.mel.mel_stft.spectrogram.window")),wh_window=values(model.tensor("wh.window"));
    std::vector<float> raw_mu(2*mu_frames*128);
    for(int channel=0;channel<2;++channel) {
        auto power=spectrogram(chunk+channel*714480,mu_samples,2048,240,mu_window,false);
        auto mel=filter(model,power,"mu.mel.mel_stft.mel_scale.fb",1025);
        for(size_t i=0;i<mel.size();++i)raw_mu[size_t(channel)*mel.size()+i]=10.0f*std::log10(std::max(mel[i],1e-10f));
        std::vector<float> padded(480000,0);std::copy_n(audio16+channel*476320,476320,padded.data());
        power=spectrogram(padded.data(),480000,400,160,wh_window,true);
        mel=filter(model,power,"wh.mel_filters",201);
        for(auto & x:mel)x=std::log10(std::max(x,1e-10f));
        const float floor=*std::max_element(mel.begin(),mel.end())-8.0f;
        for(size_t i=0;i<mel.size();++i)result.whisper[size_t(channel)*mel.size()+i]=(std::max(mel[i],floor)+4.0f)/4.0f;
    }
    // The reference constructs AmplitudeToDB() with top_db=None: no 80-dB floor.
    for(int channel=0;channel<2;++channel)for(int t=0;t<2976;++t)for(int d=0;d<128;++d)
        result.mu[(size_t(channel)*2976+t)*128+d]=(raw_mu[(size_t(channel)*mu_frames+t)*128+d]-float(6.768444971712967))/float(18.417922652295623);
    if(directory&&*directory) {
        std::filesystem::create_directories(directory);const std::filesystem::path root(directory);
        dump_channel_major(root/"stage.feature.muencoder.preprocessor_melspec_2048.f32",raw_mu,2,mu_frames,128);
        dump_channel_major(root/"input.mu_mel.f32",result.mu,2,2976,128);
        dump_channel_major(root/"input.whisper_mel.f32",result.whisper,2,3000,80);
    }
    return result;
}
}}
