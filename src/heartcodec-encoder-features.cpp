#include "encoder-features.h"

namespace heartcodec { namespace encoder {
namespace {
ggml_tensor * channel_major(Graph & g,ggml_tensor * x) {return ggml_cont(g.ctx,ggml_transpose(g.ctx,x));}
ggml_tensor * trim_time(Graph & g,ggml_tensor * x,int64_t frames) {
    return ggml_cont(g.ctx,ggml_view_2d(g.ctx,x,x->ne[0],frames,x->nb[1],0));
}
ggml_tensor * pointwise(Graph & g,const std::string & p,ggml_tensor * x) {
    auto * w=g.model.tensor(p+".weight");return g.mm(ggml_reshape_2d(g.ctx,w,w->ne[1],w->ne[2]),x);
}
}
std::vector<float> whisper(Model & model,const float * mel,const char * directory,const float * hidden,int first_layer,int end_layer) {
    if(first_layer<0||end_layer>12||first_layer>=end_layer)throw std::invalid_argument("Invalid Whisper layer interval");
    Graph g(model,directory&&*directory);auto * c=g.ctx;
    ggml_tensor * x=nullptr;
    if(hidden)x=g.input(hidden,768,1500);
    else {
        x=g.conv1("wh.conv1",g.input(mel,80,3000),1,1);
        g.keep("stage.feature.whisper_encoder.conv1",channel_major(g,x));x=ggml_gelu_erf(c,x);
        g.keep("trace.whisper.gelu1",x);
        x=g.conv1("wh.conv2",x,2,1);g.keep("stage.feature.whisper_encoder.conv2",channel_major(g,x));x=ggml_gelu_erf(c,x);
        x=ggml_add(c,x,model.tensor("wh.embed_positions.weight"));
        g.keep("trace.whisper.embedded",x);
    }
    for(int i=first_layer;i<end_layer;++i) {
        const auto p="wh.layers."+std::to_string(i);
        const auto trace="trace.whisper.layers."+std::to_string(i);
        auto * normalized=g.norm(p+".attn_layer_norm",x);
        if(i==0||i==5||i==6||i==11)g.keep(trace+".self_attn_layer_norm",normalized);
        auto * q=g.linear(p+".attn.q_proj",normalized),* k=g.linear(p+".attn.k_proj",normalized,false),* v=g.linear(p+".attn.v_proj",normalized);
        if(i==0||i==5||i==6||i==11){g.keep(trace+".self_attn.q_proj",q);g.keep(trace+".self_attn.k_proj",k);g.keep(trace+".self_attn.v_proj",v);}
        auto * attended=g.attention(q,k,v,64,12,nullptr,true);
        auto * projected=g.linear(p+".attn.out_proj",attended);
        if(i==0||i==5||i==6||i==11){g.keep(trace+".attn_head_output",attended);g.keep(trace+".self_attn.out_proj",projected);}
        x=ggml_add(c,x,projected);
        normalized=g.norm(p+".final_layer_norm",x);
        auto * fc1=g.linear(p+".fc1",normalized),* activated=ggml_gelu_erf(c,fc1),* fc2=g.linear(p+".fc2",activated);
        if(i==0||i==5||i==6||i==11){g.keep(trace+".final_layer_norm",normalized);g.keep(trace+".fc1",fc1);g.keep(trace+".activated",activated);g.keep(trace+".fc2",fc2);}
        x=ggml_add(c,x,fc2);
        x=g.keep("stage.feature.whisper_encoder.layers."+std::to_string(i),x);
    }
    x=hidden?g.keep("whisper.layer_output",x,true):g.keep("stage.feature.whisper_encoder",g.norm("wh.norm",x),true);
    g.execute(directory);return values(x);
}

std::vector<float> wavlm(Model & model,const float * waveform,int64_t samples,const char * directory) {
    Graph g(model,directory&&*directory);auto * c=g.ctx;auto * x=g.input(waveform,1,samples);
    const int strides[]={5,2,2,2,2,2,2};
    for(int i=0;i<7;++i) {
        x=g.conv1("wl.conv."+std::to_string(i)+".conv",x,strides[i],0,false);
        if(!i) {
            auto * time=ggml_norm(c,channel_major(g,x),1e-5f);
            time=ggml_add(c,ggml_mul(c,time,ggml_reshape_2d(c,model.tensor("wl.conv.0.norm.weight"),1,512)),
                          ggml_reshape_2d(c,model.tensor("wl.conv.0.norm.bias"),1,512));
            x=channel_major(g,time);
        }
        x=ggml_gelu_erf(c,x);
    }
    if(x->ne[1]!=1488)throw std::runtime_error("Unexpected WavLM feature alignment");
    g.keep("stage.feature.wavlm_encoder.feature_extractor",channel_major(g,x));
    x=g.keep("stage.feature.wavlm_encoder.feature_projection",g.linear("wl.proj.projection",g.norm("wl.proj.norm",x)));
    const int64_t frames=x->ne[1];
    auto * pos=ggml_gelu_erf(c,trim_time(g,g.conv1("wl.enc.pos.conv",x,1,64,true,16),frames));
    pos=g.keep("stage.feature.wavlm_encoder.encoder.pos_conv_embed",pos);
    x=g.norm("wl.enc.norm",ggml_add(c,x,pos));
    std::vector<int32_t> buckets(size_t(frames*frames));
    std::vector<int32_t> lookup(size_t(frames*2-1));
    for(int delta=1-int(frames);delta<int(frames);++delta) {
        const int n=std::abs(delta);int bucket=n;
        if(n>=80) {
            const float scale=std::log(float(n)/80.0f)/float(std::log(10.0));
            bucket=std::min(159,int(80.0f+scale*80.0f));
        }
        lookup[size_t(delta+frames-1)]=bucket+(delta>0?160:0);
    }
    for(int64_t q=0;q<frames;++q)for(int64_t k=0;k<frames;++k)buckets[size_t(q*frames+k)]=lookup[size_t(k-q+frames-1)];
    auto * bias=ggml_get_rows(c,model.tensor("wl.enc.layers.0.attn.rel_attn_embed.weight"),g.ids(buckets));
    bias=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,bias,12,frames,frames),2,0,1,3));
    float one=1,two=2;auto * constant_one=g.input(&one,1),* constant_two=g.input(&two,1);
    ggml_tensor * sum=nullptr;
    // The reference selects hidden_states[6:10], after layers 5,6,7,8.
    for(int i=0;i<9;++i) {
        const auto p="wl.enc.layers."+std::to_string(i);
        auto * grouped=ggml_reshape_2d(c,x,64,12*frames);
        auto * gate=g.linear(p+".attn.gru_rel_pos_linear",grouped);
        gate=ggml_sum_rows(c,ggml_reshape_4d(c,gate,4,2,12,frames));
        gate=ggml_sigmoid(c,ggml_reshape_3d(c,gate,2,12,frames));
        auto * a=ggml_cont(c,ggml_view_3d(c,gate,1,12,frames,gate->nb[1],gate->nb[2],0));
        auto * b=ggml_cont(c,ggml_view_3d(c,gate,1,12,frames,gate->nb[1],gate->nb[2],4));
        a=ggml_cont(c,ggml_permute(c,a,0,2,1,3));b=ggml_cont(c,ggml_permute(c,b,0,2,1,3));
        gate=ggml_add(c,ggml_mul(c,a,ggml_sub(c,ggml_mul(c,b,model.tensor(p+".attn.gru_rel_pos_const")),constant_one)),constant_two);
        auto * gated_bias=ggml_mul(c,bias,gate);
        auto * attended=g.attention(g.linear(p+".attn.q_proj",x),g.linear(p+".attn.k_proj",x),g.linear(p+".attn.v_proj",x),64,12,gated_bias);
        x=g.norm(p+".norm",ggml_add(c,x,g.linear(p+".attn.out_proj",attended)));
        x=g.norm(p+".final_layer_norm",ggml_add(c,x,g.linear(p+".ffn.down",ggml_gelu_erf(c,g.linear(p+".ffn.up",x)))));
        if(i==0||i==5||i==8)x=g.keep("stage.feature.wavlm_encoder.encoder.layers."+std::to_string(i),x);
        if(i>=5)sum=sum?ggml_add(c,sum,x):x;
    }
    auto * output=g.keep("wavlm",ggml_scale(c,sum,.25f),true);g.execute(directory);return values(output);
}

std::array<std::vector<float>,2> mu(Model & model,const float * mel,const char * directory) {
    Graph g(model,directory&&*directory);auto * c=g.ctx;
    auto * x=ggml_cont(c,ggml_transpose(c,g.input(mel,128,2976)));
    x=ggml_reshape_4d(c,x,2976,128,1,1);
    for(int i=0;i<2;++i) {
        const auto p="mu.conv.conv."+std::to_string(i);auto * residual=x;
        x=ggml_relu(c,g.batch_norm(p+".bn1",g.conv2(p+".conv1",x,2,1),true));
        x=g.batch_norm(p+".bn2",g.conv2(p+".conv2",x,1,1),true);
        residual=g.batch_norm(p+".bn3",g.conv2(p+".conv3",residual,2,1),true);
        x=ggml_relu(c,ggml_add(c,x,residual));
    }
    x=ggml_cont(c,ggml_permute(c,x,2,0,1,3));
    x=g.keep("stage.feature.muencoder.conv",g.linear("mu.conv.linear",ggml_reshape_2d(c,x,16384,744)));
    ggml_tensor * acoustic=nullptr;
    // MuEncoder selects hidden_states[2] and [11], after layers 1 and 10.
    // The final encoder normalization belongs to the unused last state.
    for(int i=0;i<11;++i) {
        const auto p="mu.enc.layers."+std::to_string(i);
        auto * ff=g.linear(p+".ffn1.down",ggml_silu(c,g.linear(p+".ffn1.up",g.norm(p+".ffn1_layer_norm",x))));
        x=ggml_add(c,x,ggml_scale(c,ff,.5f));
        auto * normalized=g.norm(p+".attn_layer_norm",x);
        // HF conformer rotates the shared input before Q/K projections.
        auto * rotated=ggml_reshape_3d(c,normalized,64,16,744);
        rotated=g.rotary(rotated,"mu.enc.embed_positions.inv_freq",64);
        rotated=ggml_reshape_2d(c,rotated,1024,744);
        auto * attended=g.attention(g.linear(p+".attn.linear_q",rotated),g.linear(p+".attn.linear_k",rotated),g.linear(p+".attn.linear_v",normalized),64,16);
        x=ggml_add(c,x,g.linear(p+".attn.linear_out",attended));
        normalized=g.norm(p+".conv.norm",x);
        auto * conv=pointwise(g,p+".conv.pointwise_conv1",normalized);
        conv=ggml_mul(c,g.slice(conv,0,1024),ggml_sigmoid(c,g.slice(conv,1024,1024)));
        conv=g.depthwise(p+".conv.depthwise_conv",conv,15);
        conv=ggml_silu(c,g.batch_norm(p+".conv.batch_norm",conv));
        x=ggml_add(c,x,pointwise(g,p+".conv.pointwise_conv2",conv));
        ff=g.linear(p+".ffn2.down",ggml_silu(c,g.linear(p+".ffn2.up",g.norm(p+".ffn2_layer_norm",x))));
        x=g.norm(p+".final_layer_norm",ggml_add(c,x,ggml_scale(c,ff,.5f)));
        if(i==0||i==1||i==10)x=g.keep("stage.feature.muencoder.conformer.layers."+std::to_string(i),x);
        if(i==1)acoustic=g.keep("acoustic",x,true);
    }
    auto * semantic=g.keep("semantic",x,true);g.execute(directory);return {values(acoustic),values(semantic)};
}

std::array<std::vector<float>,4> features(Model & model,const float * chunk24,const char * directory) {
    const bool dumping=directory&&*directory;
    auto audio16=resample(model,chunk24,714480,24000,16000,2);
    if(dumping){std::filesystem::create_directories(directory);dump(std::filesystem::path(directory)/"stage.feature.resample_24k_to_16k.f32",audio16.data(),audio16.size());}
    auto mel=mel_features(model,chunk24,audio16.data(),directory);
    std::array<std::vector<float>,4> result;
    for(int i=0;i<4;++i)result[size_t(i)].resize(size_t(744)*(i<2?2048:1536));
    for(int channel=0;channel<2;++channel) {
        const auto sub=std::filesystem::path(dumping?directory:"")/("channel-"+std::to_string(channel));
        const auto path=sub.string();const char * debug=dumping?path.c_str():nullptr;
        auto m=mu(model,mel.mu.data()+channel*2976*128,debug);
        auto w=whisper(model,mel.whisper.data()+channel*3000*80,debug);
        auto l=wavlm(model,audio16.data()+channel*476320,476320,debug);
        for(int frame=0;frame<744;++frame) {
            for(int d=0;d<1024;++d)for(int stream=0;stream<2;++stream)
                result[size_t(stream)][size_t(frame)*2048+channel*1024+d]=m[size_t(stream)][size_t(frame)*1024+d];
            for(int d=0;d<768;++d) {
                result[2][size_t(frame)*1536+channel*768+d]=(w[size_t(frame*2)*768+d]+w[size_t(frame*2+1)*768+d])*.5f;
                result[3][size_t(frame)*1536+channel*768+d]=(l[size_t(frame*2)*768+d]+l[size_t(frame*2+1)*768+d])*.5f;
            }
        }
    }
    if(dumping)for(int i=0;i<4;++i)dump(std::filesystem::path(directory)/("input."+std::string(std::array<const char *,4>{"acoustic","semantic","whisper","wavlm"}[size_t(i)])+".f32"),result[size_t(i)].data(),result[size_t(i)].size());
    return result;
}
}}
