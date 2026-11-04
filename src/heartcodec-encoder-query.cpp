#include "encoder-graph.h"
#include <cfloat>

namespace heartcodec {
Encoded encoder_query(Model & model,const float * acoustic,const float * semantic,const float * whisper,
                      const float * wavlm,int64_t frames,const char * directory) {
    if(!model.audio_encoder||!acoustic||!semantic||!whisper||!wavlm||frames<2||frames>744||frames%2)
        throw std::invalid_argument("Invalid encoder feature streams");
    for(const auto & pair:{std::make_pair(acoustic,2048),{semantic,2048},{whisper,1536},{wavlm,1536}})
        if(!std::all_of(pair.first,pair.first+frames*pair.second,[](float x){return std::isfinite(x);}))
            throw std::invalid_argument("Nonfinite encoder features");
    const bool dumping=directory&&*directory;encoder::Graph g(model,dumping);auto * c=g.ctx;
    auto * a=g.keep("stage.query.cond_fusion_layer_acoustic",g.linear("query.acoustic",g.input(acoustic,2048,frames)));
    auto * s=g.keep("stage.query.cond_fusion_layer_semantic",g.linear("query.semantic",g.input(semantic,2048,frames)));
    auto * p=g.keep("stage.query.cond_fusion_layer_phone",g.linear("query.phone",g.input(wavlm,1536,frames)));
    auto * fused=ggml_concat(c,ggml_concat(c,ggml_concat(c,a,s,0),g.input(whisper,1536,frames),0),p,0);
    auto * x=g.keep("stage.query.feature_proj",g.linear("query.feature_proj",fused));
    x=ggml_reshape_3d(c,x,512,2,frames/2);
    auto * cls=ggml_repeat_4d(c,model.tensor("query.cls_token"),512,1,frames/2,1);
    x=ggml_reshape_2d(c,ggml_concat(c,x,cls,1),512,frames/2*3);
    const int64_t length=x->ne[1];
    for(int layer=0;layer<6;++layer) {
        const auto b="query.blocks."+std::to_string(layer);
        auto * qkv=g.linear(b+".attn.to_qkv",x,false);
        auto * q=ggml_reshape_3d(c,g.slice(qkv,0,512),128,4,length);
        auto * k=ggml_reshape_3d(c,g.slice(qkv,512,512),128,4,length);
        q=g.norm(b+".attn.q_norm",q);k=g.norm(b+".attn.k_norm",k);
        q=g.rotary(q,b+".rope.inv_freq",64);
        k=g.rotary(k,b+".rope.inv_freq",64);
        auto * attended=g.attention(ggml_reshape_2d(c,q,512,length),ggml_reshape_2d(c,k,512,length),g.slice(qkv,1024,512),128,4);
        attended=g.linear(b+".attn.to_out",attended,false);
        x=ggml_add(c,x,ggml_mul(c,attended,model.tensor(b+".attn_scale.scale")));
        auto * ff=g.linear(b+".ff.ff.0.proj",x);
        ff=ggml_mul(c,g.slice(ff,0,2048),ggml_sigmoid(c,g.slice(ff,2048,2048)));
        ff=g.linear(b+".ff.ff.2",ff);
        x=ggml_add(c,x,ggml_mul(c,ff,model.tensor(b+".ff_scale.scale")));
        if(layer==0||layer==1||layer==5)x=g.keep("stage.query.encoder_transformers."+std::to_string(layer),x);
    }
    auto * output=ggml_cont(c,ggml_view_2d(c,x,512,frames/2,x->nb[1]*3,x->nb[1]*2));
    output=g.keep("query",output,true);g.execute(directory);
    return {encoder::values(output),frames/2};
}

std::vector<int32_t> encoder_rvq(Model & model,const float * query,int64_t frames,const char * directory) {
    if(!model.audio_encoder||!query||frames<1||frames>8192||
       !std::all_of(query,query+frames*512,[](float x){return std::isfinite(x);}))throw std::invalid_argument("Invalid RVQ encoder input");
    const bool dumping=directory&&*directory;
    std::vector<float> residual;
    {
        encoder::Graph g(model,dumping);
        auto * output=g.keep("rvq.projected",g.linear("rvq.project_in",g.input(query,512,frames)),true);
        g.execute(directory);residual=encoder::values(output);
    }
    std::vector<int32_t> codes(size_t(frames)*8);
    for(int level=0;level<8;++level) {
        const auto name="rvq.layers."+std::to_string(level)+"._codebook.embed";
        auto * embed=model.tensor(name);encoder::Graph g(model);auto * c=g.ctx;
        auto * x=g.input(residual.data(),32,frames);auto * dot=g.mm(embed,x);
        auto * x2=ggml_sum_rows(c,ggml_sqr(c,x));
        auto * e2=ggml_cont(c,ggml_transpose(c,ggml_sum_rows(c,ggml_sqr(c,embed))));
        // Exact library expression: -(x2 + e2 - 2*dot).clamp(1e-8).sqrt().
        auto * distance=ggml_add(c,ggml_add(c,ggml_repeat(c,x2,dot),e2),ggml_scale(c,dot,-2));
        distance=ggml_neg(c,ggml_sqrt(c,ggml_clamp(c,distance,1e-8f,FLT_MAX)));
        distance=g.keep("distance",distance,true);g.execute();auto scores=encoder::values(distance),embeddings=encoder::values(embed);
        std::vector<float> margins(size_t(frames),0);
        // Host selection preserves Torch's lowest-index tie rule. Existing CUDA
        // GGML argmax does not promise that rule; no shared kernel patch needed.
        for(int64_t f=0;f<frames;++f) {
            int best=0;float highest=scores[size_t(f)*8192],second=-FLT_MAX;
            for(int i=1;i<8192;++i) {
                const float value=scores[size_t(f)*8192+i];
                if(value>highest){second=highest;highest=value;best=i;}else second=std::max(second,value);
            }
            codes[size_t(f)*8+level]=best;margins[size_t(f)]=highest-second;
            for(int d=0;d<32;++d)residual[size_t(f)*32+d]-=embeddings[size_t(best)*32+d];
        }
        if(dumping)encoder::dump(std::filesystem::path(directory)/("rvq.margin."+std::to_string(level)+".f32"),margins.data(),margins.size());
    }
    return codes;
}
}
