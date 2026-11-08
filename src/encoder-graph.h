#pragma once
#include "heartcodec-internal.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <numeric>

namespace heartcodec { namespace encoder {
inline void dump(const std::filesystem::path & path, const float * data, size_t count) {
    std::ofstream file(path,std::ios::binary);
    if (!file.write(reinterpret_cast<const char *>(data),std::streamsize(count*sizeof(float))))
        throw std::runtime_error("Cannot write encoder stage: " + path.string());
}
inline std::vector<float> values(ggml_tensor * t) {
    std::vector<float> out(size_t(ggml_nelements(t)));
    ggml_backend_tensor_get(t,out.data(),0,out.size()*sizeof(float)); return out;
}
struct Graph {
    Model & model;
    ggml_context * ctx=nullptr;
    ggml_gallocr_t allocator=nullptr;
    bool dumping;
    std::map<std::string,ggml_tensor *> outputs;
    struct Input {ggml_tensor * tensor;std::vector<uint8_t> bytes;};
    std::vector<Input> inputs;
    explicit Graph(Model & m,bool d=false):model(m),dumping(d) {
        ctx=ggml_init({64*1024*1024,nullptr,true});
        if(!ctx)throw std::runtime_error("Cannot allocate encoder graph metadata");
    }
    ~Graph(){if(allocator)ggml_gallocr_free(allocator);if(ctx)ggml_free(ctx);}
    Graph(const Graph &)=delete;Graph & operator=(const Graph &)=delete;
    ggml_tensor * input(const float * data,int64_t d,int64_t n=1,int64_t b=1) {
        auto * t=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,d,n,b);ggml_set_input(t);
        Input entry{t,std::vector<uint8_t>(size_t(d*n*b)*4)};
        std::memcpy(entry.bytes.data(),data,entry.bytes.size());inputs.push_back(std::move(entry));return t;
    }
    ggml_tensor * ids(const std::vector<int32_t> & data) {
        auto * t=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,int64_t(data.size()));ggml_set_input(t);
        Input entry{t,std::vector<uint8_t>(data.size()*4)};
        std::memcpy(entry.bytes.data(),data.data(),entry.bytes.size());inputs.push_back(std::move(entry));return t;
    }
    ggml_tensor * mm(ggml_tensor * w,ggml_tensor * x) {
        if(w->ne[0]!=x->ne[0])throw std::runtime_error("Encoder matmul dimensions differ");
        auto * y=ggml_mul_mat(ctx,w,x);
        if(!ggml_prec_set_acc(y,GGML_PREC_F32)||!ggml_prec_set_src(y,GGML_PREC_F32,1))
            throw std::runtime_error("Cannot request encoder F32 matmul precision");
        return y;
    }
    ggml_tensor * linear(const std::string & p,ggml_tensor * x,bool bias=true) {
        auto * y=mm(model.tensor(p+".weight"),x);
        return bias?ggml_add(ctx,y,model.tensor(p+".bias")):y;
    }
    ggml_tensor * norm(const std::string & p,ggml_tensor * x,float eps=1e-5f) {
        return ggml_add(ctx,ggml_mul(ctx,ggml_norm(ctx,x,eps),model.tensor(p+".weight")),model.tensor(p+".bias"));
    }
    ggml_tensor * conv_kernel(ggml_tensor * w,ggml_tensor * x,int stride,int left,int right) {
        const int64_t k=w->ne[0],in=w->ne[1],out=w->ne[2];
        if(x->ne[0]!=in)throw std::runtime_error("Invalid encoder convolution channels");
        auto * time=ggml_cont(ctx,ggml_transpose(ctx,x));
        if(left||right)time=ggml_pad_ext(ctx,time,left,right,0,0,0,0,0,0);
        auto * cols=ggml_im2col(ctx,w,time,stride,0,0,0,1,0,false,GGML_TYPE_F32);
        cols=ggml_reshape_2d(ctx,cols,k*in,cols->ne[1]);
        return mm(ggml_reshape_2d(ctx,w,k*in,out),cols);
    }
    ggml_tensor * conv1(const std::string & p,ggml_tensor * x,int stride,int padding,bool bias=true,int groups=1) {
        auto * w=model.tensor(p+".weight");ggml_tensor * y=nullptr;
        if(groups==1)y=conv_kernel(w,x,stride,padding,padding);
        else {
            const int64_t group_in=x->ne[0]/groups,group_out=w->ne[2]/groups;
            if(group_in!=w->ne[1]||w->ne[2]%groups||x->ne[0]%groups)throw std::runtime_error("Invalid encoder grouped convolution");
            for(int group=0;group<groups;++group) {
                auto * kernel=ggml_view_3d(ctx,w,w->ne[0],group_in,group_out,w->nb[1],w->nb[2],size_t(group)*group_out*w->nb[2]);
                auto * piece=conv_kernel(kernel,slice(x,group*group_in,group_in),stride,padding,padding);
                y=y?ggml_concat(ctx,y,piece,0):piece;
            }
        }
        return bias?ggml_add(ctx,y,model.tensor(p+".bias")):y;
    }
    ggml_tensor * depthwise(const std::string & p,ggml_tensor * x,int padding) {
        const int64_t channels=x->ne[0],n=x->ne[1];auto * w=model.tensor(p+".weight");
        auto * time=ggml_reshape_3d(ctx,ggml_cont(ctx,ggml_transpose(ctx,x)),n,1,channels);
        auto * cols=ggml_im2col(ctx,w,time,1,0,padding,0,1,0,false,GGML_TYPE_F32);
        auto * kernel=ggml_reshape_3d(ctx,w,w->ne[0],1,channels);
        auto * sum=ggml_sum_rows(ctx,ggml_mul(ctx,cols,kernel));
        return ggml_cont(ctx,ggml_transpose(ctx,ggml_reshape_2d(ctx,sum,n,channels)));
    }
    ggml_tensor * conv2(const std::string & p,ggml_tensor * x,int stride,int padding) {
        auto * w=model.tensor(p+".weight");
        auto * cols=ggml_im2col(ctx,w,x,stride,stride,padding,padding,1,1,true,GGML_TYPE_F32);
        const int64_t width=cols->ne[1],height=cols->ne[2],out=w->ne[3];
        cols=ggml_reshape_2d(ctx,cols,cols->ne[0],width*height);
        auto * y=mm(ggml_reshape_2d(ctx,w,w->ne[0]*w->ne[1]*w->ne[2],out),cols);
        y=ggml_cont(ctx,ggml_permute(ctx,ggml_reshape_3d(ctx,y,out,width,height),2,0,1,3));
        return ggml_add(ctx,y,ggml_reshape_3d(ctx,model.tensor(p+".bias"),1,1,out));
    }
    ggml_tensor * batch_norm(const std::string & p,ggml_tensor * x,bool conv2d=false) {
        auto gamma=values(model.tensor(p+".weight")),beta=values(model.tensor(p+".bias"));
        auto mean=values(model.tensor(p+".running_mean")),var=values(model.tensor(p+".running_var"));
        for(size_t i=0;i<gamma.size();++i){gamma[i]/=std::sqrt(var[i]+1e-5f);beta[i]-=mean[i]*gamma[i];}
        auto * scale=input(gamma.data(),int64_t(gamma.size())),* shift=input(beta.data(),int64_t(beta.size()));
        if(conv2d){scale=ggml_reshape_3d(ctx,scale,1,1,int64_t(gamma.size()));shift=ggml_reshape_3d(ctx,shift,1,1,int64_t(beta.size()));}
        return ggml_add(ctx,ggml_mul(ctx,x,scale),shift);
    }
    ggml_tensor * keep(const std::string & name,ggml_tensor * x,bool required=false) {
        if(dumping||required) {x=ggml_cont(ctx,x);ggml_set_name(x,name.c_str());ggml_set_output(x);outputs[name]=x;}return x;
    }
    ggml_tensor * slice(ggml_tensor * x,int64_t first,int64_t width) {
        return ggml_cont(ctx,ggml_view_2d(ctx,x,width,x->ne[1],x->nb[1],size_t(first)*x->nb[0]));
    }
    ggml_tensor * rotary(ggml_tensor * x,const std::string & frequency_name,int width) {
        // The reference multiplies positions by its serialized F32 inv_freq.
        // Do not reconstruct those powers via backend-specific RoPE recurrence.
        const auto inv=values(model.tensor(frequency_name));
        if(int(inv.size())*2!=width||width>x->ne[0])throw std::runtime_error("Invalid encoder rotary frequencies");
        const int64_t heads=x->ne[1],frames=x->ne[2];
        std::vector<float> cosines(size_t(width*frames)),sines(size_t(width*frames));
        for(int64_t t=0;t<frames;++t)for(int d=0;d<width/2;++d) {
            const float angle=float(t)*inv[size_t(d)],co=std::cos(angle),si=std::sin(angle);
            for(int half=0;half<2;++half){cosines[size_t(t)*width+half*width/2+d]=co;sines[size_t(t)*width+half*width/2+d]=si;}
        }
        auto * co=input(cosines.data(),width,1,frames),* si=input(sines.data(),width,1,frames);
        auto * begin=ggml_cont(ctx,ggml_view_3d(ctx,x,width/2,heads,frames,x->nb[1],x->nb[2],0));
        auto * end=ggml_cont(ctx,ggml_view_3d(ctx,x,width/2,heads,frames,x->nb[1],x->nb[2],size_t(width/2)*4));
        auto * original=ggml_cont(ctx,ggml_view_3d(ctx,x,width,heads,frames,x->nb[1],x->nb[2],0));
        auto * rotated=ggml_concat(ctx,ggml_neg(ctx,end),begin,0);
        rotated=ggml_add(ctx,ggml_mul(ctx,original,co),ggml_mul(ctx,rotated,si));
        if(width<x->ne[0]) {
            auto * tail=ggml_cont(ctx,ggml_view_3d(ctx,x,x->ne[0]-width,heads,frames,x->nb[1],x->nb[2],size_t(width)*4));
            rotated=ggml_concat(ctx,rotated,tail,0);
        }
        return rotated;
    }
    ggml_tensor * attention(ggml_tensor * q,ggml_tensor * k,ggml_tensor * v,int dim,int heads,
                            ggml_tensor * bias=nullptr,bool prescale_query=false) {
        const int64_t nq=q->ne[1],nk=k->ne[1];
        if(prescale_query)q=ggml_scale(ctx,q,1.0f/std::sqrt(float(dim)));
        q=ggml_permute(ctx,ggml_reshape_3d(ctx,q,dim,heads,nq),0,2,1,3);
        k=ggml_permute(ctx,ggml_reshape_3d(ctx,k,dim,heads,nk),0,2,1,3);
        if(prescale_query){q=ggml_cont(ctx,q);k=ggml_cont(ctx,k);}
        auto * scores=mm(k,q);
        if(!prescale_query)scores=ggml_scale(ctx,scores,1.0f/std::sqrt(float(dim)));
        if(bias)scores=ggml_add(ctx,scores,bias);
        scores=ggml_soft_max(ctx,scores);
        v=ggml_reshape_3d(ctx,v,dim,heads,nk);
        v=ggml_cont(ctx,ggml_permute(ctx,v,1,2,0,3));
        auto * y=mm(v,scores);
        return ggml_reshape_2d(ctx,ggml_cont(ctx,ggml_permute(ctx,y,0,2,1,3)),dim*heads,nq);
    }
    void execute(const char * directory=nullptr) {
        auto * graph=ggml_new_graph_custom(ctx,16384,false);
        for(const auto & pair:outputs)ggml_build_forward_expand(graph,pair.second);
        for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto * node=ggml_graph_node(graph,i);
            if(!ggml_backend_supports_op(model.backend,node))
                throw std::runtime_error("Encoder backend lacks operation: "+std::string(ggml_op_name(node->op)));
        }
        allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if(!allocator||!ggml_gallocr_alloc_graph(allocator,graph))throw std::runtime_error("Cannot allocate encoder graph");
        model.graph_bytes=std::max(model.graph_bytes,ggml_gallocr_get_buffer_size(allocator,0));
        // gallocr can reuse storage. Restore every graph input, including IDs.
        for(const auto & i:inputs)ggml_backend_tensor_set(i.tensor,i.bytes.data(),0,i.bytes.size());
        if(ggml_backend_graph_compute(model.backend,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("Encoder graph execution failed");
        if(dumping&&directory&&*directory) {
            std::filesystem::create_directories(directory);
            for(const auto & pair:outputs) {auto data=values(pair.second);dump(std::filesystem::path(directory)/(pair.first+".f32"),data.data(),data.size());}
        }
    }
};
}}
