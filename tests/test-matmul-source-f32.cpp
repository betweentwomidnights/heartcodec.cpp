// Consumer reproducer for the public b77ae79 source-precision contract.
#include "ggml.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static bool shape(ggml_backend_t backend,int n) {
    const int k=512,m=64;
    std::vector<float> a(k*m),b(k*n),actual(m*n);
    for(size_t i=0;i<a.size();++i)a[i]=0.3f*std::sin(float(i)*0.137f)+0.00013f;
    for(size_t i=0;i<b.size();++i)b[i]=0.2f*std::cos(float(i)*0.171f)-0.00017f;
    auto * ctx=ggml_init({4*1024*1024,nullptr,true});if(!ctx)return false;
    auto * ta=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,m),* tb=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n);
    auto * out=ggml_mul_mat(ctx,ta,tb);
    bool passed=ggml_prec_set_acc(out,GGML_PREC_F32)&&ggml_prec_set_src(out,GGML_PREC_F32,1);
    auto * graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);if(!buffer){ggml_free(ctx);return false;}
    ggml_backend_tensor_set(ta,a.data(),0,a.size()*4);ggml_backend_tensor_set(tb,b.data(),0,b.size()*4);
    passed=ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS&&passed;
    ggml_backend_tensor_get(out,actual.data(),0,actual.size()*4);
    double err=0,ref=0,maximum=0;
    for(int col=0;col<n;++col)for(int row=0;row<m;++row) {
        double expected=0;for(int i=0;i<k;++i)expected+=double(a[row*k+i])*b[col*k+i];
        double d=double(actual[col*m+row])-expected;passed=std::isfinite(actual[col*m+row])&&passed;
        err+=d*d;ref+=expected*expected;maximum=std::fmax(maximum,std::fabs(d));
    }
    double relative=std::sqrt(err/ref);passed=passed&&relative<1e-5&&maximum<1e-5;
    std::printf("%s F32 operands/acc K=%d M=%d N=%d: max_abs=%.9g relative_l2=%.9g %s\n",
        ggml_backend_name(backend),k,m,n,maximum,relative,passed?"PASS":"FAIL");
    ggml_backend_buffer_free(buffer);ggml_free(ctx);return passed;
}
int main(int argc,char ** argv) {
    ggml_backend_load_all();bool passed=true,tested=false;
    for(size_t i=0;i<ggml_backend_dev_count();++i) {
        auto dev=ggml_backend_dev_get(i);if(argc>1&&std::string(ggml_backend_dev_name(dev))!=argv[1])continue;
        auto backend=ggml_backend_dev_init(dev,nullptr);if(!backend)continue;tested=true;
        for(int n:{1,5,8,16,17,25,64})passed=shape(backend,n)&&passed;
        ggml_backend_free(backend);
    }
    return !tested?77:passed?0:1;
}
