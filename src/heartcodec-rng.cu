// PyTorch contiguous F32 distribution mapping. Philox integer rounds adapted
// from sa3.cpp src/rng.h (MIT, copyright 2026 betweentwomidnights).
// Grid/counter semantics follow sa3.cpp src/rng.h and
// ATen/native/cuda/DistributionTemplates.h (PyTorch BSD-3-Clause; see notice).
#include "heartcodec_rng.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(std::string("RNG CUDA: ")+cudaGetErrorString(status));}
struct DeviceGuard {
    int previous;explicit DeviceGuard(int device){check(cudaGetDevice(&previous));check(cudaSetDevice(device));}
    ~DeviceGuard(){cudaSetDevice(previous);}
};
struct Buffer {float * p=nullptr;~Buffer(){if(p)cudaFree(p);}};
__device__ uint4 philox(uint4 c,uint32_t k0,uint32_t k1) {
    for(int r=0;r<10;++r) {
        const uint64_t p0=uint64_t(0xD2511F53u)*c.x,p1=uint64_t(0xCD9E8D57u)*c.z;
        c=make_uint4(uint32_t(p1>>32)^c.y^k0,uint32_t(p1),uint32_t(p0>>32)^c.w^k1,uint32_t(p0));
        k0+=0x9E3779B9u;k1+=0xBB67AE85u;
    }
    return c;
}
__device__ void normal_pair(uint32_t a,uint32_t b,float & x,float & y) {
    constexpr float inv=2.3283064365386963e-10f,angle_inv=1.4629180792671596e-09f;
    const float u=float(a)*inv+inv*.5f,v=float(b)*angle_inv+angle_inv*.5f;
    const float radius=sqrtf(-2.0f*logf(u));__sincosf(v,&x,&y);x*=radius;y*=radius;
}
template<bool Normal> __global__ void draw_kernel(float * out,uint64_t seed,uint64_t offset,
    uint64_t count,uint64_t grid,uint64_t delta,uint64_t stride,uint64_t trips) {
    const uint64_t draw=uint64_t(blockIdx.x)/grid,block=uint64_t(blockIdx.x)%grid;
    const uint64_t idx=block*256+threadIdx.x;
    for(uint64_t trip=0;trip<trips;++trip) {
        const uint64_t counter=(offset+draw*delta)/4+trip;
        const uint4 words=philox(make_uint4(uint32_t(counter),uint32_t(counter>>32),uint32_t(idx),uint32_t(idx>>32)),uint32_t(seed),uint32_t(seed>>32));
        float v[4];
        if(Normal){normal_pair(words.x,words.y,v[0],v[1]);normal_pair(words.z,words.w,v[2],v[3]);}
        else {
            const uint32_t w[4]={words.x,words.y,words.z,words.w};
            for(int k=0;k<4;++k)v[k]=float(w[k])*0x1p-32f+0x1p-33f;
        }
        for(int k=0;k<4;++k) {
            const uint64_t i=idx+(trip*4+uint64_t(k))*stride;
            if(i<count) {
                float value;
                if(Normal)value=v[k]*1.0f+0.0f;
                else value=v[k]>=1.0f-0x1p-24f?0x1p-24f:-__logf(v[k]);
                out[draw*count+i]=value;
            }
        }
    }
}
}
namespace heartcodec {
void rng_cuda(const heartcodec_rng_config_v1 & c,float * out,uint64_t & next,uint32_t & grid_out,uint32_t & cap_out) {
    DeviceGuard guard(c.device);cudaDeviceProp props;check(cudaGetDeviceProperties(&props,c.device));
    const uint64_t cap=uint64_t(props.multiProcessorCount)*(props.maxThreadsPerMultiProcessor/256);
    cap_out=uint32_t(cap);
    if(!c.count||!c.draws)return;
    const uint64_t grid=std::min((uint64_t(c.count)+255)/256,c.max_blocks?uint64_t(c.max_blocks):cap);
    if(!grid||uint64_t(c.draws)>uint64_t(props.maxGridSize[0])/grid)throw std::invalid_argument("RNG CUDA launch size exceeds device limit");
    const uint64_t stride=grid*256,trips=(uint64_t(c.count)-1)/(4*stride)+1,delta=trips*4;
    if(uint64_t(c.draws)>(UINT64_MAX-c.offset)/delta)throw std::invalid_argument("Philox offset overflow");
    next=c.offset+uint64_t(c.draws)*delta;grid_out=uint32_t(grid);
    Buffer buffer;const size_t bytes=size_t(c.count)*size_t(c.draws)*sizeof(float);check(cudaMalloc(reinterpret_cast<void **>(&buffer.p),bytes));
    if(c.distribution==HEARTCODEC_RNG_NORMAL_V1)draw_kernel<true><<<unsigned(grid*uint64_t(c.draws)),256>>>(buffer.p,c.seed,c.offset,c.count,grid,delta,stride,trips);
    else draw_kernel<false><<<unsigned(grid*uint64_t(c.draws)),256>>>(buffer.p,c.seed,c.offset,c.count,grid,delta,stride,trips);
    check(cudaGetLastError());check(cudaMemcpy(out,buffer.p,bytes,cudaMemcpyDeviceToHost));
}
}
