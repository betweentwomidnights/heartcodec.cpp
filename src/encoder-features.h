#pragma once
#include "encoder-graph.h"
#include <array>
namespace heartcodec { namespace encoder {
std::vector<float> resample(Model &,const float * audio,int64_t samples,int rate,int target,int channels);
struct Mel {std::vector<float> mu,whisper;};
Mel mel_features(Model &,const float * chunk24,const float * audio16,const char * dump);
std::vector<float> whisper(Model &,const float * mel,const char * dump,const float * hidden=nullptr,int first_layer=0,int end_layer=12);
std::vector<float> wavlm(Model &,const float * waveform,int64_t samples,const char * dump);
std::array<std::vector<float>,2> mu(Model &,const float * mel,const char * dump);
std::array<std::vector<float>,4> features(Model &,const float * chunk24,const char * dump);
}}
