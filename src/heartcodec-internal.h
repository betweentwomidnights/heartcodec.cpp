#pragma once
#include "heartcodec.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include <map>
#include <set>
#include <string>
#include <vector>

namespace heartcodec {
struct ConvSpec { int kernel, in_channels, out_channels; };
struct Model {
    ggml_context * weights = nullptr;
    gguf_context * gguf = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::map<std::string, float> slopes;
    std::map<std::string, ConvSpec> convolutions;
    std::string part;
    bool matrix_layout = false;
    size_t graph_bytes = 0;
    Model() = default;
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;
    ~Model();
    void load(const char * path, const char * backend_name, int threads);
    ggml_tensor * tensor(const std::string & name) const;
    void validate();
};
std::vector<float> sq_decode(Model & model, const float * latents, int64_t frames, const char * dump);
struct Encoded { std::vector<float> data; int64_t frames; };
Encoded sq_encode(Model & model, const float * waveform, int64_t samples, const char * dump);
heartcodec_status_v1 error(heartcodec_error_v1 * out, heartcodec_status_v1 status, const char * message);
}
struct heartcodec_context { heartcodec::Model model; };
