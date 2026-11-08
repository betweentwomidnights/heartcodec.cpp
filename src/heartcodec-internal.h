#pragma once
#include "heartcodec.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include <map>
#include <memory>
#include <stdexcept>
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
    bool flow = false;
    bool audio_encoder = false;
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
std::vector<int32_t> token_data(const heartcodec_tokens_v1 * tokens);
std::vector<float> rvq_decode(Model &, const int32_t *, int64_t frames, bool condition = false);
std::vector<float> flow_step(Model &, const float *, int64_t frames, float time, const char * dump);
std::vector<float> flow_decode(Model &, const int32_t *, int64_t frames, const heartcodec_flow_config_v1 &, const char * dump);
void validate_flow_config(const heartcodec_flow_config_v1 * config, int64_t frames);
void validate_audio_encoder(Model &);
Encoded encoder_query(Model &, const float * acoustic, const float * semantic,
                      const float * whisper, const float * wavlm, int64_t frames, const char * dump);
std::vector<int32_t> encoder_rvq(Model &, const float *, int64_t frames, const char * dump);
heartcodec_status_v1 error(heartcodec_error_v1 * out, heartcodec_status_v1 status, const char * message);
}
struct heartcodec_context {
    std::unique_ptr<heartcodec::Model> models[2];
    std::string paths[2];
    std::string backend;
    int threads = 4;
    heartcodec::Model & get(int component) {
        if (!models[component]) {
            if (paths[component].empty()) throw std::invalid_argument("Component model path is not registered");
            auto model = std::make_unique<heartcodec::Model>();
            model->load(paths[component].c_str(), backend.c_str(), threads);
            if (model->audio_encoder || model->flow != (component == 1)) throw std::invalid_argument("Wrong component architecture");
            models[component] = std::move(model);
        }
        return *models[component];
    }
};
