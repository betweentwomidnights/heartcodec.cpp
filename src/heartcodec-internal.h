#pragma once
#include "heartcodec.h"
#include "heartcodec_work.h"
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
// "MUL_MAT f32[1024,4096,1,1] (bytes 16777216) <- f32[..] f32[..]" for backend refusal messages.
inline std::string describe_op(const ggml_tensor * t) {
    auto shape = [](const ggml_tensor * x) {
        return std::string(ggml_type_name(x->type)) + "[" + std::to_string(x->ne[0]) + "," + std::to_string(x->ne[1]) + "," +
               std::to_string(x->ne[2]) + "," + std::to_string(x->ne[3]) + "]" + (ggml_is_contiguous(x) ? "" : " strided");
    };
    std::string text = std::string(ggml_op_name(t->op)) + " " + shape(t) + " (" + std::to_string(ggml_nbytes(t)) + " bytes) <-";
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; ++i)
        text += " " + shape(t->src[i]) + " (" + std::to_string(ggml_nbytes(t->src[i])) + " bytes)";
    return text;
}
struct Work {
    const heartcodec_work_control_v1 * control;
    bool cancelled=false;
    void check() {
        if(control&&control->should_cancel&&control->should_cancel(control->user)) {
            cancelled=true;throw std::runtime_error("HeartCodec work cancelled");
        }
    }
    void progress(int32_t stage,int64_t completed,int64_t total) {
        check();
        if(control&&control->on_progress){heartcodec_work_progress_v1 p{sizeof(p),stage,completed,total};control->on_progress(control->user,&p);}
        check();
    }
};
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
    Work * work = nullptr;
    void checkpoint(){if(work)work->check();}
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
    heartcodec::Work * work = nullptr;
    std::unique_ptr<heartcodec::Model> models[2];
    std::string paths[2];
    std::string backend;
    int threads = 4;
    heartcodec::Model & get(int component) {
        if (!models[component]) {
            if (paths[component].empty()) throw std::invalid_argument("Component model path is not registered");
            auto model = std::make_unique<heartcodec::Model>();
            model->work=work;
            model->load(paths[component].c_str(), backend.c_str(), threads);
            if (model->audio_encoder || model->flow != (component == 1)) throw std::invalid_argument("Wrong component architecture");
            models[component] = std::move(model);
        }
        return *models[component];
    }
};
