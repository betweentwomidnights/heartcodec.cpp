#include "heartcodec-internal.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

namespace heartcodec {
heartcodec_status_v1 error(heartcodec_error_v1 * out, heartcodec_status_v1 status, const char * message) {
    if (out && out->size >= sizeof(*out)) {
        out->status = status;
        std::snprintf(out->message, sizeof(out->message), "%s", message);
    }
    return status;
}
}
extern "C" {
void heartcodec_config_init_v1(heartcodec_config_v1 * c) {
    if (!c || c->size < sizeof(*c)) return;
    const auto size = c->size;
    std::memset(c, 0, sizeof(*c));
    c->size = size; c->backend = "cpu"; c->cpu_threads = 4;
}
void heartcodec_audio_init_v1(heartcodec_audio_v1 * a) {
    if (!a || a->size < sizeof(*a)) return;
    const auto size = a->size; std::memset(a, 0, sizeof(*a)); a->size = size;
}
void heartcodec_error_init_v1(heartcodec_error_v1 * e) {
    if (!e || e->size < sizeof(*e)) return;
    const auto size = e->size; std::memset(e, 0, sizeof(*e)); e->size = size;
}
heartcodec_status_v1 heartcodec_sq_quantize_v1(const float * input, float * output, size_t count, heartcodec_error_v1 * e) {
    if (count && (!input || !output)) return heartcodec::error(e, HEARTCODEC_INVALID_ARGUMENT_V1, "Null quantizer buffer");
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(input[i]) || !std::isfinite(9.0f * input[i]))
            return heartcodec::error(e, HEARTCODEC_INVALID_ARGUMENT_V1, "Nonfinite or overflowing SQ latent");
    for (size_t i = 0; i < count; ++i) {
        const float scaled = 9.0f * input[i], low = std::floor(scaled), fraction = scaled - low;
        // C roundf uses ties away from zero. Torch round requires ties to even.
        float rounded = fraction < 0.5f ? low : fraction > 0.5f ? low + 1 :
                        std::fmod(low, 2.0f) == 0.0f ? low : low + 1;
        if (rounded == 0) rounded = std::copysign(0.0f, scaled);
        output[i] = rounded / 9.0f;
    }
    return heartcodec::error(e, HEARTCODEC_OK_V1, "");
}
heartcodec_status_v1 heartcodec_context_create_v1(const heartcodec_config_v1 * c, heartcodec_context ** out, heartcodec_error_v1 * e) {
    if (out) *out = nullptr;
    if (!c || c->size < sizeof(*c) || !out || !c->model_path || !c->backend || c->cpu_threads < 1 || c->cpu_threads > 1024)
        return heartcodec::error(e, HEARTCODEC_INVALID_ARGUMENT_V1, "Invalid decoder configuration");
    try {
        auto context = std::make_unique<heartcodec_context>();
        context->backend = c->backend; context->threads = c->cpu_threads;
        auto model = std::make_unique<heartcodec::Model>();
        model->load(c->model_path, c->backend, c->cpu_threads);
        const int component = model->flow ? 1 : 0;
        context->paths[component] = c->model_path;
        context->models[component] = std::move(model);
        *out = context.release();
        return heartcodec::error(e, HEARTCODEC_OK_V1, "");
    } catch (const std::invalid_argument & ex) { return heartcodec::error(e, HEARTCODEC_INVALID_ARGUMENT_V1, ex.what()); }
    catch (const std::exception & ex) { return heartcodec::error(e, HEARTCODEC_ERROR_V1, ex.what()); }
    catch (...) { return heartcodec::error(e, HEARTCODEC_ERROR_V1, "Unknown decoder creation error"); }
}
void heartcodec_context_destroy_v1(heartcodec_context * c) { delete c; }
void heartcodec_sq_latents_init_v1(heartcodec_sq_latents_v1 * l) {
    if (!l || l->size < sizeof(*l)) return;
    const auto size = l->size; std::memset(l, 0, sizeof(*l)); l->size = size;
}
void heartcodec_sq_latents_free_v1(heartcodec_sq_latents_v1 * l) {
    if (!l || l->size < sizeof(*l)) return;
    delete[] l->data; heartcodec_sq_latents_init_v1(l);
}
heartcodec_status_v1 heartcodec_get_memory_v1(heartcodec_context * c, heartcodec_memory_v1 * m, heartcodec_error_v1 * e) {
    if (!c || !m || m->size < sizeof(*m))
        return heartcodec::error(e, HEARTCODEC_INVALID_ARGUMENT_V1, "Invalid memory statistics request");
    m->weight_buffer_bytes = m->graph_buffer_bytes = 0;
    for (const auto & model : c->models) if (model) {
        m->weight_buffer_bytes += ggml_backend_buffer_get_size(model->buffer);
        m->graph_buffer_bytes += model->graph_bytes;
    }
    return heartcodec::error(e, HEARTCODEC_OK_V1, "");
}
void heartcodec_audio_free_v1(heartcodec_audio_v1 * a) {
    if (!a || a->size < sizeof(*a)) return;
    delete[] a->samples;
    heartcodec_audio_init_v1(a);
}
heartcodec_status_v1 heartcodec_sq_decode_v1(heartcodec_context * c, const float * latents, int64_t frames,
                                           const char * dump, heartcodec_audio_v1 * a, heartcodec_error_v1 * e) {
    if (!c || !latents || !a || a->size < sizeof(*a) || a->samples || frames <= 0 ||
        uint64_t(frames) > std::numeric_limits<size_t>::max() / (1920 * sizeof(float)) ||
        frames > std::numeric_limits<int64_t>::max() / 1920)
        return heartcodec::error(e, HEARTCODEC_INVALID_ARGUMENT_V1, "Invalid SQ decode buffers or frame count");
    try {
        auto data = heartcodec::sq_decode(c->get(0), latents, frames, dump);
        if (!std::all_of(data.begin(), data.end(), [](float value) { return std::isfinite(value); }))
            throw std::runtime_error("SQ decoder produced nonfinite output");
        auto samples = std::make_unique<float[]>(data.size());
        std::copy(data.begin(), data.end(), samples.get());
        a->samples = samples.release(); a->num_samples = int64_t(data.size()); a->sample_rate = 48000;
        return heartcodec::error(e, HEARTCODEC_OK_V1, "");
    } catch (const std::invalid_argument & ex) { return heartcodec::error(e, HEARTCODEC_INVALID_ARGUMENT_V1, ex.what()); }
    catch (const std::exception & ex) { return heartcodec::error(e, HEARTCODEC_ERROR_V1, ex.what()); }
    catch (...) { return heartcodec::error(e, HEARTCODEC_ERROR_V1, "Unknown SQ decode error"); }
}
heartcodec_status_v1 heartcodec_sq_encode_v1(heartcodec_context * c, const float * waveform, int64_t samples,
                                           const char * dump, heartcodec_sq_latents_v1 * l, heartcodec_error_v1 * e) {
    if (!c || !waveform || !l || l->size < sizeof(*l) || l->data || samples < 2 ||
        uint64_t(samples) > std::numeric_limits<size_t>::max() / sizeof(float))
        return heartcodec::error(e, HEARTCODEC_INVALID_ARGUMENT_V1, "Invalid SQ encode buffers or sample count");
    try {
        if (!std::all_of(waveform, waveform + samples, [](float x) { return std::isfinite(x); }))
            throw std::invalid_argument("Waveform must be finite");
        auto result = heartcodec::sq_encode(c->get(0), waveform, samples, dump);
        if (!std::all_of(result.data.begin(), result.data.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("SQ encoder produced nonfinite output");
        auto data = std::make_unique<float[]>(result.data.size());
        std::copy(result.data.begin(), result.data.end(), data.get());
        l->data = data.release(); l->frames = result.frames; l->channels = 128;
        return heartcodec::error(e, HEARTCODEC_OK_V1, "");
    } catch (const std::invalid_argument & ex) { return heartcodec::error(e, HEARTCODEC_INVALID_ARGUMENT_V1, ex.what()); }
    catch (const std::exception & ex) { return heartcodec::error(e, HEARTCODEC_ERROR_V1, ex.what()); }
    catch (...) { return heartcodec::error(e, HEARTCODEC_ERROR_V1, "Unknown SQ encode error"); }
}
}
