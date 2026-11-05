#include "heartcodec-internal.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace {
template<class F> heartcodec_status_v1 guarded(heartcodec_error_v1 * e, F && f) {
    try { f(); return heartcodec::error(e, HEARTCODEC_OK_V1, ""); }
    catch (const std::invalid_argument & ex) { return heartcodec::error(e, HEARTCODEC_INVALID_ARGUMENT_V1, ex.what()); }
    catch (const std::exception & ex) { return heartcodec::error(e, HEARTCODEC_ERROR_V1, ex.what()); }
    catch (...) { return heartcodec::error(e, HEARTCODEC_ERROR_V1, "Unknown HeartCodec error"); }
}
void check_component(heartcodec_context * c, heartcodec_component_v1 component) {
    if (!c || (component != HEARTCODEC_SQ_V1 && component != HEARTCODEC_FLOW_V1))
        throw std::invalid_argument("Invalid context or component");
}
void check_result(heartcodec_context * c, heartcodec_tensor_v1 * t) {
    if (!c || !t || t->size < sizeof(*t) || t->data) throw std::invalid_argument("Invalid or occupied tensor result");
}
void assign(heartcodec_tensor_v1 * t, const std::vector<float> & data, int channels) {
    auto values = std::make_unique<float[]>(data.size());
    std::copy(data.begin(), data.end(), values.get());
    t->data = values.release(); t->frames = int64_t(data.size() / channels); t->channels = channels;
}
template<class T> void init(T * value) {
    if (!value || value->size < sizeof(*value)) return;
    const auto size = value->size; std::memset(value, 0, sizeof(*value)); value->size = size;
}
}

extern "C" {
heartcodec_status_v1 heartcodec_component_set_v1(heartcodec_context * c, heartcodec_component_v1 component,
                                                const char * path, heartcodec_error_v1 * e) {
    return guarded(e, [&] {
        check_component(c, component);
        if (!path || !*path) throw std::invalid_argument("Missing component path");
        std::string registered(path); // Own input before releasing a component.
        c->models[component].reset(); c->paths[component] = std::move(registered);
    });
}
heartcodec_status_v1 heartcodec_component_load_v1(heartcodec_context * c, heartcodec_component_v1 component, heartcodec_error_v1 * e) {
    return guarded(e, [&] { check_component(c, component); c->get(component); });
}
heartcodec_status_v1 heartcodec_component_unload_v1(heartcodec_context * c, heartcodec_component_v1 component, heartcodec_error_v1 * e) {
    return guarded(e, [&] { check_component(c, component); c->models[component].reset(); });
}
void heartcodec_tensor_init_v1(heartcodec_tensor_v1 * t) { init(t); }
void heartcodec_tensor_free_v1(heartcodec_tensor_v1 * t) {
    if (!t || t->size < sizeof(*t)) return;
    delete[] t->data; init(t);
}
void heartcodec_stereo_init_v1(heartcodec_stereo_v1 * a) { init(a); }
void heartcodec_stereo_free_v1(heartcodec_stereo_v1 * a) {
    if (!a || a->size < sizeof(*a)) return;
    delete[] a->samples; init(a);
}
void heartcodec_flow_config_init_v1(heartcodec_flow_config_v1 * c) {
    if (!c || c->size < sizeof(*c)) return;
    init(c); c->steps = 10; c->guidance = 1.25f; c->seed = 1701; c->segment_codes = 372;
}
heartcodec_status_v1 heartcodec_rvq_decode_v1(heartcodec_context * c, const heartcodec_tokens_v1 * tokens,
                                            heartcodec_tensor_v1 * t, heartcodec_error_v1 * e) {
    return guarded(e, [&] {
        check_result(c, t);
        const auto codes = heartcodec::token_data(tokens);
        assign(t, heartcodec::rvq_decode(c->get(1), codes.data(), tokens->frames), 512);
    });
}
heartcodec_status_v1 heartcodec_flow_step_v1(heartcodec_context * c, const float * input, int64_t frames,
                                          float time, const char * dump, heartcodec_tensor_v1 * t, heartcodec_error_v1 * e) {
    return guarded(e, [&] { check_result(c, t); assign(t, heartcodec::flow_step(c->get(1), input, frames, time, dump), 256); });
}
heartcodec_status_v1 heartcodec_flow_decode_v1(heartcodec_context * c, const heartcodec_tokens_v1 * tokens,
        const heartcodec_flow_config_v1 * config, const char * dump, heartcodec_tensor_v1 * t, heartcodec_error_v1 * e) {
    return guarded(e, [&] {
        check_result(c, t);
        const auto codes = heartcodec::token_data(tokens);
        heartcodec::validate_flow_config(config, tokens->frames);
        assign(t, heartcodec::flow_decode(c->get(1), codes.data(), tokens->frames, *config, dump), 256);
    });
}
heartcodec_status_v1 heartcodec_tokens_decode_v1(heartcodec_context * c, const heartcodec_tokens_v1 * tokens,
        const heartcodec_flow_config_v1 * config, heartcodec_stereo_v1 * output, heartcodec_error_v1 * e) {
    return guarded(e, [&] {
        if (!c || !output || output->size < sizeof(*output) || output->samples)
            throw std::invalid_argument("Invalid or occupied stereo output");
        const auto codes = heartcodec::token_data(tokens);
        heartcodec::validate_flow_config(config, tokens->frames);
        if (config->context || config->prefix_frames)
            throw std::invalid_argument("Token-to-audio decode requires empty explicit context");
        if (c->paths[0].empty() || c->paths[1].empty())
            throw std::invalid_argument("Both SQ and flow model paths must be registered");
        if(c->work)c->work->check();
        const int64_t window = config->segment_codes, hop = window / 93 * 80, overlap = window - hop;
        const int64_t length = std::max(window, tokens->frames);
        if (length > std::numeric_limits<int64_t>::max() - hop)
            throw std::invalid_argument("Too many token frames");
        const int64_t padded = ((length - overlap + hop - 1) / hop) * hop + overlap;
        const int64_t segments = (padded - window) / hop + 1;
        if ((config->noise && config->noise_segments != segments) || (!config->noise && config->noise_segments))
            throw std::invalid_argument("Explicit noise segment count does not match padded token windows");
        std::vector<std::vector<float>> latents;
        std::vector<int32_t> segment(size_t(window) * 8);
        std::vector<float> context(size_t(window) * 2 * 256, 0);
        c->models[0].reset();
        auto & flow = c->get(1);
        for (int64_t start = 0; start + window <= padded; start += hop) {
            for (int q = 0; q < 8; ++q) for (int64_t i = 0; i < window; ++i)
                segment[size_t(q * window + i)] = codes[size_t(q * tokens->frames + (start + i) % tokens->frames)];
            auto cfg = *config;
            cfg.seed += uint64_t(latents.size());
            if (config->noise) {
                cfg.noise = config->noise + latents.size() * size_t(window) * 2 * 256;
                cfg.noise_segments = 1;
            }
            if (!latents.empty()) {
                std::copy_n(latents.back().data() + (window - overlap) * 2 * 256, size_t(overlap) * 2 * 256, context.data());
                cfg.context = context.data(); cfg.prefix_frames = overlap * 2;
            }
            latents.push_back(heartcodec::flow_decode(flow, segment.data(), window, cfg, nullptr));
            if(c->work)c->work->progress(HEARTCODEC_WORK_FLOW_V1,int64_t(latents.size()),segments);
        }
        c->models[1].reset(); // Destroy graph and weights before SQ allocation.
        auto & sq = c->get(0);
        const int64_t window_samples = window * 3840, hop_samples = hop * 3840, overlap_samples = overlap * 3840;
        const int64_t total_samples = int64_t(latents.size() - 1) * hop_samples + window_samples;
        const int64_t target = tokens->frames * 3840;
        std::vector<float> channel_output[2];
        for (auto & channel : channel_output) channel.resize(size_t(total_samples));
        std::vector<float> sq_input(size_t(window) * 2 * 128);
        for (size_t i = 0; i < latents.size(); ++i) for (int channel = 0; channel < 2; ++channel) {
            for (int d = 0; d < 128; ++d) for (int64_t t = 0; t < window * 2; ++t)
                sq_input[size_t(d * window * 2 + t)] = latents[i][size_t(t * 256 + channel * 128 + d)];
            const auto audio = heartcodec::sq_decode(sq, sq_input.data(), window * 2, nullptr);
            auto & assembled = channel_output[channel];
            const size_t start = i * size_t(hop_samples);
            for (size_t t = 0; t < size_t(window_samples); ++t) {
                if (i && t < size_t(overlap_samples)) {
                    // Upstream numpy linspace and mixing use float64, then
                    // assign back into the float32 output, including endpoints.
                    const double weight = double(t) / double(overlap_samples - 1);
                    assembled[start + t] = float(double(assembled[start + t]) * (1 - weight) + double(audio[t]) * weight);
                } else assembled[start + t] = audio[t];
            }
            if(c->work)c->work->progress(HEARTCODEC_WORK_SQ_V1,int64_t(i)*2+channel+1,segments*2);
        }
        auto samples = std::make_unique<float[]>(size_t(target) * 2);
        for (int channel = 0; channel < 2; ++channel)
            std::copy_n(channel_output[channel].data(), size_t(target), samples.get() + size_t(channel * target));
        output->samples = samples.release(); output->num_samples = target; output->sample_rate = 48000;
    });
}

const heartcodec_api_v1 * HEARTCODEC_CALL heartcodec_get_api(uint32_t version) {
    static const heartcodec_api_v1 api = {
        uint32_t(HEARTCODEC_API_V1_SIZE), HEARTCODEC_ABI_VERSION_1,
        heartcodec_config_init_v1, heartcodec_error_init_v1, heartcodec_context_create_v1, heartcodec_context_destroy_v1,
        heartcodec_audio_init_v1, heartcodec_audio_free_v1, heartcodec_sq_latents_init_v1, heartcodec_sq_latents_free_v1,
        heartcodec_sq_encode_v1, heartcodec_sq_decode_v1, heartcodec_sq_quantize_v1, heartcodec_get_memory_v1,
        heartcodec_component_set_v1, heartcodec_component_load_v1, heartcodec_component_unload_v1,
        heartcodec_tensor_init_v1, heartcodec_tensor_free_v1, heartcodec_stereo_init_v1, heartcodec_stereo_free_v1,
        heartcodec_flow_config_init_v1, heartcodec_rvq_decode_v1, heartcodec_flow_step_v1, heartcodec_flow_decode_v1,
        heartcodec_tokens_decode_v1
    };
    return version == HEARTCODEC_ABI_VERSION_1 ? &api : nullptr;
}
}
