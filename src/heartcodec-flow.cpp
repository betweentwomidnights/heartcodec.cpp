#include "heartcodec-internal.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>

namespace heartcodec {
namespace {
void finite(const float * values, size_t count) {
    if (!values || !std::all_of(values, values + count, [](float x) { return std::isfinite(x); }))
        throw std::invalid_argument("Nonfinite or missing flow tensor");
}
void dump_values(const std::filesystem::path & path, const std::vector<float> & data) {
    std::ofstream file(path, std::ios::binary);
    if (!file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float)))
        throw std::runtime_error("Cannot write flow dump: " + path.string());
}
struct Graph {
    Model & model;
    ggml_context * ctx = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * output = nullptr;
    std::vector<ggml_tensor *> stages;
    bool dumping;
    explicit Graph(Model & m, bool dump = false) : model(m), dumping(dump) {
        if (!m.flow) throw std::invalid_argument("Model does not contain flow weights");
        ctx = ggml_init({32 * 1024 * 1024, nullptr, true});
        if (!ctx) throw std::runtime_error("Cannot allocate flow graph metadata");
    }
    ~Graph() { if (allocator) ggml_gallocr_free(allocator); if (ctx) ggml_free(ctx); }
    ggml_tensor * mm(ggml_tensor * a, ggml_tensor * b) {
        auto * y = ggml_mul_mat(ctx, a, b);
        if (!ggml_prec_set_acc(y, GGML_PREC_F32) || !ggml_prec_set_src(y, GGML_PREC_F32, 1))
            throw std::runtime_error("Cannot request flow matmul precision");
        return y;
    }
    ggml_tensor * linear(const std::string & p, ggml_tensor * x, bool bias = true) {
        auto * y = mm(model.tensor(p + ".weight"), x);
        return bias ? ggml_add(ctx, y, model.tensor(p + ".bias")) : y;
    }
    ggml_tensor * keep(ggml_tensor * t, const std::string & name) {
        if (dumping) {
            if (t->view_src) t = ggml_cont(ctx, t);
            ggml_set_name(t, name.c_str()); ggml_set_output(t); stages.push_back(t);
        }
        return t;
    }
    ggml_tensor * project(const std::string & p, ggml_tensor * x) {
        auto * w = model.tensor(p + ".ffn_1.weight");
        auto * input = ggml_cont(ctx, ggml_transpose(ctx, x));
        auto * cols = ggml_im2col(ctx, w, input, 1, 0, 1, 0, 1, 0, false, GGML_TYPE_F32);
        cols = ggml_reshape_2d(ctx, cols, 3 * x->ne[0], x->ne[1]);
        auto * y = mm(ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]), cols);
        y = ggml_add(ctx, y, model.tensor(p + ".ffn_1.bias"));
        y = ggml_scale(ctx, y, 1.0f / std::sqrt(3.0f));
        return keep(linear(p + ".ffn_2", y), "stage." + p.substr(10));
    }
    ggml_tensor * row(ggml_tensor * x, int64_t d, int row) {
        return ggml_view_1d(ctx, x, d, size_t(row) * size_t(d) * sizeof(float));
    }
    ggml_tensor * modulate(ggml_tensor * x, ggml_tensor * shift, ggml_tensor * scale) {
        return ggml_add(ctx, ggml_mul(ctx, x, ggml_add1(ctx, scale, one)), shift);
    }
    ggml_tensor * one = nullptr;
    void allocate(ggml_tensor * result) {
        output = result; ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx, 16384, false);
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * n = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(model.backend, n))
                throw std::runtime_error("Unsupported flow operation: " + describe_op(n));
        }
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph))
            throw std::runtime_error("Cannot allocate flow graph");
        model.graph_bytes = std::max(model.graph_bytes, ggml_gallocr_get_buffer_size(allocator, 0));
    }
    std::vector<float> read(ggml_tensor * t) {
        std::vector<float> result(size_t(ggml_nelements(t)));
        ggml_backend_tensor_get(t, result.data(), 0, result.size() * sizeof(float));
        finite(result.data(), result.size());
        return result;
    }
    std::vector<float> run(const char * directory = nullptr) {
        model.checkpoint();
        if (ggml_backend_graph_compute(model.backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Flow graph execution failed");
        model.checkpoint();
        if (directory && *directory) {
            std::filesystem::create_directories(directory);
            for (auto * t : stages) dump_values(std::filesystem::path(directory) / (std::string(t->name) + ".f32"), read(t));
        }
        return read(output);
    }
};

struct Estimator : Graph {
    ggml_tensor * input;
    ggml_tensor * time_embedding;
    ggml_tensor * positions;
    std::vector<int32_t> position_values;
    Estimator(Model & m, int64_t frames, bool dump = false) : Graph(m, dump) {
        input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, frames); ggml_set_input(input);
        time_embedding = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512); ggml_set_input(time_embedding);
        positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, frames); ggml_set_input(positions);
        one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1); ggml_set_input(one);
        auto * x = project("estimator.proj_in", input);
        x = group(x, 0);
        x = project("estimator.connection_proj", ggml_concat(ctx, input, x, 0));
        x = group(x, 1);
        allocate(project("estimator.proj_out", x));
        position_values.resize(size_t(frames));
        for (int32_t i = 0; i < frames; ++i) position_values[size_t(i)] = i;
    }
    ggml_tensor * attention(const std::string & p, ggml_tensor * x, int d) {
        const auto frames = x->ne[1];
        auto rotate = [&](const char * name) {
            auto * t = ggml_reshape_3d(ctx, linear(p + "." + name, x, false), d, 24, frames);
            t = ggml_rope_ext(ctx, t, positions, nullptr, d, GGML_ROPE_TYPE_NORMAL, 0, 10000, 1, 0, 1, 0, 0);
            return ggml_permute(ctx, t, 0, 2, 1, 3); // [head_dim,time,heads]
        };
        auto * q = rotate("q_proj"), * k = rotate("k_proj");
        auto * scores = mm(k, q);
        scores = ggml_soft_max(ctx, ggml_scale(ctx, scores, 1.0f / std::sqrt(float(d))));
        auto * v = ggml_reshape_3d(ctx, linear(p + ".v_proj", x, false), d, 24, frames);
        v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [time,head_dim,heads]
        auto * y = mm(v, scores);
        y = ggml_cont(ctx, ggml_permute(ctx, y, 0, 2, 1, 3));
        return linear(p + ".o_proj", ggml_reshape_2d(ctx, y, d * 24, frames), false);
    }
    ggml_tensor * group(ggml_tensor * x, int index) {
        const int d = index ? 3072 : 1536;
        const std::string suffix = index ? "_2" : "";
        const auto ada = "estimator.adaln_single" + suffix;
        auto * embedded = linear(ada + ".emb.timestep_embedder.linear_1", time_embedding);
        embedded = linear(ada + ".emb.timestep_embedder.linear_2", ggml_silu(ctx, embedded));
        auto * mod = linear(ada + ".linear", ggml_silu(ctx, embedded));
        mod = ggml_reshape_2d(ctx, mod, d, 6);
        for (int i = 0; i < (index ? 6 : 24); ++i) {
            const auto p = "estimator.transformer_blocks" + suffix + "." + std::to_string(i);
            auto * table = ggml_add(ctx, model.tensor(p + ".scale_shift_table"), mod);
            auto * norm = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), model.tensor(p + ".attn_norm.weight"));
            norm = modulate(norm, row(table, d, 0), row(table, d, 1));
            x = ggml_add(ctx, x, ggml_mul(ctx, attention(p + ".attn", norm, index ? 128 : 64), row(table, d, 2)));
            norm = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), model.tensor(p + ".mlp_norm.weight"));
            norm = modulate(norm, row(table, d, 3), row(table, d, 4));
            auto * gate = ggml_silu(ctx, linear(p + ".mlp.gate", norm, false));
            auto * mlp = linear(p + ".mlp.down", ggml_mul(ctx, gate, linear(p + ".mlp.up", norm, false)), false);
            x = ggml_add(ctx, x, ggml_mul(ctx, mlp, row(table, d, 5)));
            if (i == 0 || i == (index ? 5 : 23)) x = keep(x, "stage.transformer_blocks" + suffix + "." + std::to_string(i));
        }
        auto * table = ggml_add(ctx, model.tensor("estimator.scale_shift_table" + suffix), embedded);
        return modulate(ggml_norm(ctx, x, 1e-6f), row(table, d, 0), row(table, d, 1));
    }
    std::vector<float> evaluate(const float * values, float time, const char * dump = nullptr) {
        finite(values, size_t(ggml_nelements(input)));
        float embedding[512];
        for (int i = 0; i < 256; ++i) {
            const float freq = std::exp((-std::log(10000.0f) * float(i)) / 256.0f);
            const float angle = (time * freq) * 1000.0f;
            embedding[i] = std::cos(angle); embedding[i + 256] = std::sin(angle);
        }
        ggml_backend_tensor_set(input, values, 0, ggml_nbytes(input));
        ggml_backend_tensor_set(time_embedding, embedding, 0, sizeof(embedding));
        // Gallocr may reuse an input after its final consumer. Restore every
        // input on every invocation, including unchanged positions/constants.
        ggml_backend_tensor_set(positions, position_values.data(), 0, ggml_nbytes(positions));
        float value = 1; ggml_backend_tensor_set(one, &value, 0, sizeof(value));
        return run(dump);
    }
};
}

std::vector<int32_t> token_data(const heartcodec_tokens_v1 * t) {
    if (!t || t->size < sizeof(*t) || !t->data || t->frames <= 0 || t->codebook_stride <= 0 || t->frame_stride <= 0 ||
        uint64_t(t->frames) > std::numeric_limits<size_t>::max() / (3840 * 2 * sizeof(float)))
        throw std::invalid_argument("Invalid RVQ token view");
    const uint64_t limit = uint64_t(std::numeric_limits<ptrdiff_t>::max()) / sizeof(int32_t);
    if (uint64_t(t->codebook_stride) > limit / 7 || uint64_t(t->frame_stride) > (limit - uint64_t(t->codebook_stride) * 7) / uint64_t(t->frames))
        throw std::invalid_argument("Overflowing token strides");
    std::vector<int32_t> result(size_t(t->frames) * 8);
    for (int q = 0; q < 8; ++q) for (int64_t f = 0; f < t->frames; ++f) {
        const int code = t->data[q * t->codebook_stride + f * t->frame_stride];
        if (code < 0 || code >= 8192) throw std::invalid_argument("RVQ code must be in [0,8191]");
        result[size_t(q * t->frames + f)] = code;
    }
    return result;
}

std::vector<float> rvq_decode(Model & model, const int32_t * codes, int64_t frames, bool condition) {
    Graph g(model);
    auto * input = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, frames, 8); ggml_set_input(input);
    ggml_tensor * sum = nullptr;
    for (int i = 0; i < 8; ++i) {
        auto * indices = ggml_view_1d(g.ctx, input, frames, size_t(i * frames) * sizeof(int32_t));
        auto * embed = ggml_get_rows(g.ctx, model.tensor("vq_embed.layers." + std::to_string(i) + "._codebook.embed"), indices);
        sum = sum ? ggml_add(g.ctx, sum, embed) : embed;
    }
    auto * output = g.linear("vq_embed.project_out", sum);
    if (condition) output = g.linear("cond_feature_emb", output);
    g.allocate(output);
    ggml_backend_tensor_set(input, codes, 0, ggml_nbytes(input));
    return g.run();
}

std::vector<float> flow_step(Model & model, const float * input, int64_t frames, float time, const char * dump) {
    if (frames <= 0 || frames > INT32_MAX || !std::isfinite(time) || time < 0 || time > 1)
        throw std::invalid_argument("Invalid flow step shape or time");
    Estimator estimator(model, frames, dump && *dump);
    return estimator.evaluate(input, time, dump);
}

void validate_flow_config(const heartcodec_flow_config_v1 * c, int64_t frames) {
    if (!c || c->size < sizeof(*c) || c->steps < 1 || c->steps > 1000 || !std::isfinite(c->guidance) || c->guidance < 0 ||
        c->segment_codes < 93 || c->segment_codes % 93 || c->segment_codes > INT32_MAX / 2 ||
        c->prefix_frames < 0 || c->prefix_frames > 2 * frames || (c->prefix_frames && !c->context))
        throw std::invalid_argument("Invalid flow configuration");
    if (c->schedule) {
        finite(c->schedule, size_t(c->steps) + 1);
        if (c->schedule[0] != 0 || c->schedule[c->steps] != 1)
            throw std::invalid_argument("Schedule endpoints must be 0 and 1");
        for (int i = 0; i < c->steps; ++i) if (c->schedule[i] >= c->schedule[i + 1])
            throw std::invalid_argument("Schedule must be strictly increasing");
    }
}

std::vector<float> flow_decode(Model & model, const int32_t * codes, int64_t frames,
                               const heartcodec_flow_config_v1 & c, const char * dump) {
    validate_flow_config(&c, frames);
    if (c.noise_segments < 0 || c.noise_segments > 1 || (!c.noise && c.noise_segments))
        throw std::invalid_argument("Flow decode accepts at most one explicit noise segment");
    if (frames > INT32_MAX / 2) throw std::invalid_argument("Too many flow frames");
    const int64_t n = frames * 2;
    const size_t count = size_t(n) * 256;
    const bool dumping = dump && *dump;
    if (dumping) std::filesystem::create_directories(dump);
    const auto condition = rvq_decode(model, codes, frames, true);
    std::vector<float> mu(size_t(n) * 512);
    for (int64_t i = 0; i < n; ++i) std::copy_n(condition.data() + (i / 2) * 512, 512, mu.data() + i * 512);
    std::vector<float> x(count), context(count, 0);
    if (c.noise) { finite(c.noise, count); std::copy_n(c.noise, count, x.data()); }
    else {
        std::mt19937_64 rng(c.seed);
        // Explicit Box-Muller avoids standard-library-specific normal_distribution.
        const double factor = 1.0 / 9007199254740992.0;
        for (size_t i = 0; i < count; i += 2) {
            double u = (double(rng() >> 11) + 0.5) * factor;
            double angle = double(rng() >> 11) * factor * 6.2831853071795864769;
            double radius = std::sqrt(-2.0 * std::log(u));
            x[i] = float(radius * std::cos(angle)); x[i + 1] = float(radius * std::sin(angle));
        }
    }
    const size_t prefix = size_t(c.prefix_frames) * 256;
    if (prefix) { finite(c.context, prefix); std::copy_n(c.context, prefix, context.data()); }
    const auto noise = x;
    std::vector<float> input(size_t(n) * 1024), uncond;
    Estimator estimator(model, n);
    if (dumping) { dump_values(std::filesystem::path(dump) / "mu.f32", mu); dump_values(std::filesystem::path(dump) / "euler.0.f32", x); }
    float time = 0;
    for (int step = 0; step < c.steps; ++step) {
        const float next = c.schedule ? c.schedule[step + 1] : float(step + 1) / float(c.steps);
        const float dt = next - time;
        const float blend = 1.0f - (1.0f - 1e-6f) * time;
        for (size_t i = 0; i < prefix; ++i) x[i] = blend * noise[i] + time * context[i];
        for (int64_t i = 0; i < n; ++i) {
            std::copy_n(x.data() + i * 256, 256, input.data() + i * 1024);
            std::copy_n(context.data() + i * 256, 256, input.data() + i * 1024 + 256);
            std::fill_n(input.data() + i * 1024 + 512, 512, 0.0f);
        }
        // Execute CFG branches sequentially to bound graph memory on 8GB GPUs.
        if (c.guidance > 1) uncond = estimator.evaluate(input.data(), time);
        for (int64_t i = 0; i < n; ++i) std::copy_n(mu.data() + i * 512, 512, input.data() + i * 1024 + 512);
        auto velocity = estimator.evaluate(input.data(), time);
        if (dumping) {
            dump_values(std::filesystem::path(dump) / ("velocity.cond." + std::to_string(step) + ".f32"), velocity);
            if (c.guidance > 1) dump_values(std::filesystem::path(dump) / ("velocity.uncond." + std::to_string(step) + ".f32"), uncond);
        }
        for (size_t i = 0; i < count; ++i) {
            float v = c.guidance > 1 ? uncond[i] + c.guidance * (velocity[i] - uncond[i]) : velocity[i];
            x[i] = x[i] + dt * v;
        }
        finite(x.data(), x.size());
        if (dumping) dump_values(std::filesystem::path(dump) / ("euler." + std::to_string(step + 1) + ".f32"), x);
        time = time + dt;
    }
    std::copy_n(context.data(), prefix, x.data());
    return x;
}
}
