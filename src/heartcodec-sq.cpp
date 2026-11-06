#include "heartcodec-internal.h"
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace heartcodec {
namespace {
struct Graph {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ~Graph() { if (allocator) ggml_gallocr_free(allocator); if (ctx) ggml_free(ctx); }
};

ggml_tensor * conv(ggml_context * ctx, Model & model, const std::string & p,
                   ggml_tensor * x, int dilation, bool causal, int stride = 1) {
    auto * w = model.tensor(p + ".weight");
    const auto spec = model.convolutions.at(p);
    const int k = spec.kernel;
    if (causal && k > 1) x = ggml_pad_ext(ctx, x, dilation * (k - 1), 0, 0, 0, 0, 0, 0, 0);
    // ggml_conv_1d uses F16 im2col by default. Request F32 columns explicitly
    // for the unquantized reference path; all work is standard shared-GGML ops.
    // IM2COL reads the kernel's shape, not its data. A small shape tensor keeps
    // matrix-packed/quantized weights out of an invalid sub-block reshape.
    auto * kernel_shape = model.matrix_layout ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, spec.in_channels, 1) : w;
    auto * cols = ggml_im2col(ctx, kernel_shape, x, stride, 0, causal ? 0 : k / 2, 0,
                            dilation, 0, false, GGML_TYPE_F32);
    cols = ggml_reshape_2d(ctx, cols, k * spec.in_channels, cols->ne[1]);
    auto * matrix = model.matrix_layout ? w : ggml_reshape_2d(ctx, w, k * spec.in_channels, spec.out_channels);
    auto * y = ggml_mul_mat(ctx, matrix, cols);
    if (!ggml_prec_set_acc(y, GGML_PREC_F32) ||
        (matrix->type == GGML_TYPE_F32 && !ggml_prec_set_src(y, GGML_PREC_F32, 1)))
        throw std::runtime_error("Cannot request SQ matmul precision");
    y = ggml_cont(ctx, ggml_transpose(ctx, y));
    return ggml_add(ctx, y, ggml_reshape_2d(ctx, model.tensor(p + ".bias"), 1, spec.out_channels));
}

ggml_tensor * prelu(ggml_context * ctx, Model & model, const std::string & p, ggml_tensor * x) {
    return ggml_leaky_relu(ctx, x, model.slopes.at(p + ".weight"), false);
}

ggml_tensor * upsample(ggml_context * ctx, Model & model, const std::string & p,
                      ggml_tensor * x, int stride, int channels) {
    auto * columns = ggml_mul_mat(ctx, model.tensor(p + ".weight"),
                                 ggml_cont(ctx, ggml_transpose(ctx, x)));
    if (!ggml_prec_set_acc(columns, GGML_PREC_F32) ||
        (columns->src[0]->type == GGML_TYPE_F32 && !ggml_prec_set_src(columns, GGML_PREC_F32, 1)))
        throw std::runtime_error("Cannot request SQ transpose matmul precision");
    auto * y = ggml_col2im_1d(ctx, columns, stride, channels, 0);
    // Upstream causal ConvTranspose1d crops stride samples only on the right.
    y = ggml_cont(ctx, ggml_view_2d(ctx, y, x->ne[0] * stride, channels, y->nb[1], 0));
    return ggml_add(ctx, y, ggml_reshape_2d(ctx, model.tensor(p + ".bias"), 1, channels));
}

void dump_tensor(const std::filesystem::path & path, ggml_tensor * tensor) {
    std::vector<float> data(size_t(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size() * sizeof(float));
    std::ofstream file(path, std::ios::binary);
    if (!file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float)))
        throw std::runtime_error("Cannot write stage dump: " + path.string());
}

std::vector<float> execute(Graph & graph, Model & model, ggml_tensor * input, const float * values,
                           ggml_tensor * output, const std::vector<ggml_tensor *> & stages, const char * dump) {
    ggml_set_output(output);
    auto * computation = ggml_new_graph_custom(graph.ctx, 4096, false);
    ggml_build_forward_expand(computation, output);
    for (int i = 0; i < ggml_graph_n_nodes(computation); ++i) {
        auto * node = ggml_graph_node(computation, i);
        if (!ggml_backend_supports_op(model.backend, node))
            throw std::runtime_error("Backend does not support operation: " + describe_op(node));
    }
    graph.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!graph.allocator || !ggml_gallocr_alloc_graph(graph.allocator, computation))
        throw std::runtime_error("Cannot allocate SQ graph");
    model.graph_bytes = ggml_gallocr_get_buffer_size(graph.allocator, 0);
    ggml_backend_tensor_set(input, values, 0, ggml_nbytes(input));
    model.checkpoint();
    if (ggml_backend_graph_compute(model.backend, computation) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("SQ graph execution failed");
    model.checkpoint();
    std::vector<float> data(size_t(ggml_nelements(output)));
    ggml_backend_tensor_get(output, data.data(), 0, data.size() * sizeof(float));
    if (dump && *dump) for (auto * stage : stages)
        dump_tensor(std::filesystem::path(dump) / (std::string(stage->name) + ".f32"), stage);
    return data;
}
}

std::vector<float> sq_decode(Model & model, const float * latents, int64_t frames, const char * dump) {
    if (model.part == "encoder") throw std::invalid_argument("Model does not contain an SQ decoder");
    const bool dumping = dump && *dump;
    if (dumping) std::filesystem::create_directories(dump);
    Graph graph;
    ggml_init_params init{32 * 1024 * 1024, nullptr, true};
    graph.ctx = ggml_init(init);
    if (!graph.ctx) throw std::runtime_error("Cannot allocate graph metadata");
    auto * ctx = graph.ctx;
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, frames, 128);
    ggml_set_input(input);
    ggml_set_name(input, "sq.quantized");
    std::vector<float> quantized(size_t(frames) * 128);
    if (heartcodec_sq_quantize_v1(latents, quantized.data(), quantized.size(), nullptr) != HEARTCODEC_OK_V1)
        throw std::invalid_argument("Latents must be finite, without overflow in 9*x");
    std::vector<ggml_tensor *> stages;
    auto keep_stage = [&](ggml_tensor * tensor) {
        ggml_set_name(tensor, ("decoder." + std::to_string(stages.size())).c_str());
        if (dumping) ggml_set_output(tensor);
        stages.push_back(tensor);
        return tensor;
    };
    auto * x = keep_stage(conv(ctx, model, "sq.decoder.0", input, 1, false));
    const int strides[] = {5, 4, 4, 4, 3};
    const int dilations[] = {1, 3, 5, 7, 9};
    int channels = 2048;
    for (int stage = 1; stage <= 5; ++stage) {
        const std::string p = "sq.decoder." + std::to_string(stage);
        channels /= 2;
        x = upsample(ctx, model, p + ".up_conv.layer", x, strides[stage - 1], channels);
        for (int unit = 0; unit < 5; ++unit) {
            const std::string r = p + ".convs." + std::to_string(unit);
            auto * residual = x;
            x = prelu(ctx, model, r + ".activation1", conv(ctx, model, r + ".conv1", x, dilations[unit], true));
            x = prelu(ctx, model, r + ".activation2", conv(ctx, model, r + ".conv2", x, 1, true));
            x = ggml_add(ctx, x, residual);
        }
        x = keep_stage(x);
    }
    x = ggml_reshape_3d(ctx, x, 1, x->ne[0], channels);
    x = ggml_repeat_4d(ctx, x, 2, x->ne[1], channels, 1);
    x = ggml_reshape_2d(ctx, x, x->ne[1] * 2, channels);
    x = keep_stage(prelu(ctx, model, "sq.decoder.6.activation",
                        conv(ctx, model, "sq.decoder.6.conv", x, 1, true)));
    x = keep_stage(conv(ctx, model, "sq.decoder.7", x, 1, true));
    if (x->ne[0] != frames * 1920 || x->ne[1] != 1)
        throw std::runtime_error("Unexpected SQ decoder output shape");
    return execute(graph, model, input, quantized.data(), x, stages, dump);
}

Encoded sq_encode(Model & model, const float * waveform, int64_t samples, const char * dump) {
    if (model.part == "decoder") throw std::invalid_argument("Model does not contain an SQ encoder");
    const bool dumping = dump && *dump;
    if (dumping) std::filesystem::create_directories(dump);
    Graph graph;
    graph.ctx = ggml_init({32 * 1024 * 1024, nullptr, true});
    if (!graph.ctx) throw std::runtime_error("Cannot allocate graph metadata");
    auto * ctx = graph.ctx;
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, samples, 1);
    ggml_set_input(input);
    std::vector<ggml_tensor *> stages;
    auto keep = [&](ggml_tensor * tensor) {
        // A retained debug view needs its own storage; its source's normal
        // compute lifetime may end before stage dumps are read back.
        if (dumping && tensor->view_src) tensor = ggml_cont(ctx, tensor);
        ggml_set_name(tensor, ("encoder." + std::to_string(stages.size())).c_str());
        if (dumping) ggml_set_output(tensor);
        stages.push_back(tensor);
        return tensor;
    };
    auto * x = keep(conv(ctx, model, "sq.encoder.0", input, 1, true));
    x = prelu(ctx, model, "sq.encoder.1.activation", conv(ctx, model, "sq.encoder.1.conv", x, 1, true));
    // Non-overlapping AvgPool1d(2): discard the odd tail, group adjacent
    // samples, and reduce. These standard GGML ops work on CPU and GPU.
    const int64_t pooled = x->ne[0] / 2, channels = x->ne[1];
    x = ggml_cont(ctx, ggml_view_2d(ctx, x, pooled * 2, channels, x->nb[1], 0));
    x = ggml_reshape_3d(ctx, x, 2, pooled, channels);
    x = keep(ggml_reshape_2d(ctx, ggml_mean(ctx, x), pooled, channels));
    const int strides[] = {3, 4, 4, 4, 5}, dilations[] = {1, 3, 5, 7, 9};
    for (int stage = 2; stage <= 6; ++stage) {
        const std::string p = "sq.encoder." + std::to_string(stage);
        for (int unit = 0; unit < 5; ++unit) {
            const auto r = p + ".convs." + std::to_string(unit);
            auto * residual = x;
            x = prelu(ctx, model, r + ".activation1", conv(ctx, model, r + ".conv1", x, dilations[unit], true));
            x = prelu(ctx, model, r + ".activation2", conv(ctx, model, r + ".conv2", x, 1, true));
            x = ggml_add(ctx, x, residual);
        }
        x = keep(prelu(ctx, model, p + ".down_conv.activation",
                       conv(ctx, model, p + ".down_conv.layer", x, 1, true, strides[stage - 2])));
    }
    x = keep(ggml_tanh(ctx, conv(ctx, model, "sq.encoder.7", x, 1, true)));
    return {execute(graph, model, input, waveform, x, stages, dump), x->ne[0]};
}
}
