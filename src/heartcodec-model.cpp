#include "heartcodec-internal.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>

namespace heartcodec {
Model::~Model() {
    if (buffer) ggml_backend_buffer_free(buffer);
    if (weights) ggml_free(weights);
    if (gguf) gguf_free(gguf);
    if (backend) ggml_backend_free(backend);
}
ggml_tensor * Model::tensor(const std::string & name) const {
    auto * result = ggml_get_tensor(weights, name.c_str());
    if (!result) throw std::runtime_error("Missing tensor: " + name);
    return result;
}

void Model::load(const char * path, const char * requested, int threads) {
    static std::once_flag loaded;
    std::call_once(loaded, [] { ggml_backend_load_all(); });
    std::string wanted(requested);
    std::transform(wanted.begin(), wanted.end(), wanted.begin(), [](unsigned char c) { return std::tolower(c); });
    if (wanted != "cpu" && wanted != "cuda" && wanted != "vulkan" && wanted != "metal")
        throw std::invalid_argument("Backend must be cpu, cuda, vulkan, or metal");
    ggml_backend_dev_t chosen = nullptr;
    size_t best_memory = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        std::string name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
        if (name != wanted) continue;
        size_t free_bytes, total;
        ggml_backend_dev_memory(dev, &free_bytes, &total);
        if (!chosen || total > best_memory) { chosen = dev; best_memory = total; }
    }
    if (!chosen || !(backend = ggml_backend_dev_init(chosen, nullptr)))
        throw std::runtime_error("Requested backend unavailable: " + wanted);
    if (wanted == "cpu") {
        auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(chosen), "ggml_backend_set_n_threads"));
        if (set_threads) set_threads(backend, threads);
    }
    gguf_init_params params{true, &weights};
    gguf = gguf_init_from_file(path, params);
    if (!gguf || !weights) throw std::runtime_error("Cannot read GGUF model");
    auto string_meta = [&](const char * key, const char * value) {
        const int id = gguf_find_key(gguf, key);
        if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING ||
            std::string(gguf_get_val_str(gguf, id)) != value)
            throw std::runtime_error(std::string("Unsupported GGUF metadata: ") + key);
    };
    auto uint_meta = [&](const char * key, uint32_t value) {
        const int id = gguf_find_key(gguf, key);
        if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_UINT32 || gguf_get_val_u32(gguf, id) != value)
            throw std::runtime_error(std::string("Unsupported GGUF metadata: ") + key);
    };
    string_meta("general.architecture", "heartcodec-sq");
    const int part_id = gguf_find_key(gguf, "heartcodec.sq.part");
    if (part_id < 0 || gguf_get_kv_type(gguf, part_id) != GGUF_TYPE_STRING)
        throw std::runtime_error("Missing SQ component metadata");
    part = gguf_get_val_str(gguf, part_id);
    if (part != "encoder" && part != "decoder" && part != "codec")
        throw std::runtime_error("Unsupported SQ component");
    const int version_id = gguf_find_key(gguf, "heartcodec.sq.format_version");
    if (version_id < 0 || gguf_get_kv_type(gguf, version_id) != GGUF_TYPE_UINT32)
        throw std::runtime_error("Missing SQ format version");
    const auto version = gguf_get_val_u32(gguf, version_id);
    if (version == 2) { string_meta("heartcodec.sq.weight_layout", "matmul"); matrix_layout = true; }
    else if (version != 1 || part != "decoder") throw std::runtime_error("Unsupported SQ format version");
    uint_meta("heartcodec.sq.sample_rate", 48000);
    uint_meta("heartcodec.sq.latent_dim", 128);
    uint_meta("heartcodec.sq.samples_per_frame", 1920);
    validate(); // Reject shape errors before GGML graph assertions can fire.
    buffer = ggml_backend_alloc_ctx_tensors(weights, backend);
    if (!buffer) throw std::runtime_error("Cannot allocate model weights");
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open model payload");
    const auto length = file.tellg();
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        auto * t = tensor(gguf_get_tensor_name(gguf, i));
        const uint64_t offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, i);
        const size_t bytes = ggml_nbytes(t);
        if (length < 0 || offset > uint64_t(length) || bytes > uint64_t(length) - offset)
            throw std::runtime_error("Truncated model tensor: " + std::string(t->name));
        std::vector<uint8_t> data(bytes);
        file.seekg(static_cast<std::streamoff>(offset));
        if (!file.read(reinterpret_cast<char *>(data.data()), bytes))
            throw std::runtime_error("Cannot read model tensor: " + std::string(t->name));
        if (t->type == GGML_TYPE_F32) {
            std::vector<float> values(bytes / sizeof(float));
            std::memcpy(values.data(), data.data(), bytes);
            if (!std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }))
                throw std::runtime_error("Nonfinite model tensor: " + std::string(t->name));
            if (bytes == sizeof(float)) slopes[t->name] = values[0];
        }
        ggml_backend_tensor_set(t, data.data(), 0, bytes);
    }
}

void Model::validate() {
    std::set<std::string> expected;
    auto shape = [&](const std::string & name, std::initializer_list<int64_t> dims, bool weight = false) {
        auto * t = tensor(name);
        const bool supported_weight = matrix_layout && weight &&
            (t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q4_K ||
             t->type == GGML_TYPE_Q5_K || t->type == GGML_TYPE_Q6_K);
        if (t->type != GGML_TYPE_F32 && !supported_weight) throw std::runtime_error("Unsupported tensor type: " + name);
        int i = 0;
        for (auto dim : dims) if (t->ne[i++] != dim) throw std::runtime_error("Wrong shape: " + name);
        for (; i < GGML_MAX_DIMS; ++i) if (t->ne[i] != 1) throw std::runtime_error("Wrong rank: " + name);
        expected.insert(name);
    };
    auto conv = [&](const std::string & p, int k, int in, int out, bool transpose = false) {
        convolutions[p] = {k, in, out};
        if (transpose) shape(p + ".weight", {in, k * out}, true);
        else if (matrix_layout) shape(p + ".weight", {k * in, out}, true);
        else shape(p + ".weight", {k, in, out}, true);
        shape(p + ".bias", {out});
    };
    if (part != "encoder") {
    conv("sq.decoder.0", 5, 128, 2048);
    const int strides[] = {5, 4, 4, 4, 3};
    int channels = 2048;
    for (int stage = 1; stage <= 5; ++stage) {
        const std::string p = "sq.decoder." + std::to_string(stage);
        conv(p + ".up_conv.layer", 2 * strides[stage - 1], channels, channels / 2, true);
        channels /= 2;
        for (int unit = 0; unit < 5; ++unit) {
            const std::string r = p + ".convs." + std::to_string(unit);
            conv(r + ".conv1", 7, channels, channels);
            conv(r + ".conv2", 1, channels, channels);
            shape(r + ".activation1.weight", {1});
            shape(r + ".activation2.weight", {1});
        }
    }
    conv("sq.decoder.6.conv", 7, 64, 64);
    shape("sq.decoder.6.activation.weight", {1});
    conv("sq.decoder.7", 7, 64, 1);
    }
    if (part != "decoder") {
        conv("sq.encoder.0", 7, 1, 64);
        conv("sq.encoder.1.conv", 7, 64, 64);
        shape("sq.encoder.1.activation.weight", {1});
        const int strides[] = {3, 4, 4, 4, 5};
        int channels = 64;
        for (int stage = 2; stage <= 6; ++stage) {
            const std::string p = "sq.encoder." + std::to_string(stage);
            for (int unit = 0; unit < 5; ++unit) {
                const std::string r = p + ".convs." + std::to_string(unit);
                conv(r + ".conv1", 7, channels, channels);
                conv(r + ".conv2", 1, channels, channels);
                shape(r + ".activation1.weight", {1});
                shape(r + ".activation2.weight", {1});
            }
            conv(p + ".down_conv.layer", 2 * strides[stage - 2], channels, channels * 2);
            shape(p + ".down_conv.activation.weight", {1});
            channels *= 2;
        }
        conv("sq.encoder.7", 7, 2048, 128);
    }
    if (expected.size() != size_t(gguf_get_n_tensors(gguf)))
        throw std::runtime_error("Unexpected tensors in SQ decoder GGUF");
}
}
