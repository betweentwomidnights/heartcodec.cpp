// Model policy only. Quantization and execution use the shared GGML fork.
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Mix { ggml_type body, critical; };
static bool critical(const std::string & name) {
    return name.find(".conv2.weight") != std::string::npos ||
           name.find(".up_conv.layer.weight") != std::string::npos ||
           name.find(".down_conv.layer.weight") != std::string::npos;
}
static bool boundary(const std::string & name) {
    return name == "sq.encoder.0.weight" || name == "sq.encoder.7.weight" ||
           name == "sq.decoder.0.weight" || name == "sq.decoder.7.weight" ||
           name == "sq.encoder.1.conv.weight" || name == "sq.decoder.6.conv.weight";
}

int main(int argc, char ** argv) {
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::runtime_error("Missing option value");
            const std::string key = argv[i];
            if (key != "--input" && key != "--output" && key != "--mix" && key != "--scope")
                throw std::runtime_error("Unknown option: " + key);
            if (!args.emplace(key, argv[i + 1]).second) throw std::runtime_error("Duplicate option: " + key);
        }
        if (!args.count("--input") || !args.count("--output") || !args.count("--mix"))
            throw std::runtime_error("Usage: heartcodec-quantize --input sq-f32.gguf --output sq-quant.gguf --mix f16|q8_0|q6_k|q5_k_m|q4_k_m [--scope encoder|decoder|codec]");
        const std::string mix_name = args.at("--mix"), scope = args.count("--scope") ? args.at("--scope") : "codec";
        const std::map<std::string, Mix> mixes = {
            {"f16", {GGML_TYPE_F16, GGML_TYPE_F16}}, {"q8_0", {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0}},
            {"q6_k", {GGML_TYPE_Q6_K, GGML_TYPE_Q6_K}}, {"q5_k_m", {GGML_TYPE_Q5_K, GGML_TYPE_Q6_K}},
            {"q4_k_m", {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}}};
        if (!mixes.count(mix_name) || (scope != "encoder" && scope != "decoder" && scope != "codec"))
            throw std::runtime_error("Unsupported mix or scope");
        const auto mix = mixes.at(mix_name);
        if (std::filesystem::weakly_canonical(args.at("--input")) == std::filesystem::weakly_canonical(args.at("--output")))
            throw std::runtime_error("Input and output must be different files");
        using Gguf = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
        using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
        Gguf input(gguf_init_from_file(args.at("--input").c_str(), {true, nullptr}), gguf_free);
        if (!input) throw std::runtime_error("Cannot read source GGUF");
        auto check_string = [&](const char * key, const char * expected) {
            const auto id = gguf_find_key(input.get(), key);
            if (id < 0 || gguf_get_kv_type(input.get(), id) != GGUF_TYPE_STRING ||
                std::string(gguf_get_val_str(input.get(), id)) != expected)
                throw std::runtime_error(std::string("Unsupported model metadata: ") + key);
        };
        check_string("general.architecture", "heartcodec-sq");
        check_string("heartcodec.sq.weight_layout", "matmul");
        const auto version_id = gguf_find_key(input.get(), "heartcodec.sq.format_version");
        if (version_id < 0 || gguf_get_kv_type(input.get(), version_id) != GGUF_TYPE_UINT32 ||
            gguf_get_val_u32(input.get(), version_id) != 2) throw std::runtime_error("Expected SQ GGUF format 2");
        Gguf output(gguf_init_empty(), gguf_free);
        Context metadata(ggml_init({8 * 1024 * 1024, nullptr, true}), ggml_free);
        if (!output || !metadata) throw std::runtime_error("Cannot allocate quantizer metadata");
        gguf_set_kv(output.get(), input.get());
        gguf_set_val_str(output.get(), "heartcodec.sq.weight_precision", mix_name.c_str());
        gguf_set_val_str(output.get(), "heartcodec.sq.quantization_policy", "sq-conv-v1");
        gguf_set_val_str(output.get(), "heartcodec.sq.quantization_scope", scope.c_str());
        std::ifstream source(args.at("--input"), std::ios::binary | std::ios::ate);
        if (!source) throw std::runtime_error("Cannot open source model");
        const auto source_length = source.tellg();
        const int64_t count = gguf_get_n_tensors(input.get());
        std::vector<std::vector<uint8_t>> blobs(size_t(count), std::vector<uint8_t>{});
        std::map<std::string, size_t> counts, bytes_by_type;
        for (int64_t i = 0; i < count; ++i) {
            const std::string name = gguf_get_tensor_name(input.get(), i);
            const auto * ne = gguf_get_tensor_ne(input.get(), i);
            const int dims = gguf_get_tensor_ndims(input.get(), i);
            if (gguf_get_tensor_type(input.get(), i) != GGML_TYPE_F32)
                throw std::runtime_error("Quantize from the original F32 GGUF, not an already reduced model");
            ggml_type type = GGML_TYPE_F32;
            const bool selected = scope == "codec" || name.rfind("sq." + scope + ".", 0) == 0;
            if (selected && dims == 2 && name.size() >= 7 && name.compare(name.size() - 7, 7, ".weight") == 0) {
                const auto proposed = critical(name) ? mix.critical : mix.body;
                if ((mix_name == "f16" || !boundary(name)) && ne[0] % ggml_blck_size(proposed) == 0)
                    type = proposed;
            }
            auto * tensor = ggml_new_tensor(metadata.get(), type, dims, ne);
            ggml_set_name(tensor, name.c_str());
            const size_t input_bytes = gguf_get_tensor_size(input.get(), i);
            const uint64_t offset = gguf_get_data_offset(input.get()) + gguf_get_tensor_offset(input.get(), i);
            if (source_length < 0 || offset > uint64_t(source_length) || input_bytes > uint64_t(source_length) - offset)
                throw std::runtime_error("Truncated source tensor: " + name);
            std::vector<float> values(input_bytes / sizeof(float));
            source.seekg(static_cast<std::streamoff>(offset));
            if (!source.read(reinterpret_cast<char *>(values.data()), input_bytes)) throw std::runtime_error("Cannot read " + name);
            if (!std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); }))
                throw std::runtime_error("Nonfinite source tensor: " + name);
            auto & blob = blobs[size_t(i)];
            blob.resize(ggml_nbytes(tensor));
            if (type == GGML_TYPE_F32) std::memcpy(blob.data(), values.data(), input_bytes);
            else if (type == GGML_TYPE_F16)
                ggml_fp32_to_fp16_row(values.data(), reinterpret_cast<ggml_fp16_t *>(blob.data()), int64_t(values.size()));
            else {
                ggml_quantize_init(type);
                const auto written = ggml_quantize_chunk(type, values.data(), blob.data(), 0, ne[1], ne[0], nullptr);
                if (written != blob.size()) throw std::runtime_error("Unexpected quantized byte count: " + name);
            }
            tensor->data = blob.data();
            gguf_add_tensor(output.get(), tensor);
            counts[ggml_type_name(type)]++;
            bytes_by_type[ggml_type_name(type)] += blob.size();
        }
        if (!gguf_write_to_file(output.get(), args.at("--output").c_str(), false)) throw std::runtime_error("Cannot write quantized GGUF");
        std::ofstream report(args.at("--output") + ".quantization.json");
        report << "{\"mix\":\"" << mix_name << "\",\"scope\":\"" << scope << "\",\"policy\":\"sq-conv-v1\",\"file_bytes\":"
               << std::filesystem::file_size(args.at("--output")) << ",\"types\":{";
        bool first = true;
        for (const auto & item : counts) {
            if (!first) report << ',';
            first = false;
            report << '"' << item.first << "\":{\"tensors\":" << item.second << ",\"bytes\":" << bytes_by_type[item.first] << '}';
        }
        report << "}}\n";
        if (!report) throw std::runtime_error("Cannot write quantization report");
        ggml_quantize_free();
        std::cout << mix_name << " " << scope << ": " << std::filesystem::file_size(args.at("--output")) << " bytes\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
