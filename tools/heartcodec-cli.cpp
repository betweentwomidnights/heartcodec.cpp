#include "heartcodec.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static void write_f32(const std::filesystem::path & path, const float * data, size_t count) {
    std::ofstream file(path, std::ios::binary);
    if (!file.write(reinterpret_cast<const char *>(data), count * sizeof(float)))
        throw std::runtime_error("Cannot write " + path.string());
}

int main(int argc, char ** argv) {
    heartcodec_context * context = nullptr;
    heartcodec_audio_v1 audio{}; audio.size = sizeof(audio); heartcodec_audio_init_v1(&audio);
    heartcodec_sq_latents_v1 latent{}; latent.size = sizeof(latent); heartcodec_sq_latents_init_v1(&latent);
    try {
        const std::string operation = argc > 1 ? argv[1] : "";
        if (operation != "sq-decode" && operation != "sq-encode" && operation != "sq-roundtrip")
            throw std::runtime_error("Usage: heartcodec-cli sq-encode|sq-decode|sq-roundtrip --model MODEL.gguf --input IN.f32 --output OUT.f32 [--backend cpu|cuda|vulkan|metal] [--threads 4] [--dump-dir DIR] [--stats FILE.json]");
        std::map<std::string, std::string> args;
        for (int i = 2; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::runtime_error("Missing option value");
            const std::string key(argv[i]);
            if (key != "--model" && key != "--input" && key != "--output" && key != "--backend" && key != "--threads" && key != "--dump-dir" && key != "--stats")
                throw std::runtime_error("Unknown option: " + key);
            if (!args.emplace(key, argv[i + 1]).second) throw std::runtime_error("Duplicate option: " + key);
        }
        for (const auto * key : {"--model", "--input", "--output"})
            if (!args.count(key)) throw std::runtime_error(std::string("Missing ") + key);
        const auto size = std::filesystem::file_size(args.at("--input"));
        if (!size || size % sizeof(float) || (operation == "sq-decode" && size % (128 * sizeof(float))))
            throw std::runtime_error("Input must be little-endian F32: mono waveform or [128, frames] latents");
        std::vector<float> latents(size / sizeof(float));
        std::ifstream input(args.at("--input"), std::ios::binary);
        if (!input.read(reinterpret_cast<char *>(latents.data()), size)) throw std::runtime_error("Cannot read latents");
        heartcodec_config_v1 config{}; config.size = sizeof(config); heartcodec_config_init_v1(&config);
        config.model_path = args.at("--model").c_str();
        if (args.count("--backend")) config.backend = args.at("--backend").c_str();
        if (args.count("--threads")) {
            size_t consumed = 0;
            config.cpu_threads = std::stoi(args.at("--threads"), &consumed);
            if (consumed != args.at("--threads").size()) throw std::runtime_error("Invalid thread count");
        }
        heartcodec_error_v1 error{}; error.size = sizeof(error); heartcodec_error_init_v1(&error);
        if (heartcodec_context_create_v1(&config, &context, &error)) throw std::runtime_error(error.message);
        const auto start = std::chrono::steady_clock::now();
        const char * dump = args.count("--dump-dir") ? args.at("--dump-dir").c_str() : nullptr;
        heartcodec_memory_v1 memory{}; memory.size = sizeof(memory);
        uint64_t graph_peak = 0;
        if (operation != "sq-decode") {
            if (heartcodec_sq_encode_v1(context, latents.data(), int64_t(latents.size()), dump, &latent, &error))
                throw std::runtime_error(error.message);
            heartcodec_get_memory_v1(context, &memory, nullptr);
            graph_peak = memory.graph_buffer_bytes;
            if (dump) {
                const size_t count = size_t(latent.frames) * 128;
                write_f32(std::filesystem::path(dump) / "sq.latent.f32", latent.data, count);
                std::vector<float> quantized(count);
                if (heartcodec_sq_quantize_v1(latent.data, quantized.data(), count, &error)) throw std::runtime_error(error.message);
                write_f32(std::filesystem::path(dump) / "sq.quantized.f32", quantized.data(), count);
            }
        }
        if (operation != "sq-encode") {
            const auto * data = operation == "sq-decode" ? latents.data() : latent.data;
            const auto frames = operation == "sq-decode" ? int64_t(latents.size() / 128) : latent.frames;
            if (heartcodec_sq_decode_v1(context, data, frames, dump, &audio, &error)) throw std::runtime_error(error.message);
            write_f32(args.at("--output"), audio.samples, size_t(audio.num_samples));
        } else write_f32(args.at("--output"), latent.data, size_t(latent.frames) * 128);
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        heartcodec_get_memory_v1(context, &memory, nullptr);
        graph_peak = std::max(graph_peak, memory.graph_buffer_bytes);
        if (args.count("--stats")) {
            std::ofstream stats(args.at("--stats"));
            stats << "{\"elapsed_seconds\":" << seconds << ",\"weight_buffer_bytes\":" << memory.weight_buffer_bytes
                  << ",\"graph_buffer_bytes\":" << graph_peak << "}\n";
            if (!stats) throw std::runtime_error("Cannot write memory statistics");
        }
        std::cout << operation << ": " << seconds << "s, weights " << memory.weight_buffer_bytes
                  << " bytes, graph " << graph_peak << " bytes\n";
        heartcodec_sq_latents_free_v1(&latent); heartcodec_audio_free_v1(&audio); heartcodec_context_destroy_v1(context);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << "\n";
        heartcodec_sq_latents_free_v1(&latent); heartcodec_audio_free_v1(&audio); heartcodec_context_destroy_v1(context);
        return 1;
    }
}
