#include "heartcodec.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> static std::vector<T> read(const std::string & path) {
    const auto bytes = std::filesystem::file_size(path);
    if (!bytes || bytes % sizeof(T)) throw std::runtime_error("Invalid input file: " + path);
    std::vector<T> data(size_t(bytes / sizeof(T)));
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char *>(data.data()), bytes)) throw std::runtime_error("Cannot read: " + path);
    return data;
}
int main(int argc, char ** argv) {
    heartcodec_context * c = nullptr;
    heartcodec_tensor_v1 tensor{}; tensor.size = sizeof(tensor);
    heartcodec_stereo_v1 stereo{}; stereo.size = sizeof(stereo);
    try {
        const std::string op = argc > 1 ? argv[1] : "";
        if (op != "rvq" && op != "step" && op != "flow" && op != "decode")
            throw std::runtime_error("Usage: heartcodec-flow rvq|step|flow|decode --model FLOW.gguf --input FILE --output FILE [--sq-model SQ.gguf --backend cuda --steps 10 --guidance 1.25 --seed 1701 --segment-codes 372 --noise FILE --context FILE --prefix 0 --schedule FILE --time 0.375 --dump-dir DIR --stats FILE]");
        std::map<std::string, std::string> a;
        for (int i = 2; i < argc; i += 2) {
            if (i + 1 >= argc || !a.emplace(argv[i], argv[i + 1]).second) throw std::runtime_error("Invalid/duplicate option");
        }
        for (const auto & pair : a) {
            const auto & key = pair.first;
            if (key != "--model" && key != "--input" && key != "--output" && key != "--sq-model" && key != "--backend" && key != "--steps" &&
                key != "--guidance" && key != "--seed" && key != "--segment-codes" && key != "--noise" && key != "--context" && key != "--prefix" &&
                key != "--schedule" && key != "--time" && key != "--dump-dir" && key != "--stats") throw std::runtime_error("Unknown option: " + key);
        }
        for (const char * key : {"--model", "--input", "--output"}) if (!a.count(key)) throw std::runtime_error(std::string("Missing ") + key);
        heartcodec_config_v1 cfg{}; cfg.size = sizeof(cfg); heartcodec_config_init_v1(&cfg);
        cfg.model_path = a.at("--model").c_str();
        if (a.count("--backend")) cfg.backend = a.at("--backend").c_str();
        heartcodec_error_v1 e{}; e.size = sizeof(e);
        if (heartcodec_context_create_v1(&cfg, &c, &e)) throw std::runtime_error(e.message);
        heartcodec_flow_config_v1 fc{}; fc.size = sizeof(fc); heartcodec_flow_config_init_v1(&fc);
        if (a.count("--steps")) fc.steps = std::stoi(a.at("--steps"));
        if (a.count("--guidance")) fc.guidance = std::stof(a.at("--guidance"));
        if (a.count("--seed")) fc.seed = std::stoull(a.at("--seed"));
        if (a.count("--segment-codes")) fc.segment_codes = std::stoll(a.at("--segment-codes"));
        if (a.count("--prefix")) fc.prefix_frames = std::stoll(a.at("--prefix"));
        std::vector<float> noise, context, schedule;
        const char * dump = a.count("--dump-dir") ? a.at("--dump-dir").c_str() : nullptr;
        const auto started = std::chrono::steady_clock::now();
        heartcodec_status_v1 status;
        if (op == "step") {
            const auto input = read<float>(a.at("--input"));
            if (input.size() % 1024) throw std::runtime_error("Step input must be [frames,1024]");
            status = heartcodec_flow_step_v1(c, input.data(), int64_t(input.size() / 1024), a.count("--time") ? std::stof(a.at("--time")) : 0.375f, dump, &tensor, &e);
        } else {
            const auto input = read<int32_t>(a.at("--input"));
            if (input.size() % 8) throw std::runtime_error("Codes must be [8,frames] int32");
            heartcodec_tokens_v1 tokens{sizeof(tokens), input.data(), int64_t(input.size() / 8), int64_t(input.size() / 8), 1};
            const auto count = size_t(tokens.frames) * 2 * 256;
            if (a.count("--noise")) {
                noise = read<float>(a.at("--noise"));
                const size_t per_segment = op == "decode" ? size_t(fc.segment_codes) * 2 * 256 : count;
                if (!per_segment || noise.size() % per_segment || (op != "decode" && noise.size() != count)) throw std::runtime_error("Noise shape mismatch");
                fc.noise = noise.data(); fc.noise_segments = int64_t(noise.size() / per_segment);
            }
            if (a.count("--context")) { context = read<float>(a.at("--context")); if (context.size() != count) throw std::runtime_error("Context shape mismatch"); fc.context = context.data(); }
            if (a.count("--schedule")) { schedule = read<float>(a.at("--schedule")); if (schedule.size() != size_t(fc.steps) + 1) throw std::runtime_error("Schedule shape mismatch"); fc.schedule = schedule.data(); }
            if (op == "rvq") status = heartcodec_rvq_decode_v1(c, &tokens, &tensor, &e);
            else if (op == "flow") status = heartcodec_flow_decode_v1(c, &tokens, &fc, dump, &tensor, &e);
            else {
                if (!a.count("--sq-model")) throw std::runtime_error("Missing --sq-model");
                if (heartcodec_component_set_v1(c, HEARTCODEC_SQ_V1, a.at("--sq-model").c_str(), &e)) throw std::runtime_error(e.message);
                status = heartcodec_tokens_decode_v1(c, &tokens, &fc, &stereo, &e);
            }
        }
        if (status) throw std::runtime_error(e.message);
        std::ofstream out(a.at("--output"), std::ios::binary);
        const auto count = stereo.samples ? size_t(stereo.num_samples) * 2 : size_t(tensor.frames) * tensor.channels;
        out.write(reinterpret_cast<const char *>(stereo.samples ? stereo.samples : tensor.data), count * sizeof(float));
        if (!out) throw std::runtime_error("Cannot write output");
        heartcodec_memory_v1 memory{}; memory.size = sizeof(memory); heartcodec_get_memory_v1(c, &memory, nullptr);
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        if (a.count("--stats")) {
            std::ofstream stats(a.at("--stats"));
            stats << "{\"elapsed_seconds\":" << seconds << ",\"weight_buffer_bytes\":" << memory.weight_buffer_bytes << ",\"graph_buffer_bytes\":" << memory.graph_buffer_bytes << "}\n";
        }
        std::cout << op << ": " << seconds << "s, weights " << memory.weight_buffer_bytes << ", graph " << memory.graph_buffer_bytes << " bytes\n";
        heartcodec_tensor_free_v1(&tensor); heartcodec_stereo_free_v1(&stereo); heartcodec_context_destroy_v1(c); return 0;
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << '\n'; heartcodec_tensor_free_v1(&tensor); heartcodec_stereo_free_v1(&stereo); heartcodec_context_destroy_v1(c); return 1;
    }
}
