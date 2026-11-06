#include "heartcodec-internal.h"
#include <cmath>
#include <numeric>

namespace heartcodec {
void validate_audio_encoder(Model & model) {
    std::set<std::string> expected;
    auto shape = [&](const std::string & name, std::initializer_list<int64_t> dims) {
        auto * t = model.tensor(name);
        if (t->type != GGML_TYPE_F32) throw std::runtime_error("Encoder requires F32 tensor: " + name);
        int i = 0;
        for (auto dim : dims) if (t->ne[i++] != dim) throw std::runtime_error("Wrong encoder shape: " + name);
        for (; i < GGML_MAX_DIMS; ++i) if (t->ne[i] != 1) throw std::runtime_error("Wrong encoder rank: " + name);
        expected.insert(name);
    };
    auto linear = [&](const std::string & p, int in, int out, bool bias = true) {
        shape(p + ".weight", {in, out}); if (bias) shape(p + ".bias", {out});
    };
    auto norm = [&](const std::string & p, int d) { shape(p + ".weight", {d}); shape(p + ".bias", {d}); };
    auto bn = [&](const std::string & p, int d) {
        norm(p, d); shape(p + ".running_mean", {d}); shape(p + ".running_var", {d});
    };
    auto conv = [&](const std::string & p, int in, int out, int kernel, bool bias = true) {
        shape(p + ".weight", {kernel, in, out}); if (bias) shape(p + ".bias", {out});
    };
    shape("query.cls_token", {512});
    for (const auto * p : {"acoustic", "semantic", "phone"}) linear(std::string("query.") + p, std::string(p) == "phone" ? 1536 : 2048, 512);
    linear("query.feature_proj", 3072, 512);
    for (int i = 0; i < 6; ++i) {
        const auto p = "query.blocks." + std::to_string(i);
        linear(p + ".attn.to_qkv", 512, 1536, false); linear(p + ".attn.to_out", 512, 512, false);
        norm(p + ".attn.q_norm", 128); norm(p + ".attn.k_norm", 128);
        shape(p + ".attn_scale.scale", {512}); shape(p + ".ff_scale.scale", {512});
        linear(p + ".ff.ff.0.proj", 512, 4096); linear(p + ".ff.ff.2", 2048, 512);
        shape(p + ".rope.inv_freq", {32});
    }
    linear("rvq.project_in", 512, 32);
    for (int i = 0; i < 8; ++i) shape("rvq.layers." + std::to_string(i) + "._codebook.embed", {32, 8192});
    conv("wh.conv1", 80, 768, 3); conv("wh.conv2", 768, 768, 3);
    shape("wh.embed_positions.weight", {768, 1500}); norm("wh.norm", 768);
    shape("wh.mel_filters", {80, 201}); shape("wh.window", {400});
    for (int i = 0; i < 12; ++i) {
        const auto p = "wh.layers." + std::to_string(i);
        for (const auto * proj : {"q_proj", "k_proj", "v_proj", "out_proj"})
            linear(p + ".attn." + proj, 768, 768, std::string(proj) != "k_proj");
        norm(p + ".attn_layer_norm", 768); norm(p + ".final_layer_norm", 768);
        linear(p + ".fc1", 768, 3072); linear(p + ".fc2", 3072, 768);
    }
    shape("wl.masked_spec_embed", {768});
    const int kernels[] = {10,3,3,3,3,2,2};
    for (int i = 0; i < 7; ++i) conv("wl.conv." + std::to_string(i) + ".conv", i ? 512 : 1, 512, kernels[i], false);
    norm("wl.conv.0.norm", 512); norm("wl.proj.norm", 512); linear("wl.proj.projection", 512, 768);
    conv("wl.enc.pos.conv", 48, 768, 128); norm("wl.enc.norm", 768);
    for (int i = 0; i < 12; ++i) {
        const auto p = "wl.enc.layers." + std::to_string(i);
        shape(p + ".attn.gru_rel_pos_const", {1,1,12,1});
        for (const auto * proj : {"q_proj", "k_proj", "v_proj", "out_proj"}) linear(p + ".attn." + proj, 768, 768);
        linear(p + ".attn.gru_rel_pos_linear", 64, 8);
        if (!i) shape(p + ".attn.rel_attn_embed.weight", {12,320});
        norm(p + ".norm", 768); norm(p + ".final_layer_norm", 768);
        linear(p + ".ffn.up", 768, 3072); linear(p + ".ffn.down", 3072, 768);
    }
    shape("mu.mel.mel_stft.spectrogram.window", {2048});
    shape("mu.mel.mel_stft.mel_scale.fb", {128,1025});
    for (int i = 0; i < 2; ++i) {
        const auto p = "mu.conv.conv." + std::to_string(i);
        for (const auto * name : {"conv1", "conv2", "conv3"}) {
            const int in = i || std::string(name) == "conv2" ? 512 : 1;
            shape(p + "." + name + ".weight", {3,3,in,512}); shape(p + "." + name + ".bias", {512});
        }
        for (const auto * name : {"bn1", "bn2", "bn3"}) bn(p + "." + name, 512);
    }
    linear("mu.conv.linear", 16384, 1024);
    shape("mu.enc.embed_positions.inv_freq", {32}); conv("mu.enc.pos.conv", 64, 1024, 128); norm("mu.enc.norm", 1024);
    for (int i = 0; i < 12; ++i) {
        const auto p = "mu.enc.layers." + std::to_string(i);
        for (const auto * name : {"ffn1", "ffn2"}) {
            norm(p + "." + name + "_layer_norm", 1024);
            linear(p + "." + name + ".up", 1024, 4096); linear(p + "." + name + ".down", 4096, 1024);
        }
        norm(p + ".attn_layer_norm", 1024);
        for (const auto * name : {"linear_q", "linear_k", "linear_v", "linear_out"}) linear(p + ".attn." + name, 1024, 1024);
        norm(p + ".conv.norm", 1024);
        conv(p + ".conv.pointwise_conv1", 1024, 2048, 1, false);
        conv(p + ".conv.depthwise_conv", 1, 1024, 31, false);
        bn(p + ".conv.batch_norm", 1024); conv(p + ".conv.pointwise_conv2", 1024, 1024, 1, false);
        norm(p + ".final_layer_norm", 1024);
    }
    shape("rs16.kernel", {23,1,2});
    for (int rate : {16000,22050,32000,44100,48000,88200,96000,192000}) {
        const int gcd = std::gcd(rate,24000), orig = rate/gcd, next = 24000/gcd;
        const int width = int(std::ceil(6.0 * orig / (std::min(orig,next) * .99)));
        shape("rsin." + std::to_string(rate) + ".kernel", {2*width+orig,1,next});
    }
    if (expected.size() != size_t(gguf_get_n_tensors(model.gguf))) throw std::runtime_error("Unexpected encoder GGUF tensors");
}
}
