#include "heartcodec.h"
#include <cmath>
#include <iostream>
#include <limits>

int main() {
    // Halfway values catch the roundf/torch.round incompatibility, including negatives.
    const float input[] = {-1, -2.5f/9, -1.5f/9, -0.5f/9, 0, 0.5f/9, 1.5f/9, 2.5f/9, 1};
    const float expected[] = {-1, -2.0f/9, -2.0f/9, -0.0f, 0, 0, 2.0f/9, 2.0f/9, 1};
    float output[9];
    if (heartcodec_sq_quantize_v1(input, output, 9, nullptr)) return 1;
    for (int i = 0; i < 9; ++i) if (output[i] != expected[i]) {
        std::cerr << "Wrong quantizer value at " << i << "\n"; return 1;
    }
    if (!std::signbit(output[3])) return 1;
    const float invalid[] = {0, std::numeric_limits<float>::quiet_NaN()};
    float unchanged[] = {42, 42};
    if (heartcodec_sq_quantize_v1(invalid, unchanged, 2, nullptr) != HEARTCODEC_INVALID_ARGUMENT_V1 || unchanged[0] != 42) return 1;
    if (heartcodec_sq_quantize_v1(nullptr, nullptr, 1, nullptr) != HEARTCODEC_INVALID_ARGUMENT_V1) return 1;
    if (heartcodec_sq_quantize_v1(nullptr, nullptr, 0, nullptr)) return 1;
    heartcodec_context * context = nullptr;
    if (heartcodec_context_create_v1(nullptr, &context, nullptr) != HEARTCODEC_INVALID_ARGUMENT_V1 || context) return 1;
    return 0;
}
