#include "convolution.hpp"

#include <algorithm>

void convolution_cpu(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    std::vector<float>& output,
    const ConvConfig& config
) {
    const int output_height = config.output_height();
    const int output_width = config.output_width();
    output.assign(config.output_elements(), 0.0F);

    for (int filter = 0; filter < kOutputChannels; ++filter) {
        for (int output_y = 0; output_y < output_height; ++output_y) {
            for (int output_x = 0; output_x < output_width; ++output_x) {
                float sum = 0.0F;
                for (int channel = 0; channel < kInputChannels; ++channel) {
                    for (int kernel_y = 0; kernel_y < kKernelSize; ++kernel_y) {
                        const int input_y =
                            output_y * config.stride + kernel_y - config.padding;
                        if (input_y < 0 || input_y >= config.height) {
                            continue;
                        }

                        for (int kernel_x = 0; kernel_x < kKernelSize; ++kernel_x) {
                            const int input_x =
                                output_x * config.stride + kernel_x - config.padding;
                            if (input_x < 0 || input_x >= config.width) {
                                continue;
                            }

                            const std::size_t input_index =
                                (static_cast<std::size_t>(channel) * config.height +
                                 input_y) *
                                    config.width +
                                input_x;
                            const std::size_t weight_index =
                                ((static_cast<std::size_t>(filter) *
                                      kInputChannels +
                                  channel) *
                                     kKernelSize +
                                 kernel_y) *
                                    kKernelSize +
                                kernel_x;
                            sum += input[input_index] * weights[weight_index];
                        }
                    }
                }

                const std::size_t output_index =
                    (static_cast<std::size_t>(filter) * output_height + output_y) *
                        output_width +
                    output_x;
                output[output_index] = sum;
            }
        }
    }
}
