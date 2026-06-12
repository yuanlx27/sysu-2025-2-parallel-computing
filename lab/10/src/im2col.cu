#include "convolution.hpp"
#include "cuda_utils.cuh"

#include <cuda_runtime.h>

namespace {

__global__ void im2col_kernel(
    const float* input,
    float* columns,
    int height,
    int width,
    int output_height,
    int output_width,
    int stride,
    int padding
) {
    const int output_count = output_height * output_width;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total =
        kInputChannels * kKernelSize * kKernelSize * output_count;
    if (index >= total) {
        return;
    }

    const int output_index = index % output_count;
    const int kernel_index = index / output_count;
    const int output_y = output_index / output_width;
    const int output_x = output_index % output_width;
    const int kernel_x = kernel_index % kKernelSize;
    const int kernel_y = (kernel_index / kKernelSize) % kKernelSize;
    const int channel = kernel_index / (kKernelSize * kKernelSize);
    const int input_y = output_y * stride + kernel_y - padding;
    const int input_x = output_x * stride + kernel_x - padding;

    float value = 0.0F;
    if (input_y >= 0 && input_y < height &&
        input_x >= 0 && input_x < width) {
        const std::size_t input_index =
            (static_cast<std::size_t>(channel) * height + input_y) * width +
            input_x;
        value = input[input_index];
    }
    columns[index] = value;
}

}  // namespace

void launch_im2col(
    const float* input,
    float* columns,
    const ConvConfig& config,
    cudaStream_t stream
) {
    constexpr int block_size = 256;
    const std::size_t total = config.im2col_elements();
    const int grid_size =
        static_cast<int>((total + block_size - 1) / block_size);

    im2col_kernel<<<grid_size, block_size, 0, stream>>>(
        input,
        columns,
        config.height,
        config.width,
        config.output_height(),
        config.output_width(),
        config.stride,
        config.padding
    );
    CUDA_CHECK(cudaGetLastError());
}

void launch_im2col_convolution(
    const float* input,
    const float* weights,
    float* columns,
    float* output,
    const ConvConfig& config,
    cudaStream_t stream
) {
    launch_im2col(input, columns, config, stream);
    launch_tiled_gemm(
        weights,
        columns,
        output,
        kOutputChannels,
        kInputChannels * kKernelSize * kKernelSize,
        config.output_height() * config.output_width(),
        stream
    );
}
