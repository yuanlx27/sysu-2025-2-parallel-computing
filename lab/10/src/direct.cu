#include "convolution.hpp"
#include "cuda_utils.cuh"

#include <cuda_runtime.h>

namespace {

__global__ void direct_convolution_kernel(
    const float* input,
    const float* weights,
    float* output,
    int height,
    int width,
    int output_height,
    int output_width,
    int stride,
    int padding
) {
    const int output_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int filter = blockIdx.z;

    if (output_x >= output_width || output_y >= output_height) {
        return;
    }

    float sum = 0.0F;
    for (int channel = 0; channel < kInputChannels; ++channel) {
        for (int kernel_y = 0; kernel_y < kKernelSize; ++kernel_y) {
            const int input_y = output_y * stride + kernel_y - padding;
            if (input_y < 0 || input_y >= height) {
                continue;
            }

            for (int kernel_x = 0; kernel_x < kKernelSize; ++kernel_x) {
                const int input_x = output_x * stride + kernel_x - padding;
                if (input_x < 0 || input_x >= width) {
                    continue;
                }

                const std::size_t input_index =
                    (static_cast<std::size_t>(channel) * height + input_y) *
                        width +
                    input_x;
                const std::size_t weight_index =
                    ((static_cast<std::size_t>(filter) * kInputChannels +
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

__global__ void shared_convolution_kernel(
    const float* input,
    const float* weights,
    float* output,
    int height,
    int width,
    int output_height,
    int output_width,
    int stride,
    int padding,
    int tile_height,
    int tile_width
) {
    extern __shared__ float tile[];

    const int output_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int filter = blockIdx.z;
    const int tile_origin_x = blockIdx.x * blockDim.x * stride - padding;
    const int tile_origin_y = blockIdx.y * blockDim.y * stride - padding;
    const int linear_thread = threadIdx.y * blockDim.x + threadIdx.x;
    const int thread_count = blockDim.x * blockDim.y;
    const int tile_elements = tile_height * tile_width;

    float sum = 0.0F;
    for (int channel = 0; channel < kInputChannels; ++channel) {
        for (int index = linear_thread; index < tile_elements;
             index += thread_count) {
            const int tile_y = index / tile_width;
            const int tile_x = index % tile_width;
            const int input_y = tile_origin_y + tile_y;
            const int input_x = tile_origin_x + tile_x;

            float value = 0.0F;
            if (input_y >= 0 && input_y < height &&
                input_x >= 0 && input_x < width) {
                const std::size_t input_index =
                    (static_cast<std::size_t>(channel) * height + input_y) *
                        width +
                    input_x;
                value = input[input_index];
            }
            tile[index] = value;
        }
        __syncthreads();

        if (output_x < output_width && output_y < output_height) {
            const int local_y = threadIdx.y * stride;
            const int local_x = threadIdx.x * stride;
            for (int kernel_y = 0; kernel_y < kKernelSize; ++kernel_y) {
                for (int kernel_x = 0; kernel_x < kKernelSize; ++kernel_x) {
                    const std::size_t weight_index =
                        ((static_cast<std::size_t>(filter) * kInputChannels +
                          channel) *
                             kKernelSize +
                         kernel_y) *
                            kKernelSize +
                        kernel_x;
                    sum += tile[(local_y + kernel_y) * tile_width +
                                local_x + kernel_x] *
                           weights[weight_index];
                }
            }
        }
        __syncthreads();
    }

    if (output_x < output_width && output_y < output_height) {
        const std::size_t output_index =
            (static_cast<std::size_t>(filter) * output_height + output_y) *
                output_width +
            output_x;
        output[output_index] = sum;
    }
}

dim3 make_grid(const ConvConfig& config, int block_x, int block_y) {
    return dim3(
        (config.output_width() + block_x - 1) / block_x,
        (config.output_height() + block_y - 1) / block_y,
        kOutputChannels
    );
}

}  // namespace

void launch_direct_convolution(
    const float* input,
    const float* weights,
    float* output,
    const ConvConfig& config,
    int block_x,
    int block_y,
    cudaStream_t stream
) {
    const dim3 block(block_x, block_y);
    direct_convolution_kernel<<<make_grid(config, block_x, block_y), block, 0,
                                stream>>>(
        input,
        weights,
        output,
        config.height,
        config.width,
        config.output_height(),
        config.output_width(),
        config.stride,
        config.padding
    );
    CUDA_CHECK(cudaGetLastError());
}

void launch_shared_convolution(
    const float* input,
    const float* weights,
    float* output,
    const ConvConfig& config,
    int block_x,
    int block_y,
    cudaStream_t stream
) {
    const int tile_width = (block_x - 1) * config.stride + kKernelSize;
    const int tile_height = (block_y - 1) * config.stride + kKernelSize;
    const std::size_t shared_bytes =
        static_cast<std::size_t>(tile_width) * tile_height * sizeof(float);
    const dim3 block(block_x, block_y);

    shared_convolution_kernel<<<make_grid(config, block_x, block_y), block,
                                shared_bytes, stream>>>(
        input,
        weights,
        output,
        config.height,
        config.width,
        config.output_height(),
        config.output_width(),
        config.stride,
        config.padding,
        tile_height,
        tile_width
    );
    CUDA_CHECK(cudaGetLastError());
}
