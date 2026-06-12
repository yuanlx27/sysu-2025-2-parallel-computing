#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

constexpr int kInputChannels = 3;
constexpr int kOutputChannels = 3;
constexpr int kKernelSize = 3;

struct ConvConfig {
    int height = 0;
    int width = 0;
    int stride = 1;
    int padding = 0;

    int output_height() const {
        return (height + 2 * padding - kKernelSize) / stride + 1;
    }

    int output_width() const {
        return (width + 2 * padding - kKernelSize) / stride + 1;
    }

    std::size_t input_elements() const {
        return static_cast<std::size_t>(kInputChannels) * height * width;
    }

    std::size_t weight_elements() const {
        return static_cast<std::size_t>(kOutputChannels) *
               kInputChannels * kKernelSize * kKernelSize;
    }

    std::size_t output_elements() const {
        return static_cast<std::size_t>(kOutputChannels) *
               output_height() * output_width();
    }

    std::size_t im2col_elements() const {
        return static_cast<std::size_t>(kInputChannels) *
               kKernelSize * kKernelSize *
               output_height() * output_width();
    }
};

void convolution_cpu(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    std::vector<float>& output,
    const ConvConfig& config
);

void launch_direct_convolution(
    const float* input,
    const float* weights,
    float* output,
    const ConvConfig& config,
    int block_x,
    int block_y,
    cudaStream_t stream
);

void launch_shared_convolution(
    const float* input,
    const float* weights,
    float* output,
    const ConvConfig& config,
    int block_x,
    int block_y,
    cudaStream_t stream
);

void launch_im2col(
    const float* input,
    float* columns,
    const ConvConfig& config,
    cudaStream_t stream
);

void launch_tiled_gemm(
    const float* weights,
    const float* columns,
    float* output,
    int rows,
    int inner,
    int columns_count,
    cudaStream_t stream
);

void launch_im2col_convolution(
    const float* input,
    const float* weights,
    float* columns,
    float* output,
    const ConvConfig& config,
    cudaStream_t stream
);

class CudnnConvolution {
public:
    explicit CudnnConvolution(const ConvConfig& config);
    ~CudnnConvolution();

    CudnnConvolution(const CudnnConvolution&) = delete;
    CudnnConvolution& operator=(const CudnnConvolution&) = delete;

    bool available() const;
    const std::string& status() const;

    void run(
        const float* input,
        const float* weights,
        float* output,
        cudaStream_t stream
    );

private:
    class Impl;
    Impl* impl_;
};
