#include "convolution.hpp"
#include "cuda_utils.cuh"

#include <cuda_runtime.h>

namespace {

constexpr int kTileSize = 16;

__global__ void tiled_gemm_kernel(
    const float* left,
    const float* right,
    float* output,
    int rows,
    int inner,
    int columns
) {
    __shared__ float left_tile[kTileSize][kTileSize];
    __shared__ float right_tile[kTileSize][kTileSize];

    const int row = blockIdx.y * kTileSize + threadIdx.y;
    const int column = blockIdx.x * kTileSize + threadIdx.x;
    float sum = 0.0F;

    for (int tile = 0; tile < (inner + kTileSize - 1) / kTileSize; ++tile) {
        const int left_column = tile * kTileSize + threadIdx.x;
        const int right_row = tile * kTileSize + threadIdx.y;

        left_tile[threadIdx.y][threadIdx.x] =
            row < rows && left_column < inner
                ? left[static_cast<std::size_t>(row) * inner + left_column]
                : 0.0F;
        right_tile[threadIdx.y][threadIdx.x] =
            right_row < inner && column < columns
                ? right[static_cast<std::size_t>(right_row) * columns + column]
                : 0.0F;
        __syncthreads();

#pragma unroll
        for (int index = 0; index < kTileSize; ++index) {
            sum += left_tile[threadIdx.y][index] *
                   right_tile[index][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < rows && column < columns) {
        output[static_cast<std::size_t>(row) * columns + column] = sum;
    }
}

}  // namespace

void launch_tiled_gemm(
    const float* weights,
    const float* columns,
    float* output,
    int rows,
    int inner,
    int columns_count,
    cudaStream_t stream
) {
    const dim3 block(kTileSize, kTileSize);
    const dim3 grid(
        (columns_count + kTileSize - 1) / kTileSize,
        (rows + kTileSize - 1) / kTileSize
    );
    tiled_gemm_kernel<<<grid, block, 0, stream>>>(
        weights,
        columns,
        output,
        rows,
        inner,
        columns_count
    );
    CUDA_CHECK(cudaGetLastError());
}
