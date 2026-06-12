#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <string>
#include <vector>

struct TimingStats {
    double mean_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

struct MetricRow {
    std::string method;
    int height = 0;
    int width = 0;
    int channels = 3;
    int filters = 3;
    int kernel = 3;
    int stride = 0;
    int padding = 0;
    int block_x = 0;
    int block_y = 0;
    int output_height = 0;
    int output_width = 0;
    int warmup = 0;
    int runs = 0;
    double mean_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double gflops = 0.0;
    bool verified = false;
    std::string status = "ok";
};

TimingStats benchmark_cuda(
    const std::function<void(cudaStream_t)>& operation,
    int warmup,
    int runs,
    cudaStream_t stream
);

void write_metrics_csv(
    const std::string& path,
    const std::vector<MetricRow>& rows
);
