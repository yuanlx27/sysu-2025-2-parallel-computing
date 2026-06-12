#include "benchmark.hpp"
#include "cuda_utils.cuh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>

TimingStats benchmark_cuda(
    const std::function<void(cudaStream_t)>& operation,
    int warmup,
    int runs,
    cudaStream_t stream
) {
    for (int index = 0; index < warmup; ++index) {
        operation(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CudaEvent start;
    CudaEvent stop;
    std::vector<double> samples;
    samples.reserve(runs);

    for (int index = 0; index < runs; ++index) {
        CUDA_CHECK(cudaEventRecord(start.get(), stream));
        operation(stream);
        CUDA_CHECK(cudaEventRecord(stop.get(), stream));
        CUDA_CHECK(cudaEventSynchronize(stop.get()));

        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get()));
        samples.push_back(elapsed_ms);
    }

    TimingStats stats;
    stats.mean_ms =
        std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    stats.min_ms = *std::min_element(samples.begin(), samples.end());
    stats.max_ms = *std::max_element(samples.begin(), samples.end());
    return stats;
}

void write_metrics_csv(
    const std::string& path,
    const std::vector<MetricRow>& rows
) {
    const std::filesystem::path output_path(path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open metrics file: " + path);
    }

    output << "method,height,width,channels,filters,kernel,stride,padding,"
              "block_x,block_y,output_height,output_width,warmup,runs,"
              "mean_ms,min_ms,max_ms,gflops,verified,status\n";
    output << std::fixed << std::setprecision(6);

    for (const MetricRow& row : rows) {
        output << row.method << ','
               << row.height << ','
               << row.width << ','
               << row.channels << ','
               << row.filters << ','
               << row.kernel << ','
               << row.stride << ','
               << row.padding << ','
               << row.block_x << ','
               << row.block_y << ','
               << row.output_height << ','
               << row.output_width << ','
               << row.warmup << ','
               << row.runs << ','
               << row.mean_ms << ','
               << row.min_ms << ','
               << row.max_ms << ','
               << row.gflops << ','
               << (row.verified ? "true" : "false") << ','
               << row.status << '\n';
    }
}
