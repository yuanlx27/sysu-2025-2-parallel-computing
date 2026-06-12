#include "benchmark.hpp"
#include "convolution.hpp"
#include "cuda_utils.cuh"
#include "tensor.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options {
    std::string output = "report/assets/metrics.csv";
    bool smoke = false;
    int warmup = 3;
    int runs = 10;
};

struct VerificationState {
    bool direct = false;
    bool shared = false;
    bool im2col = false;
    bool cudnn = false;
    std::string cudnn_status = "cudnn_unavailable";
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--output") {
            if (index + 1 >= argc) {
                throw std::invalid_argument("--output requires a path");
            }
            options.output = argv[++index];
        } else if (argument == "--smoke") {
            options.smoke = true;
            options.warmup = 1;
            options.runs = 2;
        } else if (argument == "--help") {
            std::cout
                << "Usage: lab10_conv [--output PATH] [--smoke]\n"
                << "  --output PATH  CSV output path\n"
                << "  --smoke        run a small benchmark matrix\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    return options;
}

void validate_config(const ConvConfig& config) {
    if (config.height <= 0 || config.width <= 0) {
        throw std::invalid_argument("input dimensions must be positive");
    }
    if (config.stride <= 0 || config.padding < 0) {
        throw std::invalid_argument("invalid stride or padding");
    }
    if (config.output_height() <= 0 || config.output_width() <= 0) {
        throw std::invalid_argument("configuration produces an empty output");
    }
}

std::vector<float> make_random_values(std::size_t count, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float> values(count);
    std::generate(values.begin(), values.end(), [&] {
        return distribution(generator);
    });
    return values;
}

void copy_to_device(
    DeviceBuffer<float>& destination,
    const std::vector<float>& source
) {
    CUDA_CHECK(cudaMemcpy(
        destination.data(),
        source.data(),
        source.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));
}

std::vector<float> copy_to_host(const DeviceBuffer<float>& source) {
    std::vector<float> result(source.size());
    CUDA_CHECK(cudaMemcpy(
        result.data(),
        source.data(),
        source.bytes(),
        cudaMemcpyDeviceToHost
    ));
    return result;
}

void require_close(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    const std::string& method
) {
    if (expected.size() != actual.size()) {
        throw std::runtime_error(method + " returned an unexpected output size");
    }

    double maximum_error = 0.0;
    std::size_t maximum_index = 0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const double absolute_error =
            std::abs(static_cast<double>(expected[index]) - actual[index]);
        const double tolerance =
            1.0e-4 + 1.0e-4 * std::abs(static_cast<double>(expected[index]));
        if (absolute_error > maximum_error) {
            maximum_error = absolute_error;
            maximum_index = index;
        }
        if (absolute_error > tolerance) {
            throw std::runtime_error(
                method + " verification failed at index " +
                std::to_string(maximum_index) + ", maximum error " +
                std::to_string(maximum_error)
            );
        }
    }
}

VerificationState verify_implementations(cudaStream_t stream) {
    VerificationState state;
    const std::vector<std::pair<int, int>> stride_padding = {
        {1, 0}, {1, 1}, {2, 0}, {2, 1}, {3, 0}, {3, 1}
    };

    for (const auto& [stride, padding] : stride_padding) {
        const ConvConfig config{17, 19, stride, padding};
        validate_config(config);

        const std::vector<float> input =
            make_random_values(config.input_elements(), 1000U + stride * 10U +
                                                          padding);
        const std::vector<float> weights =
            make_random_values(config.weight_elements(), 2000U);
        std::vector<float> reference;
        convolution_cpu(input, weights, reference, config);

        DeviceBuffer<float> device_input(config.input_elements());
        DeviceBuffer<float> device_weights(config.weight_elements());
        DeviceBuffer<float> device_output(config.output_elements());
        DeviceBuffer<float> device_columns(config.im2col_elements());
        copy_to_device(device_input, input);
        copy_to_device(device_weights, weights);

        launch_direct_convolution(
            device_input.data(),
            device_weights.data(),
            device_output.data(),
            config,
            16,
            16,
            stream
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));
        require_close(reference, copy_to_host(device_output), "direct");
        state.direct = true;

        launch_shared_convolution(
            device_input.data(),
            device_weights.data(),
            device_output.data(),
            config,
            16,
            16,
            stream
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));
        require_close(reference, copy_to_host(device_output), "direct-shared");
        state.shared = true;

        launch_im2col_convolution(
            device_input.data(),
            device_weights.data(),
            device_columns.data(),
            device_output.data(),
            config,
            stream
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));
        require_close(reference, copy_to_host(device_output), "im2col");
        state.im2col = true;

        CudnnConvolution cudnn(config);
        state.cudnn_status = cudnn.status();
        if (cudnn.available()) {
            cudnn.run(
                device_input.data(),
                device_weights.data(),
                device_output.data(),
                stream
            );
            CUDA_CHECK(cudaStreamSynchronize(stream));
            require_close(reference, copy_to_host(device_output), "cudnn");
            state.cudnn = true;
            state.cudnn_status = "ok";
        }
    }
    return state;
}

double calculate_gflops(const ConvConfig& config, double elapsed_ms) {
    if (elapsed_ms <= 0.0) {
        return 0.0;
    }
    const double operations =
        2.0 * config.output_height() * config.output_width() *
        kOutputChannels * kInputChannels * kKernelSize * kKernelSize;
    return operations / elapsed_ms / 1.0e6;
}

MetricRow make_metric(
    const std::string& method,
    const ConvConfig& config,
    int block_x,
    int block_y,
    int warmup,
    int runs,
    bool verified,
    const TimingStats& timing
) {
    MetricRow row;
    row.method = method;
    row.height = config.height;
    row.width = config.width;
    row.stride = config.stride;
    row.padding = config.padding;
    row.block_x = block_x;
    row.block_y = block_y;
    row.output_height = config.output_height();
    row.output_width = config.output_width();
    row.warmup = warmup;
    row.runs = runs;
    row.mean_ms = timing.mean_ms;
    row.min_ms = timing.min_ms;
    row.max_ms = timing.max_ms;
    row.gflops = calculate_gflops(config, timing.mean_ms);
    row.verified = verified;
    return row;
}

void append_timed_metric(
    std::vector<MetricRow>& rows,
    const std::string& method,
    const ConvConfig& config,
    int block_x,
    int block_y,
    int warmup,
    int runs,
    bool verified,
    const std::function<void(cudaStream_t)>& operation,
    cudaStream_t stream
) {
    const TimingStats timing =
        benchmark_cuda(operation, warmup, runs, stream);
    rows.push_back(make_metric(
        method,
        config,
        block_x,
        block_y,
        warmup,
        runs,
        verified,
        timing
    ));
    const MetricRow& row = rows.back();
    std::cout << row.method << ' ' << row.height << 'x' << row.width
              << " stride=" << row.stride << " padding=" << row.padding;
    if (row.block_x != 0) {
        std::cout << " block=" << row.block_x << 'x' << row.block_y;
    }
    std::cout << " mean=" << row.mean_ms << " ms\n";
}

std::vector<MetricRow> run_benchmarks(
    const Options& options,
    const VerificationState& verification,
    cudaStream_t stream
) {
    const std::vector<int> sizes =
        options.smoke ? std::vector<int>{64} :
                        std::vector<int>{256, 512, 1024, 2048};
    const std::vector<int> strides =
        options.smoke ? std::vector<int>{1, 2} :
                        std::vector<int>{1, 2, 3};
    const std::vector<int> paddings =
        options.smoke ? std::vector<int>{0, 1} :
                        std::vector<int>{0, 1};
    const std::vector<std::pair<int, int>> blocks =
        options.smoke
            ? std::vector<std::pair<int, int>>{{16, 16}}
            : std::vector<std::pair<int, int>>{
                  {8, 8}, {16, 16}, {32, 8}, {32, 16}
              };

    std::vector<MetricRow> rows;
    bool recorded_cudnn_unavailable = false;
    const std::vector<float> weights =
        make_random_values(
            static_cast<std::size_t>(kOutputChannels) * kInputChannels *
                kKernelSize * kKernelSize,
            42U
        );

    for (int size : sizes) {
        for (int stride : strides) {
            for (int padding : paddings) {
                const ConvConfig config{size, size, stride, padding};
                validate_config(config);

                const std::vector<float> input =
                    make_random_values(
                        config.input_elements(),
                        static_cast<std::uint32_t>(
                            size * 100 + stride * 10 + padding
                        )
                    );
                DeviceBuffer<float> device_input(config.input_elements());
                DeviceBuffer<float> device_weights(config.weight_elements());
                DeviceBuffer<float> device_output(config.output_elements());
                copy_to_device(device_input, input);
                copy_to_device(device_weights, weights);

                for (const auto& block_shape : blocks) {
                    const int block_x = block_shape.first;
                    const int block_y = block_shape.second;
                    append_timed_metric(
                        rows,
                        "direct",
                        config,
                        block_x,
                        block_y,
                        options.warmup,
                        options.runs,
                        verification.direct,
                        [&](cudaStream_t operation_stream) {
                            launch_direct_convolution(
                                device_input.data(),
                                device_weights.data(),
                                device_output.data(),
                                config,
                                block_x,
                                block_y,
                                operation_stream
                            );
                        },
                        stream
                    );

                    append_timed_metric(
                        rows,
                        "direct-shared",
                        config,
                        block_x,
                        block_y,
                        options.warmup,
                        options.runs,
                        verification.shared,
                        [&](cudaStream_t operation_stream) {
                            launch_shared_convolution(
                                device_input.data(),
                                device_weights.data(),
                                device_output.data(),
                                config,
                                block_x,
                                block_y,
                                operation_stream
                            );
                        },
                        stream
                    );
                }

                DeviceBuffer<float> device_columns(config.im2col_elements());
                append_timed_metric(
                    rows,
                    "im2col",
                    config,
                    16,
                    16,
                    options.warmup,
                    options.runs,
                    verification.im2col,
                    [&](cudaStream_t operation_stream) {
                        launch_im2col_convolution(
                            device_input.data(),
                            device_weights.data(),
                            device_columns.data(),
                            device_output.data(),
                            config,
                            operation_stream
                        );
                    },
                    stream
                );

                CudnnConvolution cudnn(config);
                if (cudnn.available()) {
                    append_timed_metric(
                        rows,
                        "cudnn",
                        config,
                        0,
                        0,
                        options.warmup,
                        options.runs,
                        verification.cudnn,
                        [&](cudaStream_t operation_stream) {
                            cudnn.run(
                                device_input.data(),
                                device_weights.data(),
                                device_output.data(),
                                operation_stream
                            );
                        },
                        stream
                    );
                } else if (!recorded_cudnn_unavailable) {
                    MetricRow row;
                    row.method = "cudnn";
                    row.status = cudnn.status();
                    rows.push_back(row);
                    recorded_cudnn_unavailable = true;
                    std::cout << "cudnn status=" << cudnn.status() << '\n';
                }
            }
        }
    }
    return rows;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            throw std::runtime_error("no CUDA-capable device is available");
        }

        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        std::cout << "CUDA device: " << properties.name << '\n';

        CudaStream stream;
        std::cout << "verifying implementations...\n";
        const VerificationState verification =
            verify_implementations(stream.get());
        std::cout << "running benchmarks...\n";
        const std::vector<MetricRow> rows =
            run_benchmarks(options, verification, stream.get());
        write_metrics_csv(options.output, rows);
        std::cout << "metrics written to " << options.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
