#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

void checkCuda(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s failed: %s\n",
                     file, line, expr, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expr) checkCuda((expr), #expr, __FILE__, __LINE__)

enum class Method {
    Naive,
    Shared,
};

enum class PrintMode {
    None,
    Sample,
    Full,
};

struct Options {
    int n = -1;
    Method method = Method::Shared;
    int tile = 32;
    int blockRows = 8;
    int repeat = 20;
    unsigned int seed = 20250609U;
    PrintMode printMode = PrintMode::Sample;
    bool benchmark = false;
};

struct RunResult {
    float avgMs = 0.0F;
    double bandwidthGbps = 0.0;
    bool correct = false;
};

__global__ void transposeNaiveKernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        output[static_cast<std::size_t>(col) * n + row] =
            input[static_cast<std::size_t>(row) * n + col];
    }
}

__global__ void transposeSharedKernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int n) {
    extern __shared__ float tile[];

    const int tileDim = blockDim.x;
    const int blockRows = blockDim.y;
    const int stride = tileDim + 1;

    const int inputCol = blockIdx.x * tileDim + threadIdx.x;
    const int inputRowBase = blockIdx.y * tileDim + threadIdx.y;

    for (int offset = 0; offset < tileDim; offset += blockRows) {
        const int localRow = threadIdx.y + offset;
        const int inputRow = inputRowBase + offset;
        if (localRow < tileDim && inputRow < n && inputCol < n) {
            tile[localRow * stride + threadIdx.x] =
                input[static_cast<std::size_t>(inputRow) * n + inputCol];
        }
    }

    __syncthreads();

    const int outputCol = blockIdx.y * tileDim + threadIdx.x;
    const int outputRowBase = blockIdx.x * tileDim + threadIdx.y;

    for (int offset = 0; offset < tileDim; offset += blockRows) {
        const int localCol = threadIdx.y + offset;
        const int outputRow = outputRowBase + offset;
        if (localCol < tileDim && outputRow < n && outputCol < n) {
            output[static_cast<std::size_t>(outputRow) * n + outputCol] =
                tile[threadIdx.x * stride + localCol];
        }
    }
}

bool parseInt(const std::string& text, int* value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (text.c_str() == end || *end != '\0' || errno != 0 ||
        parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

bool parseUnsigned(const std::string& text, unsigned int* value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(text.c_str(), &end, 10);
    if (text.c_str() == end || *end != '\0' || errno != 0 ||
        parsed > std::numeric_limits<unsigned int>::max()) {
        return false;
    }
    *value = static_cast<unsigned int>(parsed);
    return true;
}

const char* methodName(Method method) {
    return method == Method::Naive ? "naive" : "shared";
}

bool parseMethod(const std::string& text, Method* method) {
    if (text == "naive") {
        *method = Method::Naive;
        return true;
    }
    if (text == "shared") {
        *method = Method::Shared;
        return true;
    }
    return false;
}

bool parsePrintMode(const std::string& text, PrintMode* mode) {
    if (text == "none") {
        *mode = PrintMode::None;
        return true;
    }
    if (text == "sample") {
        *mode = PrintMode::Sample;
        return true;
    }
    if (text == "full") {
        *mode = PrintMode::Full;
        return true;
    }
    return false;
}

bool startsWithDash(const std::string& text) {
    return !text.empty() && text[0] == '-';
}

void printUsage(const char* program) {
    std::cerr
        << "Usage:\n"
        << "  " << program << " <n> [--method naive|shared] [--tile N] [--block-rows N]\n"
        << "      [--repeat N] [--print none|sample|full] [--seed N]\n"
        << "  " << program << " --benchmark [<n>] [--repeat N]\n\n"
        << "Input n must be in [512, 2048].\n"
        << "Default method is shared, default tile size is 32, default block rows is 8.\n";
}

bool readValue(int argc, char** argv, int* index, std::string* value) {
    if (*index + 1 >= argc) {
        return false;
    }
    *value = argv[++(*index)];
    return true;
}

bool parseArgs(int argc, char** argv, Options* options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        std::string value;

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        }
        if (arg == "--benchmark") {
            options->benchmark = true;
        } else if (arg == "--method") {
            if (!readValue(argc, argv, &i, &value) || !parseMethod(value, &options->method)) {
                std::cerr << "Error: --method expects naive or shared.\n";
                return false;
            }
        } else if (arg == "--tile") {
            if (!readValue(argc, argv, &i, &value) || !parseInt(value, &options->tile)) {
                std::cerr << "Error: --tile expects an integer.\n";
                return false;
            }
        } else if (arg == "--block-rows") {
            if (!readValue(argc, argv, &i, &value) || !parseInt(value, &options->blockRows)) {
                std::cerr << "Error: --block-rows expects an integer.\n";
                return false;
            }
        } else if (arg == "--repeat") {
            if (!readValue(argc, argv, &i, &value) || !parseInt(value, &options->repeat)) {
                std::cerr << "Error: --repeat expects an integer.\n";
                return false;
            }
        } else if (arg == "--print") {
            if (!readValue(argc, argv, &i, &value) || !parsePrintMode(value, &options->printMode)) {
                std::cerr << "Error: --print expects none, sample, or full.\n";
                return false;
            }
        } else if (arg == "--seed") {
            if (!readValue(argc, argv, &i, &value) || !parseUnsigned(value, &options->seed)) {
                std::cerr << "Error: --seed expects an unsigned integer.\n";
                return false;
            }
        } else if (!startsWithDash(arg) && options->n < 0) {
            if (!parseInt(arg, &options->n)) {
                std::cerr << "Error: n must be an integer.\n";
                return false;
            }
        } else {
            std::cerr << "Error: unknown argument '" << arg << "'.\n";
            return false;
        }
    }

    if (!options->benchmark && options->n < 0) {
        if (!(std::cin >> options->n)) {
            std::cerr << "Error: missing matrix size n.\n";
            return false;
        }
    }

    return true;
}

bool validateSize(int n) {
    if (n <= 0) {
        std::cerr << "Error: n must be positive.\n";
        return false;
    }
    if (n < 512 || n > 2048) {
        std::cerr << "Error: n must be in [512, 2048].\n";
        return false;
    }
    return true;
}

bool validateKernelConfig(Method method, int tile, int blockRows) {
    if (tile <= 0 || tile > 32) {
        std::cerr << "Error: --tile must be in [1, 32].\n";
        return false;
    }

    if (method == Method::Naive) {
        if (tile * tile > 1024) {
            std::cerr << "Error: naive method uses tile x tile threads, max 1024 threads per block.\n";
            return false;
        }
        return true;
    }

    if (blockRows <= 0 || blockRows > tile) {
        std::cerr << "Error: --block-rows must be in [1, tile] for shared method.\n";
        return false;
    }
    if (tile % blockRows != 0) {
        std::cerr << "Error: tile must be divisible by block rows for shared method.\n";
        return false;
    }
    if (tile * blockRows > 1024) {
        std::cerr << "Error: shared method exceeds the 1024 threads per block limit.\n";
        return false;
    }
    return true;
}

void fillRandom(std::vector<float>* matrix, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0F, 100.0F);
    for (float& value : *matrix) {
        value = dist(rng);
    }
}

void launchTranspose(Method method,
                     const float* dInput,
                     float* dOutput,
                     int n,
                     int tile,
                     int blockRows) {
    const dim3 grid((n + tile - 1) / tile, (n + tile - 1) / tile);

    if (method == Method::Naive) {
        const dim3 block(tile, tile);
        transposeNaiveKernel<<<grid, block>>>(dInput, dOutput, n);
    } else {
        const dim3 block(tile, blockRows);
        const std::size_t sharedBytes = static_cast<std::size_t>(tile) * (tile + 1) * sizeof(float);
        transposeSharedKernel<<<grid, block, sharedBytes>>>(dInput, dOutput, n);
    }

    CUDA_CHECK(cudaGetLastError());
}

float timeTranspose(Method method,
                    const float* dInput,
                    float* dOutput,
                    int n,
                    int tile,
                    int blockRows,
                    int repeat) {
    launchTranspose(method, dInput, dOutput, n, tile, blockRows);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        launchTranspose(method, dInput, dOutput, n, tile, blockRows);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return elapsedMs / static_cast<float>(repeat);
}

bool verifyTranspose(const std::vector<float>& input, const std::vector<float>& output, int n) {
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            const float expected = input[static_cast<std::size_t>(col) * n + row];
            const float actual = output[static_cast<std::size_t>(row) * n + col];
            if (actual != expected) {
                std::cerr << "Mismatch at output(" << row << ", " << col << "): expected "
                          << expected << ", got " << actual << "\n";
                return false;
            }
        }
    }
    return true;
}

double effectiveBandwidthGbps(int n, float avgMs) {
    const double bytes = 2.0 * static_cast<double>(n) * static_cast<double>(n) * sizeof(float);
    return bytes / (static_cast<double>(avgMs) / 1000.0) / 1.0e9;
}

void printMatrixFull(const char* name, const std::vector<float>& matrix, int n) {
    std::cout << name << " (" << n << " x " << n << "):\n";
    std::cout << std::fixed << std::setprecision(2);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            std::cout << std::setw(8) << matrix[static_cast<std::size_t>(row) * n + col];
        }
        std::cout << '\n';
    }
}

void printMatrixSample(const char* name, const std::vector<float>& matrix, int n) {
    const int rows = std::min(n, 8);
    const int cols = std::min(n, 8);

    std::cout << name << " top-left " << rows << " x " << cols << " sample:\n";
    std::cout << std::fixed << std::setprecision(2);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            std::cout << std::setw(8) << matrix[static_cast<std::size_t>(row) * n + col];
        }
        std::cout << '\n';
    }
}

RunResult runCase(int n,
                  Method method,
                  int tile,
                  int blockRows,
                  int repeat,
                  unsigned int seed,
                  PrintMode printMode) {
    const std::size_t total = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = total * sizeof(float);

    std::vector<float> hInput(total);
    std::vector<float> hOutput(total, 0.0F);
    fillRandom(&hInput, seed);

    float* dInput = nullptr;
    float* dOutput = nullptr;
    CUDA_CHECK(cudaMalloc(&dInput, bytes));
    CUDA_CHECK(cudaMalloc(&dOutput, bytes));
    CUDA_CHECK(cudaMemcpy(dInput, hInput.data(), bytes, cudaMemcpyHostToDevice));

    const float avgMs = timeTranspose(method, dInput, dOutput, n, tile, blockRows, repeat);

    CUDA_CHECK(cudaMemcpy(hOutput.data(), dOutput, bytes, cudaMemcpyDeviceToHost));

    const bool correct = verifyTranspose(hInput, hOutput, n);
    const double bandwidth = effectiveBandwidthGbps(n, avgMs);

    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dOutput));

    if (printMode == PrintMode::Full) {
        printMatrixFull("A", hInput, n);
        printMatrixFull("A_T", hOutput, n);
    } else if (printMode == PrintMode::Sample) {
        printMatrixSample("A", hInput, n);
        printMatrixSample("A_T", hOutput, n);
    }

    return RunResult{avgMs, bandwidth, correct};
}

void printRunSummary(int n, Method method, int tile, int blockRows, const RunResult& result) {
    std::cout << std::fixed << std::setprecision(4)
              << "n=" << n
              << ", method=" << methodName(method)
              << ", tile=" << tile
              << ", block_rows=" << (method == Method::Naive ? tile : blockRows)
              << ", avg_kernel_time_ms=" << result.avgMs
              << ", effective_bandwidth_GBps=" << result.bandwidthGbps
              << ", correct=" << (result.correct ? "yes" : "no") << '\n';
}

bool runBenchmark(const Options& options) {
    const std::vector<int> sizes = options.n > 0 ? std::vector<int>{options.n}
                                                 : std::vector<int>{512, 1024, 2048};
    const std::vector<int> tiles{8, 16, 32};

    std::cout << "n,method,tile,block_rows,avg_kernel_time_ms,effective_bandwidth_GBps,correct\n";
    for (const int n : sizes) {
        if (!validateSize(n)) {
            return false;
        }

        for (const int tile : tiles) {
            if (!validateKernelConfig(Method::Naive, tile, tile)) {
                return false;
            }
            const RunResult naive =
                runCase(n, Method::Naive, tile, tile, options.repeat, options.seed, PrintMode::None);
            std::cout << std::fixed << std::setprecision(4)
                      << n << ",naive," << tile << ',' << tile << ','
                      << naive.avgMs << ',' << naive.bandwidthGbps << ','
                      << (naive.correct ? "yes" : "no") << '\n';

            const std::vector<int> blockRowsCandidates{tile, tile / 2, tile >= 8 ? 8 : tile};
            for (const int blockRows : blockRowsCandidates) {
                if (blockRows <= 0 || tile % blockRows != 0) {
                    continue;
                }
                if (!validateKernelConfig(Method::Shared, tile, blockRows)) {
                    return false;
                }
                const RunResult shared =
                    runCase(n, Method::Shared, tile, blockRows, options.repeat, options.seed, PrintMode::None);
                std::cout << std::fixed << std::setprecision(4)
                          << n << ",shared," << tile << ',' << blockRows << ','
                          << shared.avgMs << ',' << shared.bandwidthGbps << ','
                          << (shared.correct ? "yes" : "no") << '\n';
            }
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parseArgs(argc, argv, &options)) {
        printUsage(argv[0]);
        return 1;
    }

    if (options.repeat <= 0) {
        std::cerr << "Error: --repeat must be positive.\n";
        return 1;
    }

    if (options.benchmark) {
        const bool ok = runBenchmark(options);
        CUDA_CHECK(cudaDeviceReset());
        return ok ? 0 : 1;
    }

    if (!validateSize(options.n) ||
        !validateKernelConfig(options.method, options.tile, options.blockRows)) {
        return 1;
    }

    const RunResult result = runCase(options.n,
                                     options.method,
                                     options.tile,
                                     options.blockRows,
                                     options.repeat,
                                     options.seed,
                                     options.printMode);

    printRunSummary(options.n, options.method, options.tile, options.blockRows, result);
    CUDA_CHECK(cudaDeviceReset());
    return result.correct ? 0 : 1;
}
