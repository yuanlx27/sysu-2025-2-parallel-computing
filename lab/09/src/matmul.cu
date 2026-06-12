#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {

constexpr int kMinDimension = 128;
constexpr int kMaxDimension = 2048;

void checkCuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }

    std::ostringstream message;
    message << "CUDA error at " << file << ':' << line << " for " << expression
            << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

enum class KernelKind {
    Naive,
    Tiled,
    Coarsened,
};

enum class PrintMode {
    None,
    Summary,
    Full,
};

struct Options {
    int m = 0;
    int n = 0;
    int k = 0;
    int blockSize = 16;
    int warmup = 2;
    int repeat = 10;
    std::uint32_t seed = 2025;
    KernelKind kernel = KernelKind::Tiled;
    PrintMode printMode = PrintMode::Summary;
    bool allKernels = false;
    bool csv = false;
    bool verify = true;
    bool dimensionsProvided = false;
};

struct Timing {
    float kernelMs = 0.0F;
    double totalMs = 0.0;
};

struct RunResult {
    KernelKind kernel;
    int blockSize;
    Timing timing;
    double maxAbsError;
    double maxRelError;
    bool verificationPerformed;
    bool verified;
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                cudaFree(data_);
            }
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    T* get() {
        return data_;
    }

    const T* get() const {
        return data_;
    }

    std::size_t bytes() const {
        return count_ * sizeof(T);
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

class CudaEvent {
public:
    CudaEvent() {
        CUDA_CHECK(cudaEventCreate(&event_));
    }

    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const {
        return event_;
    }

private:
    cudaEvent_t event_ = nullptr;
};

__global__ void matmulNaive(const float* a, const float* b, float* c, int m, int n, int k) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= k) {
        return;
    }

    float sum = 0.0F;
    for (int inner = 0; inner < n; ++inner) {
        sum += a[row * n + inner] * b[inner * k + col];
    }
    c[row * k + col] = sum;
}

__global__ void matmulTiled(const float* a, const float* b, float* c, int m, int n, int k) {
    extern __shared__ float shared[];
    const int tileSize = blockDim.x;
    float* tileA = shared;
    float* tileB = shared + tileSize * tileSize;

    const int localRow = threadIdx.y;
    const int localCol = threadIdx.x;
    const int row = blockIdx.y * tileSize + localRow;
    const int col = blockIdx.x * tileSize + localCol;
    float sum = 0.0F;

    const int tileCount = (n + tileSize - 1) / tileSize;
    for (int tile = 0; tile < tileCount; ++tile) {
        const int aCol = tile * tileSize + localCol;
        const int bRow = tile * tileSize + localRow;
        tileA[localRow * tileSize + localCol] =
            (row < m && aCol < n) ? a[row * n + aCol] : 0.0F;
        tileB[localRow * tileSize + localCol] =
            (bRow < n && col < k) ? b[bRow * k + col] : 0.0F;
        __syncthreads();

        for (int inner = 0; inner < tileSize; ++inner) {
            sum += tileA[localRow * tileSize + inner]
                   * tileB[inner * tileSize + localCol];
        }
        __syncthreads();
    }

    if (row < m && col < k) {
        c[row * k + col] = sum;
    }
}

// Each thread computes two neighboring C elements, reusing the same A values.
__global__ void matmulCoarsened(
    const float* a, const float* b, float* c, int m, int n, int k) {
    extern __shared__ float shared[];
    const int tileSize = blockDim.x;
    float* tileA = shared;
    float* tileB = shared + tileSize * tileSize;

    const int localRow = threadIdx.y;
    const int localCol = threadIdx.x;
    const int row = blockIdx.y * tileSize + localRow;
    const int col0 = blockIdx.x * (2 * tileSize) + localCol;
    const int col1 = col0 + tileSize;
    float sum0 = 0.0F;
    float sum1 = 0.0F;

    const int tileCount = (n + tileSize - 1) / tileSize;
    for (int tile = 0; tile < tileCount; ++tile) {
        const int aCol = tile * tileSize + localCol;
        const int bRow = tile * tileSize + localRow;

        tileA[localRow * tileSize + localCol] =
            (row < m && aCol < n) ? a[row * n + aCol] : 0.0F;
        tileB[localRow * (2 * tileSize) + localCol] =
            (bRow < n && col0 < k) ? b[bRow * k + col0] : 0.0F;
        tileB[localRow * (2 * tileSize) + tileSize + localCol] =
            (bRow < n && col1 < k) ? b[bRow * k + col1] : 0.0F;
        __syncthreads();

        for (int inner = 0; inner < tileSize; ++inner) {
            const float aValue = tileA[localRow * tileSize + inner];
            sum0 += aValue * tileB[inner * (2 * tileSize) + localCol];
            sum1 += aValue * tileB[inner * (2 * tileSize) + tileSize + localCol];
        }
        __syncthreads();
    }

    if (row < m && col0 < k) {
        c[row * k + col0] = sum0;
    }
    if (row < m && col1 < k) {
        c[row * k + col1] = sum1;
    }
}

std::string_view kernelName(KernelKind kernel) {
    switch (kernel) {
        case KernelKind::Naive:
            return "naive";
        case KernelKind::Tiled:
            return "tiled";
        case KernelKind::Coarsened:
            return "coarsened";
    }
    return "unknown";
}

int parseInteger(const std::string& text, std::string_view name) {
    std::size_t consumed = 0;
    long value = 0;
    try {
        value = std::stol(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " must be an integer: " + text);
    }
    if (consumed != text.size() || value < std::numeric_limits<int>::min()
        || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(name) + " must be an integer: " + text);
    }
    return static_cast<int>(value);
}

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " M N K [options]\n\n"
        << "Options:\n"
        << "  --kernel naive|tiled|coarsened|all  CUDA implementation (default: tiled)\n"
        << "  --block 8|16|32                    square thread-block size (default: 16)\n"
        << "  --warmup N                         warm-up launches (default: 2)\n"
        << "  --repeat N                         timed launches (default: 10)\n"
        << "  --seed N                           random seed (default: 2025)\n"
        << "  --print none|summary|full          matrix output mode (default: summary)\n"
        << "  --csv                              print machine-readable CSV results\n"
        << "  --no-verify                        skip sampled CPU correctness check\n"
        << "  --help                             show this message\n\n"
        << "If M N K are omitted, they are read from stdin.\n";
}

Options parseOptions(int argc, char** argv) {
    Options options;
    std::vector<std::string> positional;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto requireValue = [&](std::string_view option) -> std::string {
            if (++index >= argc) {
                throw std::invalid_argument(std::string(option) + " requires a value");
            }
            return argv[index];
        };

        if (argument == "--help" || argument == "-h") {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (argument == "--kernel") {
            const std::string value = requireValue(argument);
            if (value == "naive") {
                options.kernel = KernelKind::Naive;
            } else if (value == "tiled") {
                options.kernel = KernelKind::Tiled;
            } else if (value == "coarsened") {
                options.kernel = KernelKind::Coarsened;
            } else if (value == "all") {
                options.allKernels = true;
            } else {
                throw std::invalid_argument("unknown kernel: " + value);
            }
        } else if (argument == "--block") {
            options.blockSize = parseInteger(requireValue(argument), "block size");
        } else if (argument == "--warmup") {
            options.warmup = parseInteger(requireValue(argument), "warmup count");
        } else if (argument == "--repeat") {
            options.repeat = parseInteger(requireValue(argument), "repeat count");
        } else if (argument == "--seed") {
            const int seed = parseInteger(requireValue(argument), "seed");
            if (seed < 0) {
                throw std::invalid_argument("seed must be non-negative");
            }
            options.seed = static_cast<std::uint32_t>(seed);
        } else if (argument == "--print") {
            const std::string value = requireValue(argument);
            if (value == "none") {
                options.printMode = PrintMode::None;
            } else if (value == "summary") {
                options.printMode = PrintMode::Summary;
            } else if (value == "full") {
                options.printMode = PrintMode::Full;
            } else {
                throw std::invalid_argument("unknown print mode: " + value);
            }
        } else if (argument == "--no-verify") {
            options.verify = false;
        } else if (argument == "--csv") {
            options.csv = true;
            options.printMode = PrintMode::None;
        } else if (!argument.empty() && argument.front() == '-') {
            throw std::invalid_argument("unknown option: " + argument);
        } else {
            positional.push_back(argument);
        }
    }

    if (!positional.empty()) {
        if (positional.size() != 3) {
            throw std::invalid_argument("expected exactly three dimensions: M N K");
        }
        options.m = parseInteger(positional[0], "M");
        options.n = parseInteger(positional[1], "N");
        options.k = parseInteger(positional[2], "K");
        options.dimensionsProvided = true;
    }

    if (options.blockSize != 8 && options.blockSize != 16 && options.blockSize != 32) {
        throw std::invalid_argument("block size must be 8, 16, or 32");
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("warmup must be non-negative and repeat must be positive");
    }
    return options;
}

void validateDimensions(int m, int n, int k) {
    if (m < kMinDimension || m > kMaxDimension || n < kMinDimension
        || n > kMaxDimension || k < kMinDimension || k > kMaxDimension) {
        std::ostringstream message;
        message << "M, N, and K must all be in [" << kMinDimension << ", "
                << kMaxDimension << ']';
        throw std::invalid_argument(message.str());
    }
}

std::vector<float> randomMatrix(std::size_t count, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float> matrix(count);
    for (float& value : matrix) {
        value = distribution(generator);
    }
    return matrix;
}

void launchKernel(
    KernelKind kernel,
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    int blockSize) {
    const dim3 block(blockSize, blockSize);
    dim3 grid;
    std::size_t sharedBytes = 0;

    switch (kernel) {
        case KernelKind::Naive:
            grid = dim3(
                (k + blockSize - 1) / blockSize,
                (m + blockSize - 1) / blockSize);
            matmulNaive<<<grid, block>>>(a, b, c, m, n, k);
            break;
        case KernelKind::Tiled:
            grid = dim3(
                (k + blockSize - 1) / blockSize,
                (m + blockSize - 1) / blockSize);
            sharedBytes =
                2ULL * static_cast<std::size_t>(blockSize) * blockSize * sizeof(float);
            matmulTiled<<<grid, block, sharedBytes>>>(a, b, c, m, n, k);
            break;
        case KernelKind::Coarsened:
            grid = dim3(
                (k + 2 * blockSize - 1) / (2 * blockSize),
                (m + blockSize - 1) / blockSize);
            sharedBytes =
                3ULL * static_cast<std::size_t>(blockSize) * blockSize * sizeof(float);
            matmulCoarsened<<<grid, block, sharedBytes>>>(a, b, c, m, n, k);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

Timing execute(
    KernelKind kernel,
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c,
    int m,
    int n,
    int k,
    int blockSize,
    int warmup,
    int repeat) {
    const auto totalStart = std::chrono::steady_clock::now();
    DeviceBuffer<float> deviceA(a.size());
    DeviceBuffer<float> deviceB(b.size());
    DeviceBuffer<float> deviceC(c.size());

    CUDA_CHECK(cudaMemcpy(deviceA.get(), a.data(), deviceA.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB.get(), b.data(), deviceB.bytes(), cudaMemcpyHostToDevice));

    for (int run = 0; run < warmup; ++run) {
        launchKernel(kernel, deviceA.get(), deviceB.get(), deviceC.get(), m, n, k, blockSize);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaEvent start;
    CudaEvent stop;
    CUDA_CHECK(cudaEventRecord(start.get()));
    for (int run = 0; run < repeat; ++run) {
        launchKernel(kernel, deviceA.get(), deviceB.get(), deviceC.get(), m, n, k, blockSize);
    }
    CUDA_CHECK(cudaEventRecord(stop.get()));
    CUDA_CHECK(cudaEventSynchronize(stop.get()));

    float elapsedMs = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start.get(), stop.get()));
    CUDA_CHECK(cudaMemcpy(c.data(), deviceC.get(), deviceC.bytes(), cudaMemcpyDeviceToHost));
    const auto totalStop = std::chrono::steady_clock::now();

    const double totalMs =
        std::chrono::duration<double, std::milli>(totalStop - totalStart).count();
    return {elapsedMs / static_cast<float>(repeat), totalMs};
}

std::pair<double, double> verifySampled(
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::vector<float>& c,
    int m,
    int n,
    int k) {
    constexpr int kSamplesPerAxis = 8;
    double maxAbsError = 0.0;
    double maxRelError = 0.0;

    for (int sampleRow = 0; sampleRow < kSamplesPerAxis; ++sampleRow) {
        const int row = sampleRow * (m - 1) / (kSamplesPerAxis - 1);
        for (int sampleCol = 0; sampleCol < kSamplesPerAxis; ++sampleCol) {
            const int col = sampleCol * (k - 1) / (kSamplesPerAxis - 1);
            double expected = 0.0;
            for (int inner = 0; inner < n; ++inner) {
                expected += static_cast<double>(a[row * n + inner])
                            * static_cast<double>(b[inner * k + col]);
            }

            const double actual = c[row * k + col];
            const double absError = std::abs(actual - expected);
            const double relError = absError / std::max(1.0, std::abs(expected));
            maxAbsError = std::max(maxAbsError, absError);
            maxRelError = std::max(maxRelError, relError);
        }
    }
    return {maxAbsError, maxRelError};
}

void printMatrix(
    std::string_view name,
    const std::vector<float>& matrix,
    int rows,
    int cols,
    PrintMode mode) {
    if (mode == PrintMode::None) {
        return;
    }

    const int shownRows = mode == PrintMode::Full ? rows : std::min(rows, 6);
    const int shownCols = mode == PrintMode::Full ? cols : std::min(cols, 6);
    std::cout << name << " (" << rows << " x " << cols << "):\n";
    for (int row = 0; row < shownRows; ++row) {
        for (int col = 0; col < shownCols; ++col) {
            std::cout << std::setw(11) << std::fixed << std::setprecision(5)
                      << matrix[row * cols + col] << ' ';
        }
        if (shownCols < cols) {
            std::cout << "...";
        }
        std::cout << '\n';
    }
    if (shownRows < rows) {
        std::cout << "...\n";
    }
}

RunResult runOne(
    const Options& options,
    KernelKind kernel,
    int blockSize,
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c) {
    const Timing timing = execute(
        kernel,
        a,
        b,
        c,
        options.m,
        options.n,
        options.k,
        blockSize,
        options.warmup,
        options.repeat);

    double maxAbsError = 0.0;
    double maxRelError = 0.0;
    bool verified = false;
    if (options.verify) {
        std::tie(maxAbsError, maxRelError) =
            verifySampled(a, b, c, options.m, options.n, options.k);
        verified = maxRelError <= 1.0e-3;
    }
    return {
        kernel,
        blockSize,
        timing,
        maxAbsError,
        maxRelError,
        options.verify,
        verified,
    };
}

void printCsvHeader() {
    std::cout
        << "m,n,k,kernel,block,warmup,repeat,kernel_ms,total_ms,gflops,"
           "max_sample_abs_error,max_sample_rel_error,verify\n";
}

void printResult(
    const RunResult& result,
    int m,
    int n,
    int k,
    int warmup,
    int repeat,
    bool csv) {
    const double operations = 2.0 * static_cast<double>(m) * n * k;
    const double gflops = operations / (result.timing.kernelMs * 1.0e6);
    if (csv) {
        std::cout << m << ',' << n << ',' << k << ',' << kernelName(result.kernel) << ','
                  << result.blockSize << ',' << warmup << ',' << repeat << ','
                  << std::fixed << std::setprecision(6) << result.timing.kernelMs << ','
                  << result.timing.totalMs << ',' << std::setprecision(3) << gflops << ','
                  << std::scientific << result.maxAbsError << ','
                  << result.maxRelError << ',';
        if (result.verificationPerformed) {
            std::cout << (result.verified ? "PASS" : "FAIL");
        } else {
            std::cout << "SKIP";
        }
        std::cout << '\n';
        return;
    }

    std::cout << "kernel=" << kernelName(result.kernel)
              << ", block=" << result.blockSize << "x" << result.blockSize
              << ", kernel_time=" << std::fixed << std::setprecision(4)
              << result.timing.kernelMs << " ms"
              << ", throughput=" << std::setprecision(2) << gflops << " GFLOP/s"
              << ", total_time=" << std::setprecision(4) << result.timing.totalMs << " ms";
    if (result.verificationPerformed) {
        std::cout << ", max_sample_rel_error=" << std::scientific << result.maxRelError
                  << ", verify=" << (result.verified ? "PASS" : "FAIL");
    }
    std::cout << '\n';
}

void printDeviceInfo(std::ostream& output) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    output << "GPU: " << properties.name << ", compute capability "
           << properties.major << '.' << properties.minor
           << ", global memory "
           << (properties.totalGlobalMem / (1024ULL * 1024ULL)) << " MiB\n";
}

void runSingle(Options options) {
    if (!options.dimensionsProvided) {
        std::cout << "Enter M N K: ";
        if (!(std::cin >> options.m >> options.n >> options.k)) {
            throw std::invalid_argument("failed to read M N K from stdin");
        }
    }
    validateDimensions(options.m, options.n, options.k);
    printDeviceInfo(options.csv ? std::cerr : std::cout);

    const auto a = randomMatrix(
        static_cast<std::size_t>(options.m) * options.n, options.seed);
    const auto b = randomMatrix(
        static_cast<std::size_t>(options.n) * options.k, options.seed + 1);
    std::vector<float> c(static_cast<std::size_t>(options.m) * options.k);

    printMatrix("A", a, options.m, options.n, options.printMode);
    printMatrix("B", b, options.n, options.k, options.printMode);
    if (options.csv) {
        printCsvHeader();
    }

    if (options.allKernels) {
        const std::vector<KernelKind> kernels = {
            KernelKind::Naive, KernelKind::Tiled, KernelKind::Coarsened};
        for (KernelKind kernel : kernels) {
            const RunResult result =
                runOne(options, kernel, options.blockSize, a, b, c);
            printResult(
                result,
                options.m,
                options.n,
                options.k,
                options.warmup,
                options.repeat,
                options.csv);
            if (options.verify && !result.verified) {
                throw std::runtime_error(
                    "verification failed for kernel " + std::string(kernelName(kernel)));
            }
        }
    } else {
        const RunResult result =
            runOne(options, options.kernel, options.blockSize, a, b, c);
        printResult(
            result,
            options.m,
            options.n,
            options.k,
            options.warmup,
            options.repeat,
            options.csv);
        if (options.verify && !result.verified) {
            throw std::runtime_error("verification failed");
        }
    }

    printMatrix("C", c, options.m, options.k, options.printMode);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        runSingle(options);
        CUDA_CHECK(cudaDeviceReset());
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
