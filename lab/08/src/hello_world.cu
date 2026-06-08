#include <cuda_runtime.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void checkCuda(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s failed: %s\n",
                     file, line, expr, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expr) checkCuda((expr), #expr, __FILE__, __LINE__)

__global__ void helloKernel() {
    printf("Hello World from Thread (%d, %d) in Block %d!\n",
           threadIdx.x, threadIdx.y, blockIdx.x);
}

bool parseInt(const char* text, int* value) {
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (text == end || *end != '\0' || parsed < INT_MIN || parsed > INT_MAX) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

void printUsage(const char* program) {
    std::cerr << "Usage: " << program << " <num_blocks> <block_dim_x> <block_dim_y>\n"
              << "All three values must be in [1, 32]. If no arguments are given,\n"
              << "the program reads the three integers from standard input.\n";
}

bool inRequiredRange(int value) {
    return value >= 1 && value <= 32;
}

}  // namespace

int main(int argc, char** argv) {
    int n = 0;
    int m = 0;
    int k = 0;

    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        printUsage(argv[0]);
        return 0;
    }

    if (argc == 4) {
        if (!parseInt(argv[1], &n) || !parseInt(argv[2], &m) || !parseInt(argv[3], &k)) {
            printUsage(argv[0]);
            return 1;
        }
    } else if (argc == 1) {
        if (!(std::cin >> n >> m >> k)) {
            printUsage(argv[0]);
            return 1;
        }
    } else {
        printUsage(argv[0]);
        return 1;
    }

    if (!inRequiredRange(n) || !inRequiredRange(m) || !inRequiredRange(k)) {
        std::cerr << "Error: num_blocks, block_dim_x, and block_dim_y must all be in [1, 32].\n";
        return 1;
    }

    std::cout << "Hello World from the host!" << std::endl;

    const dim3 grid(static_cast<unsigned int>(n));
    const dim3 block(static_cast<unsigned int>(m), static_cast<unsigned int>(k));
    helloKernel<<<grid, block>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
