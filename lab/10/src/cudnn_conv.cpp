#include "convolution.hpp"
#include "cuda_utils.cuh"
#include "tensor.hpp"

#include <sstream>
#include <stdexcept>
#include <string>

#if LAB10_HAVE_CUDNN
#include <cudnn.h>

namespace {

void check_cudnn(
    cudnnStatus_t result,
    const char* expression,
    const char* file,
    int line
) {
    if (result == CUDNN_STATUS_SUCCESS) {
        return;
    }

    std::ostringstream message;
    message << file << ':' << line << ": " << expression << " failed: "
            << cudnnGetErrorString(result);
    throw std::runtime_error(message.str());
}

#define CUDNN_CHECK(expression) \
    check_cudnn((expression), #expression, __FILE__, __LINE__)

}  // namespace
#endif

class CudnnConvolution::Impl {
public:
    explicit Impl(const ConvConfig& config) {
#if LAB10_HAVE_CUDNN
        CUDNN_CHECK(cudnnCreate(&handle_));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_descriptor_));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_descriptor_));
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_descriptor_));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convolution_descriptor_));

        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            input_descriptor_,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            1,
            kInputChannels,
            config.height,
            config.width
        ));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(
            filter_descriptor_,
            CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_NCHW,
            kOutputChannels,
            kInputChannels,
            kKernelSize,
            kKernelSize
        ));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            convolution_descriptor_,
            config.padding,
            config.padding,
            config.stride,
            config.stride,
            1,
            1,
            CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT
        ));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            output_descriptor_,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            1,
            kOutputChannels,
            config.output_height(),
            config.output_width()
        ));

        int returned = 0;
        cudnnConvolutionFwdAlgoPerf_t performances[8]{};
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
            handle_,
            input_descriptor_,
            filter_descriptor_,
            convolution_descriptor_,
            output_descriptor_,
            8,
            &returned,
            performances
        ));

        bool found = false;
        for (int index = 0; index < returned; ++index) {
            if (performances[index].status == CUDNN_STATUS_SUCCESS) {
                algorithm_ = performances[index].algo;
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("cuDNN did not return a usable algorithm");
        }

        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            handle_,
            input_descriptor_,
            filter_descriptor_,
            convolution_descriptor_,
            output_descriptor_,
            algorithm_,
            &workspace_bytes_
        ));
        if (workspace_bytes_ != 0) {
            CUDA_CHECK(cudaMalloc(&workspace_, workspace_bytes_));
        }
        available_ = true;
        status_ = "ok";
#else
        (void)config;
        status_ = "cudnn_unavailable";
#endif
    }

    ~Impl() {
#if LAB10_HAVE_CUDNN
        if (workspace_ != nullptr) {
            cudaFree(workspace_);
        }
        if (convolution_descriptor_ != nullptr) {
            cudnnDestroyConvolutionDescriptor(convolution_descriptor_);
        }
        if (filter_descriptor_ != nullptr) {
            cudnnDestroyFilterDescriptor(filter_descriptor_);
        }
        if (output_descriptor_ != nullptr) {
            cudnnDestroyTensorDescriptor(output_descriptor_);
        }
        if (input_descriptor_ != nullptr) {
            cudnnDestroyTensorDescriptor(input_descriptor_);
        }
        if (handle_ != nullptr) {
            cudnnDestroy(handle_);
        }
#endif
    }

    bool available() const {
        return available_;
    }

    const std::string& status() const {
        return status_;
    }

    void run(
        const float* input,
        const float* weights,
        float* output,
        cudaStream_t stream
    ) {
#if LAB10_HAVE_CUDNN
        CUDNN_CHECK(cudnnSetStream(handle_, stream));
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        CUDNN_CHECK(cudnnConvolutionForward(
            handle_,
            &alpha,
            input_descriptor_,
            input,
            filter_descriptor_,
            weights,
            convolution_descriptor_,
            algorithm_,
            workspace_,
            workspace_bytes_,
            &beta,
            output_descriptor_,
            output
        ));
#else
        (void)input;
        (void)weights;
        (void)output;
        (void)stream;
        throw std::runtime_error("cuDNN backend is unavailable");
#endif
    }

private:
    bool available_ = false;
    std::string status_ = "cudnn_unavailable";

#if LAB10_HAVE_CUDNN
    cudnnHandle_t handle_ = nullptr;
    cudnnTensorDescriptor_t input_descriptor_ = nullptr;
    cudnnTensorDescriptor_t output_descriptor_ = nullptr;
    cudnnFilterDescriptor_t filter_descriptor_ = nullptr;
    cudnnConvolutionDescriptor_t convolution_descriptor_ = nullptr;
    cudnnConvolutionFwdAlgo_t algorithm_{};
    void* workspace_ = nullptr;
    std::size_t workspace_bytes_ = 0;
#endif
};

CudnnConvolution::CudnnConvolution(const ConvConfig& config)
    : impl_(new Impl(config)) {}

CudnnConvolution::~CudnnConvolution() {
    delete impl_;
}

bool CudnnConvolution::available() const {
    return impl_->available();
}

const std::string& CudnnConvolution::status() const {
    return impl_->status();
}

void CudnnConvolution::run(
    const float* input,
    const float* weights,
    float* output,
    cudaStream_t stream
) {
    impl_->run(input, weights, output, stream);
}
