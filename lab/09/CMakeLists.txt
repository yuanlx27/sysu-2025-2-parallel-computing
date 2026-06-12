cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Native code for common lab GPUs plus PTX for forward-compatible JIT.
    set(CMAKE_CUDA_ARCHITECTURES "75;80;89")
endif()

project(lab09_cuda_matmul LANGUAGES CXX CUDA)

add_executable(matmul src/matmul.cu)

target_compile_features(matmul PRIVATE cxx_std_17 cuda_std_17)

target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-Xcompiler=-Wall;-Xcompiler=-Wextra>
)
