cmake_minimum_required(VERSION 3.18)

project(sysu_lab08_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)
endif()

add_executable(cuda_hello_world
    src/hello_world.cu
)

add_executable(cuda_matrix_transpose
    src/matrix_transpose.cu
)

set_target_properties(cuda_hello_world cuda_matrix_transpose
    PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
)
