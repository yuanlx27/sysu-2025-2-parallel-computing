cmake_minimum_required(VERSION 3.24)

project(lab10_cuda_convolution LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# run.py overrides this with the architecture reported by nvidia-smi when possible.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures")
endif()

find_package(CUDAToolkit REQUIRED)

option(LAB10_ENABLE_CUDNN "Build the optional cuDNN convolution backend" ON)
set(LAB10_HAVE_CUDNN OFF)

if(LAB10_ENABLE_CUDNN)
    find_path(CUDNN_INCLUDE_DIR
        NAMES cudnn.h
        HINTS
            ENV CUDNN_ROOT
            ${CUDAToolkit_INCLUDE_DIRS}
        PATH_SUFFIXES include
    )
    find_library(CUDNN_LIBRARY
        NAMES cudnn
        HINTS
            ENV CUDNN_ROOT
            ${CUDAToolkit_LIBRARY_DIR}
        PATH_SUFFIXES lib lib64
    )

    if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
        set(LAB10_HAVE_CUDNN ON)
        message(STATUS "cuDNN backend enabled")
    else()
        message(WARNING "cuDNN development files were not found; the backend will be recorded as unavailable")
    endif()
endif()

add_executable(lab10_conv
    src/main.cu
    src/cpu_reference.cpp
    src/direct.cu
    src/im2col.cu
    src/gemm.cu
    src/cudnn_conv.cpp
    src/benchmark.cpp
)

target_include_directories(lab10_conv PRIVATE src)
target_link_libraries(lab10_conv PRIVATE CUDA::cudart)
target_compile_definitions(lab10_conv PRIVATE
    LAB10_HAVE_CUDNN=$<BOOL:${LAB10_HAVE_CUDNN}>
)

if(LAB10_HAVE_CUDNN)
    target_include_directories(lab10_conv PRIVATE ${CUDNN_INCLUDE_DIR})
    target_link_libraries(lab10_conv PRIVATE ${CUDNN_LIBRARY})
endif()

target_compile_options(lab10_conv PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>
)

set_target_properties(lab10_conv PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
