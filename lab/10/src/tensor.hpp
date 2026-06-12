#pragma once

#include "cuda_utils.cuh"

#include <cstddef>
#include <utility>

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) {
        allocate(count);
    }

    ~DeviceBuffer() {
        reset();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    void allocate(std::size_t count) {
        reset();
        count_ = count;
        if (count_ != 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), bytes()));
        }
    }

    void reset() {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
        }
        count_ = 0;
    }

    T* data() {
        return data_;
    }

    const T* data() const {
        return data_;
    }

    std::size_t size() const {
        return count_;
    }

    std::size_t bytes() const {
        return count_ * sizeof(T);
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};
