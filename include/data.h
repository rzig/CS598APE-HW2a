#pragma once
#include <cstddef>
#include <cstdlib>
#include <vector>
#include "cuda_macros.h"
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * Stores data in batches, where each batch is column major,
 * and batches are stored column major. When batch size = 1,
 * this is row major, and when batch size = num cols, this
 * is column major.
 */
template <typename T> class Dataset {
public:
  Dataset(const std::vector<std::vector<T>> &input)
      : underlying_data_((T *)std::aligned_alloc(
            64, sizeof(T) * (input[0].size() * input.size()))),
        num_columns_(input[0].size()),
        num_batches_(input.size() / batch_size_) {
    size_t i = 0;
    for (size_t row_idx = 0; row_idx < input.size(); row_idx += batch_size_) {
      for (size_t column_idx = 0; column_idx < input[0].size(); column_idx++) {
        for (size_t j = 0; j < batch_size_; j++) {
          underlying_data_[i++] = input[row_idx + j][column_idx];
        }
      }
    }
    #ifdef CUDA_MODE
    size_t total_size = sizeof(T)*input.size()*input[0].size();
    CHECK_CUDA(cuMemAlloc(&device_data_, total_size));
    CHECK_CUDA(cuMemcpyHtoD(device_data_, underlying_data_, total_size));
    #endif
  }

  T *column_of_batch(size_t batch_idx, size_t column) const noexcept {
    size_t batch_data_items = batch_size_ * num_columns_;
    return &underlying_data_[batch_idx * batch_data_items +
                             column * batch_size_];
  }

  size_t num_batches() const noexcept { return num_batches_; }

  static constexpr size_t batch_size() { return batch_size_; }

  ~Dataset() {
    free(underlying_data_);
    #ifdef CUDA_MODE
      if (device_data_) {
        cuMemFree(device_data_);
      }
    #endif
}
#ifndef CUDA_MODE
  static constexpr size_t batch_size_ = 16;
#endif
#ifdef CUDA_MODE
  static constexpr size_t batch_size_ = 1; // row major
#endif
#ifdef CUDA_MODE
CUdeviceptr device_data() const {
  return device_data_;
}
#endif

private:
  T *underlying_data_;
#ifdef CUDA_MODE
CUdeviceptr device_data_;
#endif
  size_t num_batches_;
  size_t num_columns_;
};
