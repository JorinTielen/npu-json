#pragma once

#include <cstddef>
#include <vector>
#include <cassert>
#include <algorithm>

namespace matrix {

class Matrix {
public:
  Matrix() : rows_(0), cols_(0) {}

  Matrix(size_t rows, size_t cols, float init = 0.0f)
    : rows_(rows), cols_(cols), data_(rows * cols, init) {}

  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }

  float& operator()(size_t r, size_t c) {
    assert(r < rows_ && c < cols_);
    return data_[r * cols_ + c];
  }

  const float& operator()(size_t r, size_t c) const {
    assert(r < rows_ && c < cols_);
    return data_[r * cols_ + c];
  }

  float* data() { return data_.data(); }
  const float* data() const { return data_.data(); }

  void fill(float val) { std::fill(data_.begin(), data_.end(), val); }

private:
  size_t rows_;
  size_t cols_;
  std::vector<float> data_;
};

}