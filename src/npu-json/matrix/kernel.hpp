#pragma once

#include <cstdint>
#include <functional>
#include <string_view>

#include <npu-json/engine.hpp>
#include <npu-json/npu/chunk-index.hpp>
#include <npu-json/npu/kernel.hpp>
#include <npu-json/matrix/ops.hpp>

namespace matrix {

class MatrixKernel {
public:
  MatrixKernel(std::string_view json);

  void call(npu::ChunkIndex *index, size_t chunk_idx, std::function<void()> callback);
  void wait_for_previous();

private:
  std::vector<uint8_t> json_data;
  Matrix weight_matrix;

  bool previous_string_carry = false;
  bool previous_escape_carry = false;

  void construct_combined_index(
    const char *chunk,
    npu::ChunkIndex &index,
    bool first_escape_carry,
    bool first_string_carry,
    size_t chunk_idx
  );
};

}