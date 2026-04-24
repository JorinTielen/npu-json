#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

#include <npu-json/npu/chunk-index.hpp>
#include <npu-json/util/tracer.hpp>
#include <npu-json/engine.hpp>

#include <npu-json/util/xrt.hpp>

namespace matrix::npu {

constexpr size_t NPU_NUM_COLS = 8;
constexpr size_t NPU_NUM_ROWS = 2;
constexpr size_t NPU_BLOCKS_PER_COLUMN = Engine::BLOCKS_PER_CHUNK / NPU_NUM_COLS;
constexpr size_t NPU_BLOCKS_PER_ROW = NPU_BLOCKS_PER_COLUMN / NPU_NUM_ROWS;
constexpr size_t NPU_INPUT_BLOCK_SIZE = 64 + Engine::BLOCK_SIZE;

struct RunHandle {
  xrt::run handle;
  ::npu::ChunkIndex *index;
  size_t chunk_idx;
  std::function<void()> callback;
  util::trace_id npu_trace;
};

struct KernelBuffer {
  xrt::bo input;
  xrt::bo string_output;
  xrt::bo structural_output;
};

class NPUMatrixKernel {
public:
  NPUMatrixKernel(std::string_view json);

  NPUMatrixKernel(const NPUMatrixKernel&) = delete;
  NPUMatrixKernel& operator=(const NPUMatrixKernel&) = delete;

  void call(::npu::ChunkIndex *index, size_t chunk_idx, std::function<void()> callback);

  void wait_for_previous();

private:
  xrt::bo instr;
  size_t instr_size;
  xrt::kernel kernel;

  std::optional<RunHandle> previous_run;
  bool previous_string_carry = false;

  std::vector<char> padded_json;
  size_t json_length = 0;

  std::vector<bool> chunk_escape_carries;

  size_t current = 0;
  KernelBuffer buffers[2];
  uint8_t *input_maps[2] = { nullptr, nullptr };
  uint64_t *string_output_maps[2] = { nullptr, nullptr };
  uint64_t *structural_output_maps[2] = { nullptr, nullptr };

  void prepare_kernel_input(::npu::ChunkIndex &index, size_t chunk_idx, size_t buffer);
  void read_kernel_output(::npu::ChunkIndex &index, size_t chunk_idx);
};

}