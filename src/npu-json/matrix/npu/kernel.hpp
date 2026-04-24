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

struct RunHandle {
  xrt::run handle;
  ::npu::ChunkIndex *index;
  size_t chunk_idx;
  std::function<void()> callback;
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
  bool previous_escape_carry = false;

  xrt::bo json_data_input;
  std::vector<xrt::bo> json_chunk_inputs;
  uint8_t *json_data_map = nullptr;

  size_t current = 0;
  KernelBuffer buffers[2];
  uint8_t *input_maps[2] = { nullptr, nullptr };
  uint64_t *string_output_maps[2] = { nullptr, nullptr };
  uint64_t *structural_output_maps[2] = { nullptr, nullptr };

  util::trace_id trace;

  void prepare_kernel_input(const char *chunk, ::npu::ChunkIndex &index, bool first_escape_carry, size_t buffer);
  void read_kernel_output(::npu::ChunkIndex &index, bool first_string_carry, size_t chunk_idx);
};

}