#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

#include <npu-json/npu/chunk-index.hpp>
#include <npu-json/util/tracer.hpp>

#ifndef NPU_JSON_CPU_BACKEND
#include <npu-json/util/xrt.hpp>
#endif

namespace npu {

#ifndef NPU_JSON_CPU_BACKEND
struct RunHandle {
  xrt::run handle;
  ChunkIndex *index;
  size_t chunk_idx;
  std::function<void()> callback;
};

struct KernelBuffer {
  xrt::bo input;
  xrt::bo output;
};
#endif

class Kernel {
public:
  Kernel(std::string_view json);

  Kernel(const Kernel&) = delete;
  Kernel& operator=(const Kernel&) = delete;

  void call(ChunkIndex *index, size_t chunk_idx, std::function<void()> callback);
  void wait_for_previous();

  size_t get_prefix_size() const { return prefix_size; }

private:
  static constexpr size_t PAGE_ALIGN = 4096;

  size_t prefix_size = 0;

#ifdef NPU_JSON_CPU_BACKEND
  std::vector<uint8_t> json_data;
  bool previous_string_carry = false;
  bool previous_escape_carry = false;

  void construct_combined_index(const char *chunk, ChunkIndex &index, bool first_escape_carry, bool first_string_carry, size_t chunk_idx);
#else
  const char *prefix_start = nullptr;
  const char *aligned_start = nullptr;
  size_t aligned_length = 0;
  size_t full_chunk_count = 0;
  bool has_tail = false;

  xrt::bo instr;
  size_t instr_size;
  xrt::kernel kernel;

  std::optional<RunHandle> previous_run;
  bool previous_string_carry = false;
  bool previous_escape_carry = false;

  xrt::bo json_data_input;
  std::vector<xrt::bo> json_chunk_inputs;
  uint8_t *json_data_map = nullptr;

  xrt::bo tail_input;
  uint8_t *tail_map = nullptr;

  size_t current = 0;
  KernelBuffer string_buffers[2];
  KernelBuffer structural_buffers[2];
  uint8_t *string_input_maps[2] = { nullptr, nullptr };
  uint64_t *string_output_maps[2] = { nullptr, nullptr };
  uint64_t *structural_output_maps[2] = { nullptr, nullptr };

  void prepare_kernel_input(const char *chunk, ChunkIndex &index, bool first_escape_carry, size_t buffer);
  void read_kernel_output(ChunkIndex &index, bool first_string_carry, size_t chunk_idx);
#endif

  util::trace_id trace;
};

void construct_escape_carry_index(const char *chunk, ChunkIndex &index, bool first_escape_carry);

} // namespace npu
