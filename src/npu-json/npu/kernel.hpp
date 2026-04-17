#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <npu-json/util/tracer.hpp>
#include <npu-json/util/xrt.hpp>

namespace npu {

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

// Class managing the XRT runtime of the JSON indexing NPU kernel.
class Kernel {
public:
  Kernel(std::string_view json);

  Kernel(const Kernel&) = delete;
  Kernel& operator=(const Kernel&) = delete;

  void call(ChunkIndex *index, size_t chunk_idx, std::function<void()> callback);

  void wait_for_previous();
private:
  xrt::bo instr;
  size_t instr_size;
  xrt::kernel kernel;
  // std::vector<uint8_t> quote_map;
  // std::vector<uint8_t> slash_map;

  std::optional<RunHandle> previous_run;
  bool previous_string_carry = false;
  bool previous_escape_carry = false;

  xrt::bo json_data_input;
  std::vector<xrt::bo> json_chunk_inputs;
  uint8_t *json_data_map = nullptr;

  size_t current = 0;
  KernelBuffer string_buffers[2];
  KernelBuffer structural_buffers[2];
  uint8_t *string_input_maps[2] = { nullptr, nullptr };
  uint64_t *string_output_maps[2] = { nullptr, nullptr };
  uint64_t *structural_output_maps[2] = { nullptr, nullptr };

  util::trace_id trace;

  void prepare_kernel_input(const char *chunk, ChunkIndex &index, bool first_escape_carry, size_t buffer);
  void read_kernel_output(ChunkIndex &index, bool first_string_carry, size_t chunk_idx);
  // void initialize_maps(std::string_view &json);
};

// Outside for testing purposes
void construct_escape_carry_index(const char *chunk, ChunkIndex &index, bool first_escape_carry);

} // namespace npu
