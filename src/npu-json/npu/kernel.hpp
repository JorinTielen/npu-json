#pragma once

#include <cstdint>
#include <optional>

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
  Kernel(std::string &json);

  Kernel(const Kernel&) = delete;
  Kernel& operator=(const Kernel&) = delete;

  void call(ChunkIndex *index, size_t chunk_idx, std::function<void()> callback);

  void wait_for_previous();
private:
  xrt::bo instr;
  size_t instr_size;
  xrt::kernel kernel;

  std::optional<RunHandle> previous_run;
  bool previous_string_carry = false;
  bool previous_escape_carry = false;

  xrt::bo json_data_input;

  size_t current = 0;
  KernelBuffer string_buffers[2];
  KernelBuffer structural_buffers[2];

  util::trace_id trace;

  void prepare_kernel_input(const char *chunk, ChunkIndex &index, bool first_escape_carry, size_t buffer);
  void read_kernel_output(ChunkIndex &index, bool first_string_carry, size_t chunk_idx);

  void construct_escape_carry_index(const char *chunk, ChunkIndex &index, bool first_escape_carry);
};

} // namespace npu
