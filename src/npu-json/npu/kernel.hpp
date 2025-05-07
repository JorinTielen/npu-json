#pragma once

#include <npu-json/util/xrt.hpp>

namespace npu {

// Class managing the XRT runtime of the JSON indexing NPU kernel.
class Kernel {
public:
  Kernel(std::string &json);

  Kernel(const Kernel&) = delete;
  Kernel& operator=(const Kernel&) = delete;

  void call(const char * chunk, ChunkIndex &index, bool first_string_carry, size_t chunk_idx);
private:
  xrt::bo instr;
  size_t instr_size;
  xrt::kernel kernel;

  struct StringIndex {
    xrt::bo input;
    xrt::bo output;
  } string_index;

  struct StructuralIndexBuffers {
    xrt::bo input;
    xrt::bo output;
  } structural_index;
};

} // namespace npu
