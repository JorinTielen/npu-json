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
  struct StringIndex {
    void call(const char * chunk, ChunkIndex &index, bool first_string_carry);
    xrt::kernel kernel;
    xrt::bo instr;
    xrt::bo input;
    xrt::bo output;
    size_t instr_size;
  } string_index;

  struct StructuralIndexBuffers {
    void call(const char * chunk, ChunkIndex &index, size_t chunk_idx);
    xrt::kernel kernel;
    xrt::bo instr;
    xrt::bo input;
    xrt::bo output;
    size_t instr_size;
  } structural_index;
};

} // namespace npu
