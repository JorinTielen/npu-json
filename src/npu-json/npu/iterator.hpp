#pragma once

#include <cstdint>
#include <memory>

namespace npu {

class StructuralIterator {
public:
  virtual ~StructuralIterator() = default;

  virtual void setup(std::string_view json) = 0;
  virtual void reset() = 0;
  virtual uint32_t* get_next_structural_character() = 0;
  virtual uint32_t* get_chunk_structural_index_end_ptr() = 0;
  virtual void set_chunk_structural_pos(uint32_t *pos) = 0;
};

}