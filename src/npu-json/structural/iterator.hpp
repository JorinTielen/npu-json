#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <memory>

#include <npu-json/npu/indexer.hpp>
#include <npu-json/engine.hpp>

namespace structural {


class Iterator {
public:
  Iterator(std::string &json);
  // Gives the next structural character
  StructuralCharacter* get_next_structural_character();
private:
  size_t chunk_idx = 0;
  bool chunk_carry_escape = false;
  bool chunk_carry_string = false;

  std::unique_ptr<npu::StructuralIndexer> indexer;

  std::string &json;
  std::unique_ptr<std::array<uint8_t, Engine::CHUNK_SIZE>> chunk;
  std::shared_ptr<npu::StructuralIndex> structural_index;

  void switch_to_next_chunk();
};

} // namespace structural
