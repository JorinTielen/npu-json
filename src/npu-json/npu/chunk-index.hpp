#pragma once

#include <array>
#include <cstdint>

#include <npu-json/engine.hpp>

namespace npu {

// Carry indices have a single flag per block inside the chunk, plus one for the chunk-level carry.
constexpr std::size_t CHUNK_CARRY_INDEX_SIZE = Engine::CHUNK_SIZE / Engine::BLOCK_SIZE + 1;
// Bit indices have a single flag per character (byte) of the chunk.
constexpr std::size_t CHUNK_BIT_INDEX_SIZE = Engine::CHUNK_SIZE / 8;

// Structural indices for a chunk.
struct ChunkIndex {
  // The escape carry flags for each block in the chunk.
  std::array<bool, CHUNK_CARRY_INDEX_SIZE> escape_carry_index;
  // The string index of the current chunk.
  std::array<uint64_t, CHUNK_BIT_INDEX_SIZE / 8> string_index;
  // The structural character index of the current block.
  // At a maximum all characters (bytes) in the chunk are a structural.
  std::array<StructuralCharacter, Engine::CHUNK_SIZE> structural_characters;
  size_t structurals_count = 0;

  inline bool ends_in_string() {
    auto last_vector = string_index[CHUNK_BIT_INDEX_SIZE / 8 - 1];
    return (static_cast<int64_t>(last_vector) >> 63) & 1;
  }

  inline bool ends_with_escape() {
    return escape_carry_index[CHUNK_CARRY_INDEX_SIZE - 1];
  }
};

} // namespace npu
