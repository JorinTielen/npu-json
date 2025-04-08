#include <cstring>
#include <stdexcept>

#include <npu-json/structural/iterator.hpp>
#include <npu-json/options.hpp>

namespace structural {

Iterator::Iterator(std::string &json) : json(json) {
  indexer = std::make_unique<npu::StructuralIndexer>(XCLBIN_PATH, INSTS_PATH, true);
  chunk = std::make_unique<std::array<uint8_t, Engine::CHUNK_SIZE>>();
}

std::optional<StructuralCharacter> Iterator::get_next_structural_character() {
  // For the first chunk we need to load it in first.
  if (chunk_idx == 0) switch_to_next_chunk();

  // Return next structural character in the current chunk if there is one.
  auto possible_structural = structural_index->get_next_structural_character();
  if (possible_structural.has_value()) return possible_structural;

  // Reached the end of the input, no more structurals to return.
  if (chunk_idx >= json.length()) return std::optional<StructuralCharacter>();

  // Ran out of structurals in the current chunk, index the next one.
  switch_to_next_chunk();

  // Return structural from next chunk.
  return structural_index->get_next_structural_character();
}

void Iterator::switch_to_next_chunk() {
  if (chunk_idx >= json.length()) throw std::logic_error("Iterator passed the end of input");

  // Prepare the chunk for indexing
  // TODO: Avoid double copy
  auto remaining_length = json.length() - chunk_idx;
  auto padding_needed = remaining_length < Engine::CHUNK_SIZE;
  auto n = padding_needed ? remaining_length : Engine::CHUNK_SIZE;
  memcpy(chunk->data(), json.c_str() + chunk_idx, n);
  if (padding_needed) {
    // Pad with spaces at end
    memset(chunk->data() + n, ' ', Engine::CHUNK_SIZE - n);
  }

  // Index the new current chunk
  auto chunk_data = reinterpret_cast<const char *>(chunk->data());
  structural_index = indexer->construct_structural_index(
    chunk_data,
    chunk_carry_escape,
    chunk_carry_string
  );

  // Keep track of state between chunks
  chunk_carry_escape = structural_index->ends_with_escape();
  chunk_carry_string = structural_index->ends_in_string();

  chunk_idx += Engine::CHUNK_SIZE;
}

} // namespace structural
