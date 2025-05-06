#include <cstring>

#include <npu-json/npu/pipeline.hpp>
#include <npu-json/util/debug.hpp>

namespace npu {

// Main function of the indexer thread
static void run_indexer(
    Kernel * const kernel, const std::string *const json,
    ChunkIndexQueue *const index_queue) {
  PipelinedIndexer indexer(*kernel, json);

  while (!indexer.is_at_end()) {
    auto index = index_queue->reserve_write_space();
    indexer.index_chunk(*index);
    index_queue->release_write_space(index);
  }
}

PipelinedIterator::PipelinedIterator()
  : index_queue(std::make_unique<ChunkIndexQueue>())
  , kernel(std::make_unique<Kernel>()) {}

void PipelinedIterator::setup(const std::string *const json) {
  this->json = json;

  indexer_thread = std::make_unique<std::thread>([this] {
    run_indexer(
      this->kernel.get(),
      this->json,
      this->index_queue.get()
    );
  });

  indexer_thread->detach();
}

void PipelinedIterator::reset() {
  this->json = nullptr;
  this->index = nullptr;
  indexer_thread.reset();
  index_queue->reset();

  chunk_idx = 0;
  current_pos_in_chunk = 0;
}

bool PipelinedIterator::switch_to_next_chunk() {
  if (index != nullptr) index_queue->release_token(index);

  index = nullptr;

  if (chunk_idx >= json->length()) return false;

  index = index_queue->claim_read_token();

  chunk_idx += Engine::CHUNK_SIZE;
  current_pos_in_chunk = 0;

  return true;
}

StructuralCharacter* PipelinedIterator::get_next_structural_character() {
  if (index == nullptr) switch_to_next_chunk();

  // Return potential next structural character in the current chunk if there is one.
  auto potential_structural = get_next_structural_character_in_chunk();
  if (potential_structural != nullptr) {
    assert(potential_structural->c == json->at(potential_structural->pos));
    return potential_structural;
  }

  // Switch to the next chunk if there are none left in the current chunk.
  if (!switch_to_next_chunk()) {
    // No next chunk, end of input.
    return nullptr;
  }

  // Return structural from next chunk.
  auto next_potential_structural = get_next_structural_character_in_chunk();
  assert(next_potential_structural->c == json->at(next_potential_structural->pos));
  return next_potential_structural;
}

StructuralCharacter* PipelinedIterator::get_next_structural_character_in_chunk() {
  if (current_pos_in_chunk < index->structurals_count) {
    auto ptr = &index->structural_characters[current_pos_in_chunk];
    current_pos_in_chunk++;
    return ptr;
  }

  return nullptr;
}

void PipelinedIndexer::construct_escape_carry_index(const char *chunk, ChunkIndex &index) {
  index.escape_carry_index[0] = chunk_carry_escape;
  for (size_t i = 1; i <= Engine::CHUNK_SIZE / Engine::BLOCK_SIZE; i++) {
    auto is_escape_char = chunk[i * Engine::BLOCK_SIZE - 1] == '\\';
    if (!is_escape_char) {
      index.escape_carry_index[i] = false;
      continue;
    }

    auto escape_char_count = 1;
    while (chunk[(i * Engine::BLOCK_SIZE - 1) - escape_char_count] == '\\') {
      is_escape_char = !is_escape_char;
      escape_char_count++;
    }

    index.escape_carry_index[i] = is_escape_char;
  }
}

static std::array<char, Engine::CHUNK_SIZE> backup_chunk;

void PipelinedIndexer::index_chunk(ChunkIndex &index) {
  if (chunk_idx >= json->length()) {
    throw std::logic_error("Attempted to index past end of JSON");
  }

  // For the last chunk we will need to pad with spaces.
  auto remaining_length = json->length() - chunk_idx;
  auto padding_needed = remaining_length < Engine::CHUNK_SIZE;
  auto n = padding_needed ? remaining_length : Engine::CHUNK_SIZE;
  if (padding_needed) {
    // Pad with spaces at end
    memcpy(backup_chunk.data(), json->c_str() + chunk_idx, n);
    memset(backup_chunk.data() + n, ' ', Engine::CHUNK_SIZE - n);
  }

  const char *chunk = padding_needed ? backup_chunk.data() : json->c_str() + chunk_idx;

  // Prepare escape carries for string index
  construct_escape_carry_index(chunk, index);

  // Perform string index and structural index on NPU
  kernel.call(chunk, index, chunk_carry_string, chunk_idx);

  // Keep track of state between chunks
  chunk_carry_escape = index.ends_with_escape();
  chunk_carry_string = index.ends_in_string();

  chunk_idx += Engine::CHUNK_SIZE;
}

bool PipelinedIndexer::is_at_end() {
  return chunk_idx >= json->length();
}

} // namespace npu
