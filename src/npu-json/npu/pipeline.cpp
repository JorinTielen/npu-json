#include <cstring>

#include <npu-json/npu/pipeline.hpp>
#include <npu-json/util/debug.hpp>
#include <npu-json/util/tracer.hpp>

namespace npu {

// Main function of the indexer thread
static void run_indexer(
    Kernel * const kernel, const std::string *const json,
    ChunkIndexQueue *const index_queue) {
  PipelinedIndexer indexer(*kernel, json);

  while (!indexer.is_at_end()) {
    auto index = index_queue->reserve_write_space();
    indexer.index_chunk(index, [index_queue, index]{
      // Only release the write space once the callback comes back.
      // Because of ping-pong buffering, the index is not finished
      // once the `index_chunk` function returns.
      index_queue->release_write_space(index);
    });
  }

  indexer.wait_for_last_chunk();
}

PipelinedIterator::PipelinedIterator(std::string &json)
  : index_queue(std::make_unique<ChunkIndexQueue>())
  , kernel(std::make_unique<Kernel>(json)) {}

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
  current_pos_in_block = 0;
  current_block = 0;
}

bool PipelinedIterator::switch_to_next_chunk() {
  auto& tracer = util::Tracer::get_instance();
  static util::trace_id automaton_trace;

  if (index != nullptr) {
    index_queue->release_token(index);
    // Finish the trace if there is one.
    if (automaton_trace) tracer.finish_trace(automaton_trace);
  }

  index = nullptr;

  if (chunk_idx >= json->length()) return false;

  index = index_queue->claim_read_token();

  automaton_trace = tracer.start_trace("automaton");

  chunk_idx += Engine::CHUNK_SIZE;
  current_pos_in_block = 0;
  current_block = 0;

  return true;
}

uint32_t* PipelinedIterator::get_next_structural_character() {
  if (index == nullptr) switch_to_next_chunk();

  // Return potential next structural character in the current chunk if there is one.
  auto potential_structural = get_next_structural_character_in_chunk();
  if (potential_structural != nullptr) {
    return potential_structural;
  }

  // Switch to the next chunk if there are none left in the current chunk.
  if (!switch_to_next_chunk()) {
    // No next chunk, end of input.
    return nullptr;
  }

  // Return structural from next chunk.
  auto next_potential_structural = get_next_structural_character_in_chunk();
  return next_potential_structural;
}

uint32_t* PipelinedIterator::get_chunk_structural_index_end_ptr() {
  auto count = index->blocks[current_block].structural_characters_count;
  return &index->blocks[current_block].structural_characters[count];
}

void PipelinedIterator::set_chunk_structural_pos(uint32_t *pos) {
  current_pos_in_block = pos + 1 - index->blocks[current_block].structural_characters.data();
}

uint32_t* PipelinedIterator::get_next_structural_character_in_chunk() {
  auto potential_structural = get_next_structural_character_in_block();
  if (potential_structural != nullptr) {
    return potential_structural;
  }

  // Try the next block, in the slim case an entire block is empty we
  // continue trying.
  current_block++;
  current_pos_in_block = 0;
  while (current_block < StructuralCharacterBlock::BLOCKS_PER_CHUNK) {
    potential_structural = get_next_structural_character_in_block();
    if (potential_structural != nullptr) {
      return potential_structural;
    }
    current_block++;
    current_pos_in_block = 0;
  }

  return nullptr;
}

uint32_t* PipelinedIterator::get_next_structural_character_in_block() {
  if (current_pos_in_block < index->blocks[current_block].structural_characters_count) {
    auto ptr = &index->blocks[current_block].structural_characters[current_pos_in_block];
    current_pos_in_block++;
    return ptr;
  }

  return nullptr;
}

void PipelinedIndexer::index_chunk(ChunkIndex *index, std::function<void()> callback) {
  if (chunk_idx >= json->length()) {
    throw std::logic_error("Attempted to index past end of JSON");
  }

  // Perform string index and structural index on NPU
  kernel.call(index, chunk_idx, callback);

  chunk_idx += Engine::CHUNK_SIZE;
}

void PipelinedIndexer::wait_for_last_chunk() {
  kernel.wait_for_previous();
}

bool PipelinedIndexer::is_at_end() {
  return chunk_idx >= json->length();
}

} // namespace npu
