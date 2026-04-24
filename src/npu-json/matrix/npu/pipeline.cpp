#include <cstring>

#include <npu-json/matrix/npu/pipeline.hpp>
#include <npu-json/util/debug.hpp>
#include <npu-json/util/tracer.hpp>

namespace matrix::npu {

static void run_matrix_indexer(
  NPUMatrixKernel *const kernel, const std::string_view json,
  ChunkIndexQueue *const index_queue
) {
  NPUMatrixIndexer indexer(*kernel, json);

  while (!indexer->is_at_end()) {
    auto index = index_queue->reserve_write_space();
    indexer.index_chunk(index, [index_queue, index]{
      index_queue->release_write_space(index);
    });
  }

  indexer.wait_for_last_chunk();
}

NPUMatrixPipeline::NPUMatrixPipeline(std::string_view json)
  : index_queue(std::make_unique<ChunkIndexQueue>())
  , kernel(std::make_unique<NPUMatrixKernel>(json)) {}

void NPUMatrixPipeline::setup(std::string_view json) {
  this->json_ = json;

  indexer_thread = std::make_unique<std::thread>([this] {
    run_matrix_indexer(
      this->kernel.get(),
      this->json_,
      this->index_queue.get()
    );
  });

  indexer_thread->detach();
}

void NPUMatrixPipeline::reset() {
  this->json_ = "";
  index = nullptr;
  indexer_thread.reset();
  index_queue->reset();
  chunk_idx = 0;
  current_pos_in_block = 0;
}

bool NPUMatrixPipeline::switch_to_next_chunk() {
  auto& tracer = util::Tracer::get_instance();
  static util::trace_id automaton_trace;

  if (index != nullptr) {
    index_queue->release_token(index);
    if (automaton_trace) tracer.finish_trace(automaton_trace);
  }

  index = nullptr;

  if (chunk_idx >= json_.length()) return false;

  index = index_queue->claim_read_token();

  automaton_trace = tracer.start_trace("automaton_matrix_npu");

  chunk_idx += Engine::CHUNK_SIZE;
  current_pos_in_block = 0;

  return true;
}

uint32_t* NPUMatrixPipeline::get_next_structural_character() {
  if (index == nullptr) switch_to_next_chunk();

  auto potential_structural = get_next_structural_character_in_chunk();
  if (potential_structural != nullptr) {
    return potential_structural;
  }

  if (!switch_to_next_chunk()) {
    return nullptr;
  }

  return get_next_structural_character_in_chunk();
}

uint32_t* NPUMatrixPipeline::get_chunk_structural_index_end_ptr() {
  auto count = index->block.structural_characters_count;
  return &index->block.structural_characters[count];
}

void NPUMatrixPipeline::set_chunk_structural_pos(uint32_t *pos) {
  current_pos_in_block = pos + 1 - index->block.structural_characters.data();
}

uint32_t* NPUMatrixPipeline::get_next_structural_character_in_chunk() {
  auto potential_structural = get_next_structural_character_in_block();
  if (potential_structural != nullptr) {
    return potential_structural;
  }

  return nullptr;
}

uint32_t* NPUMatrixPipeline::get_next_structural_character_in_block() {
  if (current_pos_in_block < index->block.structural_characters_count) {
    auto ptr = &index->block.structural_characters[current_pos_in_block];
    current_pos_in_block++;
    return ptr;
  }

  return nullptr;
}

void NPUMatrixIndexer::index_chunk(::npu::ChunkIndex *chunk_index, std::function<void()> callback) {
  if (chunk_idx >= json.length()) {
    throw std::logic_error("Attempted to index past end of JSON");
  }

  kernel.call(chunk_index, chunk_idx, callback);

  chunk_idx += Engine::CHUNK_SIZE;
}

void NPUMatrixIndexer::wait_for_last_chunk() {
  kernel.wait_for_previous();
}

bool NPUMatrixIndexer::is_at_end() {
  return chunk_idx >= json.length();
}

}