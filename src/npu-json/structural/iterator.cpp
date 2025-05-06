#include <cstring>
#include <stdexcept>

#include <npu-json/structural/iterator.hpp>
#include <npu-json/util/tracer.hpp>

namespace structural {

Iterator::Iterator(std::string &json) : json(json) {
  indexer = std::make_unique<npu::StructuralIndexer>(true);
  chunk = std::make_unique<std::array<uint8_t, Engine::CHUNK_SIZE>>();
}

StructuralCharacter* Iterator::get_next_structural_character() {
  auto& tracer = util::Tracer::get_instance();
  static util::trace_id automaton_trace;

  // For the first chunk we need to load it in first.
  if (chunk_idx == 0) {
    switch_to_next_chunk();
    automaton_trace = tracer.start_trace("automaton");
  }

  // Return next structural character in the current chunk if there is one.
  auto possible_structural = structural_index->get_next_structural_character();
  if (possible_structural != nullptr) {
    return possible_structural;
  }

  // Reached the end of the input, no more structurals to return.
  if (chunk_idx >= json.length()) {
    // Finish the last trace
    tracer.finish_trace(automaton_trace);
    return nullptr;
  }

  // If we have a trace, finish it so we can start a new one after indexing the next chunk.
  if (automaton_trace != 0) {
    tracer.finish_trace(automaton_trace);
  }

  // Ran out of structurals in the current chunk, index the next one.
  switch_to_next_chunk();

  automaton_trace = tracer.start_trace("automaton");

  // Return structural from next chunk.
  return structural_index->get_next_structural_character();
}

void Iterator::switch_to_next_chunk() {
  if (chunk_idx >= json.length()) throw std::logic_error("Iterator passed the end of input");

  // Prepare the chunk for indexing
  auto remaining_length = json.length() - chunk_idx;
  auto padding_needed = remaining_length < Engine::CHUNK_SIZE;
  auto n = padding_needed ? remaining_length : Engine::CHUNK_SIZE;
  if (padding_needed) {
    // Pad with spaces at end
    memcpy(chunk->data(), json.c_str() + chunk_idx, n);
    memset(chunk->data() + n, ' ', Engine::CHUNK_SIZE - n);
  }


  // Index the new current chunk
  auto chunk_data = padding_needed
    ? reinterpret_cast<const char *>(chunk->data())
    : json.c_str() + chunk_idx;
  structural_index = indexer->construct_structural_index(
    chunk_data,
    chunk_carry_escape,
    chunk_carry_string,
    chunk_idx
  );


  // Keep track of state between chunks
  chunk_carry_escape = structural_index->ends_with_escape();
  chunk_carry_string = structural_index->ends_in_string();

  chunk_idx += Engine::CHUNK_SIZE;
}

} // namespace structural
