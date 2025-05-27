#pragma once

#include <atomic>
#include <array>
#include <string>
#include <memory>

#include <npu-json/npu/chunk-index.hpp>
#include <npu-json/npu/kernel.hpp>
#include <npu-json/npu/queue.hpp>
#include <npu-json/engine.hpp>

namespace npu {

constexpr const std::size_t QUEUE_DEPTH = 4;

using ChunkIndexQueue = Queue<ChunkIndex, QUEUE_DEPTH>;

// Structural iterator that indexes the JSON on the NPU in a background thread,
// allowing for pipelined execution with the JSONPath automaton running on the CPU.
class PipelinedIterator {
public:
  PipelinedIterator(std::string &json);

  void setup(const std::string *const json);
  void reset();

  // Gives a pointer to the next structural character, and consumes it.
  uint32_t* get_next_structural_character();

  uint32_t* get_chunk_structural_index_end_ptr();
  void set_chunk_structural_pos(uint32_t *pos);
private:
  const std::string *json = nullptr;

  ChunkIndex *index = nullptr;

  std::unique_ptr<std::thread> indexer_thread;
  std::unique_ptr<ChunkIndexQueue> index_queue;
  std::unique_ptr<Kernel> kernel;

  std::size_t chunk_idx = 0;
  std::size_t current_pos_in_chunk = 0;

  bool switch_to_next_chunk();
  uint32_t* get_next_structural_character_in_chunk();
};

// New implementation of the indexer, aiming to keep the NPU busy 100% of the time
// to maximise throughput, by utilizing ping-pong (double) buffering when
// preparing the input/output of the NPU kernels.
class PipelinedIndexer {
public:
  PipelinedIndexer(Kernel &kernel, const std::string *const json)
    : kernel(kernel), json(json) {}

  void index_chunk(ChunkIndex *chunk_index, std::function<void()> callback);

  void wait_for_last_chunk();

  bool is_at_end();
private:
  Kernel &kernel;
  const std::string *const json;

  std::size_t chunk_idx = 0;
  bool chunk_carry_escape = false;
  bool chunk_carry_string = false;

  void construct_escape_carry_index(const char *chunk, ChunkIndex &index);
};

} // namespace npu
