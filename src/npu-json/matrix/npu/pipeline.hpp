#pragma once

#include <atomic>
#include <string>
#include <memory>

#include <npu-json/npu/chunk-index.hpp>
#include <npu-json/npu/iterator.hpp>
#include <npu-json/npu/queue.hpp>
#include <npu-json/matrix/npu/kernel.hpp>
#include <npu-json/engine.hpp>

namespace matrix::npu {

constexpr std::size_t QUEUE_DEPTH = 4;

using ChunkIndexQueue = ::npu::Queue<::npu::ChunkIndex, QUEUE_DEPTH>;

class NPUMatrixPipeline : public ::npu::StructuralIterator {
public:
  NPUMatrixPipeline(std::string_view json);

  void setup(std::string_view json) override;
  void reset() override;
  uint32_t* get_next_structural_character() override;
  uint32_t* get_chunk_structural_index_end_ptr() override;
  void set_chunk_structural_pos(uint32_t *pos) override;

private:
  std::string_view json_ = "";

  ::npu::ChunkIndex *index = nullptr;

  std::unique_ptr<std::thread> indexer_thread;
  std::unique_ptr<ChunkIndexQueue> index_queue;
  std::unique_ptr<NPUMatrixKernel> kernel;

  std::size_t chunk_idx = 0;
  std::size_t current_pos_in_block = 0;

  bool switch_to_next_chunk();
  uint32_t* get_next_structural_character_in_chunk();
  uint32_t* get_next_structural_character_in_block();
};

class NPUMatrixIndexer {
public:
  NPUMatrixIndexer(NPUMatrixKernel &kernel, const std::string_view json)
    : kernel(kernel), json(json) {}

  void index_chunk(::npu::ChunkIndex *chunk_index, std::function<void()> callback);
  void wait_for_last_chunk();
  bool is_at_end();

private:
  NPUMatrixKernel &kernel;
  const std::string_view json;
  std::size_t chunk_idx = 0;
};

}