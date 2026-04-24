#pragma once

#include <memory>
#include <string_view>

#include <npu-json/engine.hpp>
#include <npu-json/npu/chunk-index.hpp>
#include <npu-json/npu/iterator.hpp>
#include <npu-json/matrix/kernel.hpp>

namespace matrix {

class MatrixPipeline : public npu::StructuralIterator {
public:
  MatrixPipeline(std::string_view json);

  void setup(std::string_view json) override;
  void reset() override;
  uint32_t* get_next_structural_character() override;
  uint32_t* get_chunk_structural_index_end_ptr() override;
  void set_chunk_structural_pos(uint32_t *pos) override;

private:
  std::string_view json_;
  std::unique_ptr<MatrixKernel> kernel_;
  npu::ChunkIndex current_index_{};
  size_t chunk_idx_ = 0;
  size_t current_pos_ = 0;
  bool has_more_chunks_ = false;

  bool load_next_chunk();
};

}