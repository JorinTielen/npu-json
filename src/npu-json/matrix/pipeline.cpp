#include <npu-json/matrix/pipeline.hpp>
#include <npu-json/util/debug.hpp>
#include <npu-json/util/tracer.hpp>

namespace matrix {

MatrixPipeline::MatrixPipeline(std::string_view json)
  : json_(json)
  , kernel_(std::make_unique<MatrixKernel>(json)) {}

void MatrixPipeline::setup(std::string_view json) {
  json_ = json;
  chunk_idx_ = 0;
  current_pos_ = 0;
  has_more_chunks_ = false;
  current_index_ = {};
  kernel_ = std::make_unique<MatrixKernel>(json_);
}

void MatrixPipeline::reset() {
  json_ = "";
  current_index_ = {};
  chunk_idx_ = 0;
  current_pos_ = 0;
  has_more_chunks_ = false;
}

bool MatrixPipeline::load_next_chunk() {
  if (chunk_idx_ >= json_.length()) {
    has_more_chunks_ = false;
    return false;
  }

  kernel_->call(&current_index_, chunk_idx_, []{});
  chunk_idx_ += Engine::CHUNK_SIZE;
  current_pos_ = 0;
  has_more_chunks_ = true;
  return true;
}

uint32_t* MatrixPipeline::get_next_structural_character() {
  if (!has_more_chunks_) {
    if (!load_next_chunk()) {
      return nullptr;
    }
  }

  if (current_pos_ < current_index_.block.structural_characters_count) {
    auto ptr = &current_index_.block.structural_characters[current_pos_];
    current_pos_++;
    return ptr;
  }

  if (!load_next_chunk()) {
    return nullptr;
  }

  if (current_pos_ < current_index_.block.structural_characters_count) {
    auto ptr = &current_index_.block.structural_characters[current_pos_];
    current_pos_++;
    return ptr;
  }

  return nullptr;
}

uint32_t* MatrixPipeline::get_chunk_structural_index_end_ptr() {
  return &current_index_.block.structural_characters[current_index_.block.structural_characters_count];
}

void MatrixPipeline::set_chunk_structural_pos(uint32_t *pos) {
  current_pos_ = pos + 1 - current_index_.block.structural_characters.data();
}

}