#pragma once

#include <memory>

#include <npu-json/jsonpath/query.hpp>

namespace npu {
class StructuralIndexer;
}

class Engine {
public:
  // Must be kept in sync with the DATA_CHUNK_SIZE AND DATA_BLOCK_SIZE in `src/aie/gen_mlir_design.py`.
  static constexpr size_t BLOCK_SIZE = 1024;
  static constexpr size_t CHUNK_SIZE = 50 * 1000 * BLOCK_SIZE;

  Engine();
  ~Engine();

  void run_query_on(jsonpath::Query& query, std::string& json);
private:
  std::unique_ptr<npu::StructuralIndexer> indexer;
};
