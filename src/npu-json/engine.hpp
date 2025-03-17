#pragma once

#include <npu-json/jsonpath/query.hpp>

class Engine {
public:
  static constexpr size_t BLOCK_SIZE = 1024;
  static constexpr size_t CHUNK_SIZE = 1024 * BLOCK_SIZE;
  void run_query_on(jsonpath::Query& query, std::string& json);
};
