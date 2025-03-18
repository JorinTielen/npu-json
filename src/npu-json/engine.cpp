#include <cstring>
#include <iostream>
#include <variant>

#include <npu-json/jsonpath/query.hpp>
#include <npu-json/npu/indexer.hpp>
#include <npu-json/util/debug.hpp>
#include <npu-json/engine.hpp>
#include <npu-json/options.hpp>

void Engine::run_query_on(jsonpath::Query &query, std::string &json) {
  auto chunk = new char[CHUNK_SIZE];

  for (size_t chunk_idx = 0; chunk_idx < json.length(); chunk_idx += CHUNK_SIZE) {
    // Prepare buffer for chunk to be passed to NPU
    auto remaining_length = json.length() - chunk_idx;
    auto padding_needed = remaining_length < CHUNK_SIZE;
    auto n = padding_needed ? remaining_length : CHUNK_SIZE;
    memcpy(chunk, json.c_str() + chunk_idx, n);
    if (padding_needed) {
      // Pad with spaces at end
      memset(chunk + n, ' ', CHUNK_SIZE - n);
    }

    // Create indices on NPU
    auto indexer = npu::StructuralIndexer(XCLBIN_PATH, INSTS_PATH);
    auto structural_index = indexer.construct_structural_index(chunk);

    // print_input_and_index(chunk, structural_index->string_index.data(), 1024 * 4);

    // Iterate over structural character stream
    while (auto s = structural_index->get_next_structural_character()) {
      switch (s.value().c) {
        case '{':
          std::cout << "structural: '{'" << std::endl; break;
        case '}':
          std::cout << "structural: '}'" << std::endl; break;
        case '[':
          std::cout << "structural: '['" << std::endl; break;
        case ']':
          std::cout << "structural: ']'" << std::endl; break;
        case ':':
          std::cout << "structural: ':'" << std::endl; break;
        case ',':
          std::cout << "structural: ','" << std::endl; break;
      }
    }
  }

  delete[] chunk;
}
