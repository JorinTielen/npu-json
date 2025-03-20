#include <cstring>
#include <iostream>
#include <variant>
#include <memory>

#include <npu-json/jsonpath/query.hpp>
#include <npu-json/npu/indexer.hpp>
#include <npu-json/util/debug.hpp>
#include <npu-json/engine.hpp>
#include <npu-json/options.hpp>

Engine::Engine() {
  indexer = std::make_unique<npu::StructuralIndexer>(XCLBIN_PATH, INSTS_PATH);
}

Engine::~Engine() {}

void Engine::run_query_on(jsonpath::Query &query, std::string &json) {
  auto chunk = new char[CHUNK_SIZE];

  bool chunk_carry_escape = false;
  bool chunk_carry_string = false;

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
    auto structural_index = indexer->construct_structural_index(chunk, chunk_carry_escape, chunk_carry_string);

    std::cout << "Indices:" << std::endl;
    print_carry_index(structural_index->escape_carry_index.data());
    std::cout << "------------- carry ----------------" << std::endl;
    print_input_and_index(chunk, structural_index->string_index.data(), (1024 * 2) / 64 - 1);
    print_input_and_index(chunk, structural_index->string_index.data(), (1024 * 2) / 64);
    std::cout << "------------- start ----------------" << std::endl;
    print_input_and_index(chunk, structural_index->string_index.data(), 0);
    std::cout << "-------------- end -----------------" << std::endl;
    print_input_and_index(chunk, structural_index->string_index.data(), CHUNK_SIZE / 64 - 1);
    std::cout << "------------------------------------" << std::endl << std::endl;

    // Keep track of state between chunks
    chunk_carry_escape = chunk[CHUNK_SIZE - 1] == '\\';
    chunk_carry_string = structural_index->ends_in_string();

    // Iterate over structural character stream
    // std::cout << "Autmaton:" << std::endl;
    // while (auto s = structural_index->get_next_structural_character()) {
    //   switch (s.value().c) {
    //     case '{':
    //       std::cout << "structural: '{'" << std::endl; break;
    //     case '}':
    //       std::cout << "structural: '}'" << std::endl; break;
    //     case '[':
    //       std::cout << "structural: '['" << std::endl; break;
    //     case ']':
    //       std::cout << "structural: ']'" << std::endl; break;
    //     case ':':
    //       std::cout << "structural: ':'" << std::endl; break;
    //     case ',':
    //       std::cout << "structural: ','" << std::endl; break;
    //   }
    // }
  }

  delete[] chunk;
}
