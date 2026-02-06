#include <cassert>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <npu-json/npu/pipeline.hpp>
#include <npu-json/npu/chunk-index.hpp>
#include <npu-json/util/debug.hpp>
#include <npu-json/util/files.hpp>
#include <npu-json/util/strings.hpp>
#include <npu-json/engine.hpp>

void write_expected_index(uint64_t *index, std::string &expected_str) {
  assert(expected_str.size() % 64 == 0);
  assert(expected_str.size() / 8 < npu::CHUNK_BIT_INDEX_SIZE);

  auto index_offset = 0;
  for (size_t i = 0; i < expected_str.size(); i += 64) {
    auto vector_str = expected_str.substr(i, 64);
    std::reverse(vector_str.begin(), vector_str.end());
    auto vector_bitset = std::bitset<64>(vector_str);
    index[index_offset] = vector_bitset.to_ullong();
    index_offset++;
  }
}

bool test_string_index(const char *test) {
  // Setup buffers
  auto chunk = new char[Engine::CHUNK_SIZE] {};
  constexpr auto index_size = npu::CHUNK_BIT_INDEX_SIZE / 8;
  auto expected_index = new uint64_t[index_size] {};

  // Read in test file
  auto test_path = "test/npu/fixtures/" + std::string(test) + ".txt";
  std::string data = util::load_file_content(test_path);
  auto newline_idx = data.find('\n');
  auto json = data.substr(0, newline_idx);
  auto expected_str = data.substr(newline_idx);
  expected_str.erase(std::remove(expected_str.begin(), expected_str.end(), '\n'), expected_str.cend());
  expected_str = util::pad_to_multiple(expected_str, 64, '0');

  // Copy JSON part from testfile
  assert(json.length() < Engine::CHUNK_SIZE);
  memset(chunk, ' ', Engine::CHUNK_SIZE);
  memcpy(chunk, json.c_str(), json.length());

  // Setup kernel and indexer
  std::cout << "Setting up kernel" << std::endl;
  auto kernel = std::make_unique<npu::Kernel>(json);
  std::cout << "Setting up pipeline indexer" << std::endl;
  auto indexer = std::make_shared<npu::PipelinedIndexer>(*kernel, json);
  std::cout << "Finished setting up" << std::endl;

  // Copy expected index part from testfile
  write_expected_index(expected_index, expected_str);

  auto chunk_index = new npu::ChunkIndex();

  // Run indexer
  indexer->index_chunk(chunk_index, []{});
  indexer->wait_for_last_chunk();

  // Expect results
  auto result = false;
  for (size_t i = 0; i < index_size; i++) {
    auto eq = chunk_index->string_index[i] == expected_index[i];
    if (!eq) {
      std::cout << "Error at index " << i << "\n";
      std::cout << "Expecting: " << std::hex << expected_index[i] << "\n";
      std::cout << "Getting:   " << std::hex << chunk_index->string_index[i] << std::endl;
      result = true;
      break;
    }
  }

  // Cleanup
  delete chunk_index;
  delete[] chunk;

  return result;
}

void print_test_results(const char *test, bool status) {
  if (status) {
    std::cout << test << ": FAIL!" << std::endl;
  } else {
    std::cout << test << ": PASS!" << std::endl;
  }
}

int main(int argc, const char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage: ./test [xclbin] [insts]" << std::endl;
    return -1;
  }

  auto global_status = 0;

  auto status_test_string_index = test_string_index("basic");
  print_test_results("basic", status_test_string_index);
  global_status = global_status || status_test_string_index;

  return global_status;
}
