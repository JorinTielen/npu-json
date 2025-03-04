#include <bitset>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <iostream>

#include <npu-json/util/files.hpp>
#include <npu-json/util/xrt.hpp>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: ./nj [json]" << std::endl;
    return -1;
  }

  // Initialize NPU
  auto [device, kernel] = util::init_npu("build/src/aie/json.xclbin");

  // Read in JSON file
  std::string data = util::load_file_content(argv[1]);
  data = util::pad_to_multiple(data, 64);


  constexpr size_t data_chunk_size = 10 * 1024 * 1024;
  constexpr size_t index_chunk_size = data_chunk_size / 8;

  std::string mask;
  mask.reserve(index_chunk_size);

  // Setup instruction buffer
  auto instr_v = util::load_instr_sequence("build/src/aie/json-insts.txt");
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));

  // Setup input/output buffers
  auto bo_data = xrt::bo(device, data_chunk_size * sizeof(uint8_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_index = xrt::bo(device, index_chunk_size * sizeof(uint8_t),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  // Copy instructions to buffer
  void *buf_instr = bo_instr.map<void *>();
  memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

  // Copy input into buffer
  uint8_t *buf_data = bo_data.map<uint8_t *>();
  memcpy(buf_data, data.data(), data_chunk_size * sizeof(uint8_t));

  // Zero out buffer bo_out
  uint64_t *buf_index = bo_index.map<uint64_t *>();
  memset(buf_index, 0, index_chunk_size * sizeof(uint8_t));

  // sync host to device memories
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_data.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_index.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Run benchmark
  constexpr size_t WARMUP_ITERS = 3;
  constexpr size_t BENCH_ITERS = 10;

  for (size_t i = 0; i < WARMUP_ITERS; i++) {
    auto run = kernel(3, bo_instr, instr_v.size(), bo_data, bo_index);
    run.wait();
  }

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < BENCH_ITERS; i++) {
    auto run = kernel(3, bo_instr, instr_v.size(), bo_data, bo_index);
    run.wait();
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto avg_runtime = (end - start) / BENCH_ITERS;

  auto seconds = std::chrono::duration<double>(avg_runtime).count();
  double gigabytes = (double)data_chunk_size / 1000 / 1000 / 1000;
  std::cout << "size: " << gigabytes << "GB" << std::endl;
  std::cout << "GB/s: " << gigabytes / seconds << std::endl;

  bo_index.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  memcpy(mask.data(), buf_index, index_chunk_size * sizeof(uint8_t));

  // Print output preview
  auto mask_bitset = std::bitset<60>(reinterpret_cast<const uint64_t *>(mask.c_str())[0]);
  auto mask_bitset_str = mask_bitset.to_string();
  std::reverse(mask_bitset_str.begin(), mask_bitset_str.end());
  std::cout << std::endl << "data (first 60 bytes):" << std::endl;
  std::cout << "input: |" << data.substr(0, 60) << "|" << std::endl;
  std::cout << "mask:  |" << mask_bitset_str << "|" << std::endl;

  return 0;
}
