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

#include "../test_utils.hpp"

constexpr int VECTOR_SIZE = 50 * 1000 * 1024;
constexpr size_t BENCH_ITERS = 100;

int main(int argc, const char *argv[]) {
  // if (argc < 3) {
  //   std::cout << "Usage: ./test [xclbin] [insts]" << std::endl;
  //   return -1;
  // }

  // Initialize NPU
  std::cout << "Intializing NPU..." << std::endl;
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  auto xclbin = xrt::xclbin("build/multi_kernel.xclbin");

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_sequence("build/insts.txt");

  // Get the kernel from the xclbin
  std::cout << "Checking kernels in xclbin..." << std::endl;
  auto xkernels = xclbin.get_kernels();
  auto xkernel0 = *std::find_if(xkernels.begin(), xkernels.end(),
                                [](xrt::xclbin::kernel &k) {
                                  auto name = k.get_name();
                                  std::cout << "Name: " << name << std::endl;
                                  return name == "VECTORSCALARADD";
                                });
  auto kernelName0 = xkernel0.get_name();
  auto xkernel1 = *std::find_if(xkernels.begin(), xkernels.end(),
                                [](xrt::xclbin::kernel &k) {
                                  auto name = k.get_name();
                                  std::cout << "Name: " << name << std::endl;
                                  return name == "VECTORSCALARMUL";
                                });
  auto kernelName1 = xkernel1.get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());

  auto kernel0 = xrt::kernel(context, kernelName0);

  std::cout << "Setting up buffers..." << std::endl;
  auto bo0_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                           XCL_BO_FLAGS_CACHEABLE, kernel0.group_id(1));

  auto bo0_in  = xrt::bo(device, VECTOR_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel0.group_id(3));
  auto bo0_out = xrt::bo(device, VECTOR_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel0.group_id(5));

  auto kernel1 = xrt::kernel(context, kernelName1);

  auto bo1_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                           XCL_BO_FLAGS_CACHEABLE, kernel1.group_id(1));
  auto bo1_in  = xrt::bo(device, VECTOR_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(3));
  auto bo1_out = xrt::bo(device, VECTOR_SIZE* sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(5));

  std::cout << "Writing data into buffer objects..." << std::endl;
  uint32_t *buf_in = bo0_in.map<uint32_t *>();
  std::vector<uint32_t> src_vec;
  for (int i = 0; i < VECTOR_SIZE; i++)
    src_vec.push_back(i);
  memcpy(buf_in, src_vec.data(), (src_vec.size() * sizeof(uint32_t)));

  std::cout << "input (preview): ";
  for (size_t i = 0; i < 16; i++) {
    std::cout << src_vec[i] << " ";
  }
  std::cout << std::endl;

  void *buf0_instr = bo0_instr.map<void *>();
  void *buf1_instr = bo1_instr.map<void *>();
  memcpy(buf0_instr, instr_v.data(), instr_v.size() * sizeof(int));
  memcpy(buf1_instr, instr_v.data(), instr_v.size() * sizeof(int));

  bo0_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo1_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo0_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < BENCH_ITERS; i++) {
    std::cout << "... performing bench iteration: " << i << std::endl;
    auto run0 =
        kernel0(opcode, bo0_instr, instr_v.size(), bo0_in, bo0_out);
    run0.wait();

    bo0_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // copy kernel0 output to kernel1 input
    // auto buf0_out = bo0_out.map<uint32_t *>();
    // auto buf1_in = bo1_in.map<uint32_t *>();
    // memcpy(buf1_in, buf0_out, VECTOR_SIZE * sizeof(uint32_t));
    // bo1_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run1 =
        kernel1(opcode, bo1_instr, instr_v.size(), bo0_out, bo1_out);
    run1.wait();
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto avg_runtime = (end - start) / BENCH_ITERS;

  std::cout << "results:" << std::endl;
  auto seconds = std::chrono::duration<double>(avg_runtime).count();
  double gigabytes = (double)src_vec.size() * 4 / 1000 / 1000 / 1000;
  std::cout << "size: " << gigabytes << "GB" << std::endl;
  std::cout << "GB/s: " << gigabytes / seconds << std::endl;


  bo1_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  auto buf_out = bo1_out.map<uint32_t *>();

  std::cout << "output (preview): ";
  for (size_t i = 0; i < 16; i++) {
    std::cout << buf_out[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Checking all results..." << std::endl;
  bool valid = true;

  for (size_t i = 0; i < VECTOR_SIZE; i++) {
    if (buf_out[i] != (i + 1) * 3) {
      valid = false;
    }
  }

  if (valid) {
    std::cout << "PASS!" << std::endl;
  } else {
    std::cout << "FAIL!" << std::endl;
  }

  return 0;
}
