#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

namespace util {

inline std::pair<xrt::device, xrt::hw_context> init_npu(xrt::xclbin xclbin) {
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  return std::make_pair(device, context);
}

// inline std::vector<uint32_t> load_instr_sequence(std::string instr_path) {
//   std::ifstream instr_file(instr_path);
//   std::string line;
//   std::vector<uint32_t> instr_v;
//   while (std::getline(instr_file, line)) {
//     std::istringstream iss(line);
//     uint32_t a;
//     if (!(iss >> std::hex >> a)) {
//       throw std::runtime_error("Unable to parse instruction file");
//     }
//     instr_v.push_back(a);
//   }
//   return instr_v;
// }

inline std::vector<uint32_t> load_instr_sequence(std::string instr_path) {
  // Open file in binary mode
  std::ifstream instr_file(instr_path, std::ios::binary);
  if (!instr_file.is_open()) {
    throw std::runtime_error("Unable to open instruction file\n");
  }

  // Get the size of the file
  instr_file.seekg(0, std::ios::end);
  std::streamsize size = instr_file.tellg();
  instr_file.seekg(0, std::ios::beg);

  // Check that the file size is a multiple of 4 bytes (size of uint32_t)
  if (size % 4 != 0) {
    throw std::runtime_error("File size is not a multiple of 4 bytes\n");
  }

  // Allocate vector and read the binary data
  std::vector<uint32_t> instr_v(size / 4);
  if (!instr_file.read(reinterpret_cast<char *>(instr_v.data()), size)) {
    throw std::runtime_error("Failed to read instruction file\n");
  }
  return instr_v;
}

} // namespace util
