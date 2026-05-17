#pragma once

#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace util {

std::string load_file_content(std::string filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Could not open file: " << filename << std::endl;
    return "";
  }

  std::stringstream buffer;
  buffer << file.rdbuf();

  file.close();
  return buffer.str();
}

struct MmapBuffer {
  void* data = nullptr;
  size_t size = 0;       // actual file content size
  size_t capacity = 0;   // mmap allocation size

  MmapBuffer() = default;
  MmapBuffer(void* d, size_t s, size_t c) : data(d), size(s), capacity(c) {}
  ~MmapBuffer() {
    if (data) munmap(data, capacity);
  }
  MmapBuffer(const MmapBuffer&) = delete;
  MmapBuffer& operator=(const MmapBuffer&) = delete;
  MmapBuffer(MmapBuffer&& other) noexcept
    : data(other.data), size(other.size), capacity(other.capacity) {
    other.data = nullptr;
    other.size = 0;
    other.capacity = 0;
  }
  MmapBuffer& operator=(MmapBuffer&& other) noexcept {
    if (this != &other) {
      if (data) munmap(data, capacity);
      data = other.data; size = other.size; capacity = other.capacity;
      other.data = nullptr; other.size = 0; other.capacity = 0;
    }
    return *this;
  }
};

MmapBuffer load_file_hugepage(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Could not open file: " << filename << std::endl;
    return MmapBuffer{};
  }

  size_t file_size = file.tellg();

  // Pad to 2MB huge page boundary
  constexpr size_t HPAGE = 2UL * 1024 * 1024;
  size_t alloc_size = ((file_size + HPAGE - 1) / HPAGE) * HPAGE;

  // Try huge pages first
  void* ptr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
  if (ptr == MAP_FAILED) {
    // Fallback to regular pages
    ptr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
      std::cerr << "mmap failed for " << filename << std::endl;
      return MmapBuffer{};
    }
  }

  // Read file directly into the mapped buffer
  file.seekg(0);
  file.read(reinterpret_cast<char*>(ptr), file_size);

  // Pad remainder with spaces for NPU chunk alignment
  memset(reinterpret_cast<char*>(ptr) + file_size, ' ', alloc_size - file_size);

  file.close();

  return MmapBuffer{ptr, file_size, alloc_size};
}

MmapBuffer load_file_lazy_thp(const std::string& filename) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd < 0) {
    std::cerr << "Could not open file: " << filename << std::endl;
    return MmapBuffer{};
  }

  off_t file_size = lseek(fd, 0, SEEK_END);
  if (file_size <= 0) {
    close(fd);
    return MmapBuffer{};
  }

  void* ptr = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  close(fd);

  if (ptr == MAP_FAILED) {
    std::cerr << "mmap failed for " << filename << std::endl;
    return MmapBuffer{};
  }

  // Encourage THP + sequential readahead to reduce page-fault overhead
  madvise(ptr, file_size, MADV_HUGEPAGE);
  madvise(ptr, file_size, MADV_SEQUENTIAL);

  return MmapBuffer{ptr, (size_t)file_size, (size_t)file_size};
}

MmapBuffer load_file_lazy(const std::string& filename) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd < 0) {
    std::cerr << "Could not open file: " << filename << std::endl;
    return MmapBuffer{};
  }

  off_t file_size = lseek(fd, 0, SEEK_END);
  if (file_size <= 0) {
    close(fd);
    return MmapBuffer{};
  }

  // File-backed MAP_PRIVATE: pages loaded lazily on first access
  void* ptr = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  close(fd);

  if (ptr == MAP_FAILED) {
    std::cerr << "mmap failed for " << filename << std::endl;
    return MmapBuffer{};
  }

  return MmapBuffer{ptr, (size_t)file_size, (size_t)file_size};
}

} // namespace util
