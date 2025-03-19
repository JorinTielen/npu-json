#include <cstring>
#include <stdexcept>

#include <npu-json/npu/indexer.hpp>

namespace npu {

std::optional<StructuralIndex::StructuralCharacter> StructuralIndex::get_next_structural_character() {
  return std::optional<StructuralCharacter>();
}

StructuralIndexer::StructuralIndexer(std::string xclbin_path, std::string insts_path) {
  // Initialize NPU
  auto [device, k] = util::init_npu(xclbin_path);
  kernel = k;

  // Setup instruction buffer
  auto instr_v = util::load_instr_sequence(insts_path);
  instr_size = instr_v.size();
  bo_instr = xrt::bo(device, instr_size * sizeof(uint32_t),
                     XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));

  // Setup input/output buffers
  bo_in    = xrt::bo(device, Engine::CHUNK_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  bo_out   = xrt::bo(device, INDEX_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  bo_carry = xrt::bo(device, CARRY_INDEX_SIZE * sizeof(uint32_t),
                     XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  // Copy instructions to buffer
  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(uint32_t));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Zero out output buffer
  memset(bo_out.map<uint8_t *>(), 0, INDEX_SIZE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

void StructuralIndexer::construct_escape_carry_index(const char *chunk, std::array<uint32_t, CARRY_INDEX_SIZE> &index) {
  for (size_t i = 1; i <= Engine::CHUNK_SIZE / Engine::BLOCK_SIZE; i++) {
    auto is_escape_char = chunk[i * Engine::BLOCK_SIZE - 1] == '\\';
    if (!is_escape_char) continue;

    auto escape_char_count = 1;
    while (chunk[(i * Engine::BLOCK_SIZE - 1) - escape_char_count] == '\\') {
      is_escape_char = !is_escape_char;
      escape_char_count++;
    }

    index[i] = is_escape_char;
  }
}

void StructuralIndexer::construct_string_index(const char *chunk, uint64_t *index, uint32_t *escape_carries) {
  // Copy input into buffer
  auto buf_in = bo_in.map<uint8_t *>();
  memcpy(buf_in, chunk, Engine::CHUNK_SIZE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Copy carry index into buffer
  auto buf_carry = bo_carry.map<uint32_t *>();
  memcpy(buf_carry, escape_carries, CARRY_INDEX_SIZE * sizeof(uint32_t));
  bo_carry.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(3, bo_instr, instr_size, bo_in, bo_out, bo_carry);
  run.wait();

  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto buf_out = bo_out.map<uint64_t *>();
  memcpy(index, buf_out, INDEX_SIZE);
}

std::unique_ptr<StructuralIndex> StructuralIndexer::construct_structural_index(const char *chunk) {
  auto index = std::make_unique<StructuralIndex>();

  construct_escape_carry_index(chunk, index->escape_carry_index);
  construct_string_index(chunk, index->string_index.data(), index->escape_carry_index.data());

  return index;
}

} // namespace npu

