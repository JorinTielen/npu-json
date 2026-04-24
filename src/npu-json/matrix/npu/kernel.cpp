#include <cstring>
#include <immintrin.h>
#include <stdexcept>

#include <npu-json/matrix/npu/kernel.hpp>
#include <npu-json/npu/chunk-index.hpp>
#include <npu-json/npu/kernel.hpp>
#include <npu-json/util/debug.hpp>
#include <npu-json/util/tracer.hpp>
#include <npu-json/engine.hpp>
#include <npu-json/options.hpp>

namespace matrix::npu {

__attribute__((always_inline)) inline uint64_t count_ones(uint64_t mask) {
  return __builtin_popcountll(mask);
}

__attribute__((always_inline)) inline void write_structural_index(
  uint32_t *tail,
  uint64_t bits,
  const size_t position,
  const size_t count
) {
  if (bits == 0) {
    return;
  }

  const __m512i indexes = _mm512_maskz_compress_epi8(bits, _mm512_set_epi32(
    0x3f3e3d3c, 0x3b3a3938, 0x37363534, 0x33323130,
    0x2f2e2d2c, 0x2b2a2928, 0x27262524, 0x23222120,
    0x1f1e1d1c, 0x1b1a1918, 0x17161514, 0x13121110,
    0x0f0e0d0c, 0x0b0a0908, 0x07060504, 0x03020100
  ));
  const __m512i start_index = _mm512_set1_epi32(position);

  __m512i t0 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(indexes));
  _mm512_storeu_si512(tail, _mm512_add_epi32(t0, start_index));

  if (count > 16) {
    const __m512i t1 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(indexes, 1));
    _mm512_storeu_si512(tail + 16, _mm512_add_epi32(t1, start_index));
    if (count > 32) {
      const __m512i t2 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(indexes, 2));
      _mm512_storeu_si512(tail + 32, _mm512_add_epi32(t2, start_index));
      if (count > 48) {
        const __m512i t3 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(indexes, 3));
        _mm512_storeu_si512(tail + 48, _mm512_add_epi32(t3, start_index));
      }
    }
  }
}

NPUMatrixKernel::NPUMatrixKernel(std::string_view json) {
  auto xclbin = xrt::xclbin(MATRIX_XCLBIN_PATH);
  auto [device, context] = util::init_npu(xclbin);

  kernel = xrt::kernel(context, "MLIR_AIE");

  auto instr_v = util::load_instr_sequence(MATRIX_INSTS_PATH);
  instr_size = instr_v.size();
  instr = xrt::bo(device, instr_size * sizeof(uint32_t),
                   XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));

  size_t input_buffer_size = Engine::BLOCKS_PER_CHUNK * NPU_INPUT_BLOCK_SIZE;
  size_t output_buffer_size = Engine::CHUNK_SIZE / 8;

  size_t padded_json_size = (json.size() + Engine::CHUNK_SIZE - 1) / Engine::CHUNK_SIZE * Engine::CHUNK_SIZE;
  if (padded_json_size == 0) padded_json_size = Engine::CHUNK_SIZE;

  for (size_t i = 0; i < 2; i++) {
    buffers[i].input = xrt::bo(device, input_buffer_size, XRT_BO_FLAGS_HOST_ONLY,
                                kernel.group_id(3));
    buffers[i].string_output = xrt::bo(device, output_buffer_size, XRT_BO_FLAGS_HOST_ONLY,
                                       kernel.group_id(4));
    buffers[i].structural_output = xrt::bo(device, output_buffer_size, XRT_BO_FLAGS_HOST_ONLY,
                                           kernel.group_id(5));
  }

  memcpy(instr.map<void *>(), instr_v.data(), instr_size * sizeof(uint32_t));
  instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  padded_json.resize(padded_json_size, ' ');
  memcpy(padded_json.data(), json.data(), json.length());
  json_length = json.length();

  auto chunk_count = padded_json_size / Engine::CHUNK_SIZE;
  chunk_escape_carries.resize(chunk_count + 1, false);
  bool carry = false;
  for (size_t c = 0; c < chunk_count; c++) {
    chunk_escape_carries[c] = carry;
    const char *chunk = padded_json.data() + c * Engine::CHUNK_SIZE;
    for (size_t i = 1; i <= Engine::BLOCKS_PER_CHUNK; i++) {
      auto is_escape_char = chunk[i * Engine::BLOCK_SIZE - 1] == '\\';
      if (!is_escape_char) {
        carry = false;
        continue;
      }
      auto escape_char_count = 1;
      while (chunk[(i * Engine::BLOCK_SIZE - 1) - escape_char_count] == '\\') {
        is_escape_char = !is_escape_char;
        escape_char_count++;
      }
      carry = is_escape_char;
    }
  }
  chunk_escape_carries[chunk_count] = carry;

  for (size_t i = 0; i < 2; i++) {
    input_maps[i] = buffers[i].input.map<uint8_t *>();
    string_output_maps[i] = buffers[i].string_output.map<uint64_t *>();
    structural_output_maps[i] = buffers[i].structural_output.map<uint64_t *>();
  }
}

void NPUMatrixKernel::prepare_kernel_input(
  ::npu::ChunkIndex &index,
  size_t chunk_idx,
  size_t buffer
) {
  auto &tracer = util::Tracer::get_instance();
  auto trace = tracer.start_trace("prepare_kernel_input_matrix");

  auto input_buf = input_maps[buffer];
  constexpr size_t CARRY_SECTION_SIZE = 64;

  for (size_t col = 0; col < NPU_NUM_COLS; col++) {
    size_t column_offset = col * NPU_BLOCKS_PER_COLUMN * NPU_INPUT_BLOCK_SIZE;
    for (size_t b = 0; b < NPU_BLOCKS_PER_ROW; b++) {
      for (size_t row = 0; row < NPU_NUM_ROWS; row++) {
        size_t block_idx = col * NPU_BLOCKS_PER_COLUMN + b * NPU_NUM_ROWS + row;
        size_t offset = column_offset + (b * NPU_NUM_ROWS + row) * NPU_INPUT_BLOCK_SIZE;

        uint32_t carry_flags = 0;
        if (index.escape_carry_index[block_idx]) carry_flags |= 2;

        memset(input_buf + offset, 0, CARRY_SECTION_SIZE);
        memcpy(input_buf + offset, &carry_flags, sizeof(uint32_t));

        size_t data_offset = chunk_idx + block_idx * Engine::BLOCK_SIZE;
        const char *src = padded_json.data() + data_offset;
        memcpy(input_buf + offset + CARRY_SECTION_SIZE, src, Engine::BLOCK_SIZE);
      }
    }
  }

  tracer.finish_trace(trace);
}

void NPUMatrixKernel::read_kernel_output(
  ::npu::ChunkIndex &index,
  size_t chunk_idx
) {
  auto &tracer = util::Tracer::get_instance();

  constexpr const size_t VECTOR_BYTES = 64;
  constexpr const auto VECTORS_IN_CHUNK = Engine::CHUNK_SIZE / VECTOR_BYTES;

  auto trace = tracer.start_trace("read_kernel_output_matrix");

  auto output_buffer = current;

  auto string_index_buf = string_output_maps[output_buffer];
  auto structural_index_buf = structural_output_maps[output_buffer];

  auto tail = index.block.structural_characters.data();
  index.block.structural_characters_count = 0;

  bool last_inside_string = previous_string_carry;
  size_t i = 0;

  for (; i + 3 < VECTORS_IN_CHUNK; i += 4) {
    uint64_t xs0 = string_index_buf[i];
    uint64_t xs1 = string_index_buf[i + 1];
    uint64_t xs2 = string_index_buf[i + 2];
    uint64_t xs3 = string_index_buf[i + 3];

    uint64_t s0 = xs0 ^ (-static_cast<uint64_t>(last_inside_string));
    last_inside_string = static_cast<int64_t>(s0) >> 63;
    uint64_t s1 = xs1 ^ (-static_cast<uint64_t>(last_inside_string));
    last_inside_string = static_cast<int64_t>(s1) >> 63;
    uint64_t s2 = xs2 ^ (-static_cast<uint64_t>(last_inside_string));
    last_inside_string = static_cast<int64_t>(s2) >> 63;
    uint64_t s3 = xs3 ^ (-static_cast<uint64_t>(last_inside_string));
    last_inside_string = static_cast<int64_t>(s3) >> 63;

    index.string_index[i] = s0;
    index.string_index[i + 1] = s1;
    index.string_index[i + 2] = s2;
    index.string_index[i + 3] = s3;

    uint64_t r0 = structural_index_buf[i] & ~s0;
    uint64_t r1 = structural_index_buf[i + 1] & ~s1;
    uint64_t r2 = structural_index_buf[i + 2] & ~s2;
    uint64_t r3 = structural_index_buf[i + 3] & ~s3;

    if ((r0 | r1 | r2 | r3) == 0) {
      continue;
    }

    if (r0) {
      const auto count = count_ones(r0);
      write_structural_index(tail, r0, i * VECTOR_BYTES + chunk_idx, count);
      index.block.structural_characters_count += count;
      tail += count;
    }

    if (r1) {
      const auto count = count_ones(r1);
      write_structural_index(tail, r1, (i + 1) * VECTOR_BYTES + chunk_idx, count);
      index.block.structural_characters_count += count;
      tail += count;
    }

    if (r2) {
      const auto count = count_ones(r2);
      write_structural_index(tail, r2, (i + 2) * VECTOR_BYTES + chunk_idx, count);
      index.block.structural_characters_count += count;
      tail += count;
    }

    if (r3) {
      const auto count = count_ones(r3);
      write_structural_index(tail, r3, (i + 3) * VECTOR_BYTES + chunk_idx, count);
      index.block.structural_characters_count += count;
      tail += count;
    }
  }

  for (; i < VECTORS_IN_CHUNK; i++) {
    uint64_t raw_prefix_xor = string_index_buf[i];
    uint64_t rectified_string = raw_prefix_xor ^ (-static_cast<uint64_t>(last_inside_string));
    index.string_index[i] = rectified_string;
    last_inside_string = static_cast<int64_t>(rectified_string) >> 63;

    uint64_t nonquoted_structural = structural_index_buf[i] & ~rectified_string;
    if (nonquoted_structural == 0) {
      continue;
    }

    const auto count = count_ones(nonquoted_structural);
    write_structural_index(tail, nonquoted_structural, i * VECTOR_BYTES + chunk_idx, count);
    index.block.structural_characters_count += count;
    tail += count;
  }

  tracer.finish_trace(trace);
}

void NPUMatrixKernel::call(::npu::ChunkIndex *index, size_t chunk_idx, std::function<void()> callback) {
  size_t chunk_slot = chunk_idx / Engine::CHUNK_SIZE;

  ::npu::construct_escape_carry_index(
    padded_json.data() + chunk_idx,
    *index,
    chunk_escape_carries[chunk_slot]
  );

  if (previous_run.has_value()) {
    prepare_kernel_input(*index, chunk_idx, !current);

    previous_run->handle.wait();

    buffers[current].string_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    buffers[current].structural_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    read_kernel_output(*previous_run->index, previous_run->chunk_idx);

    previous_string_carry = previous_run->index->ends_in_string();
    previous_run->callback();
    previous_run.reset();

    current = !current;
  } else {
    prepare_kernel_input(*index, chunk_idx, current);
  }

  buffers[current].input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(3, instr, instr_size,
    buffers[current].input,
    buffers[current].string_output,
    buffers[current].structural_output);

  previous_run = std::optional<RunHandle>({run, index, chunk_idx, callback});
}

void NPUMatrixKernel::wait_for_previous() {
  if (!previous_run.has_value()) {
    throw std::logic_error("Called wait for previous without previous run");
  }

  previous_run->handle.wait();

  buffers[current].string_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  buffers[current].structural_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  read_kernel_output(*previous_run->index, previous_run->chunk_idx);

  previous_string_carry = previous_run->index->ends_in_string();
  previous_run->callback();
  previous_run.reset();
}

}