# STATE.md — npu-json Project State

## Current Status

**All backends compile and pass all tests.** The NPU matrix backend now produces correct results for both single-chunk and multi-chunk JSON inputs.

## Bugs Fixed in This Session

1. **AIE data alignment bug**: `data_ptr = in_buffer + 4` was misaligned for `aie::load_v<64>()`. Fixed by padding carry section to 64 bytes (`CARRY_SECTION_SIZE = 64`, `NPU_INPUT_BLOCK_SIZE = 64 + BLOCK_SIZE`).

2. **String carry propagation divergence**: AIE kernels produced `prefix_xor` with inter-vector carry already XORed in, diverging from the CPU backend. Fixed by removing `prev_in_string` from AIE kernels; host now handles all inter-vector carry via `raw_prefix_xor ^ carry_mask`.

3. **Unused data DMA causing potential deadlock**: Removed `shim_fifos_data_in` and `json_data_input`/`json_chunk_inputs` from MLIR design and host. Switched to 3-buffer runtime (input, string_out, structural_out).

4. **Duplicate line in xdna1.cc**: Removed duplicate `auto structurals = braces | brackets | colons_and_commas;`.

5. **Multi-chunk carry ordering bug**: `construct_escape_carry_index` was called with stale `previous_escape_carry` before the previous chunk's output was read. Fixed by reading previous output and updating carries before constructing the next chunk's escape carry index.

6. **Data offset bug in `prepare_kernel_input`**: Second+ chunks always read from the beginning of `padded_json` instead of offsetting by `chunk_idx`. Fixed by adding `chunk_idx` parameter and adding it to data offsets.

7. **Buffer index bug in `read_kernel_output`**: Used `!current` after restructuring, but output is in `buffers[current]` before flip. Fixed to use `current`.

## Build Commands

```bash
just build-cpu-only          # CPU SIMD backend
just build-cpu-matrix         # CPU matrix ops backend
just build-npu-matrix-hx370   # NPU matrix backend
```

## Test Results

| Backend | Build | Unit Tests | E2E Tests |
|---------|-------|------------|-----------|
| `cpu` (SIMD) | ✅ | ✅ | ✅ |
| `matrix` (CPU ops) | ✅ | ✅ | ✅ |
| `npu-matrix` | ✅ | ✅ (22 cases, 102 assertions) | ✅ |

## Key Design Decisions

- AIE outputs raw `structurals`; host masks with `structural & ~rectified_string_index`
- AIE outputs raw `prefix_xor` for string_index WITHOUT inter-vector carry; host propagates carry
- 3-buffer XRT runtime (input, string_out, structural_out)
- Carry padded to 64 bytes for AIE alignment; input per block = 64B carry + 16384B data
- Escape carry pre-computed per-block on CPU via `construct_escape_carry_index`
- `previous_escape_carry` and `previous_string_carry` updated before constructing next chunk's input

## Key Files

- `src/aie/json_matrix_xdna2.cc` / `json_matrix_xdna1.cc` — Combined AIE kernel
- `src/aie/gen_mlir_matrix_design.py` — MLIR design (3-buffer, 64B carry)
- `src/npu-json/matrix/npu/kernel.hpp` / `kernel.cpp` — NPUMatrixKernel (3-buffer XRT, carry rectification)
- `src/npu-json/matrix/npu/pipeline.hpp` / `pipeline.cpp` — NPUMatrixPipeline
- `test/unit/npu_matrix_backend_test.cpp` — Multi-chunk and single-chunk tests