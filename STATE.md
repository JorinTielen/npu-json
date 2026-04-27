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

## Performance Optimizations

1. **Combined read_kernel_output passes**: Merged string index rectification and structural extraction into a single loop (was two separate passes over 131K vectors each), cutting loop iterations in half and halving buffer reads.

2. **4-vector grouped early-out**: Structural extraction processes 4 vectors at a time with an `(r0|r1|r2|r3)==0` skip check, matching the original NPU backend pattern. Most 64-byte blocks contain no structural characters, so this skips most iterations.

3. **Pre-computed chunk escape carries**: Escape carry state for each chunk boundary is computed eagerly in the constructor, eliminating the need to read the previous chunk's output before preparing the current chunk's input. This restores pipelining overlap between input preparation and kernel execution.

4. **Removed unused `previous_escape_carry` member**: Superseded by pre-computed `chunk_escape_carries` vector.

### Performance Impact

| Benchmark | Before (GB/s) | After (GB/s) | Improvement |
|-----------|--------------|-------------|-------------|
| twitter T1 | 6.13 | 10.12 | 1.65x |
| twitter T2 | 6.09 | 10.11 | 1.66x |
| bestbuy B1 | 6.39 | 9.98 | 1.56x |
| bestbuy B2 | 6.37 | 10.54 | 1.65x |
| nspl N1 | 6.20 | 10.77 | 1.74x |
| walmart W1 | 5.94 | 7.60 | 1.28x |
| wikipedia Wi | 6.41 | 9.55 | 1.49x |

The NPU matrix backend is now within 2-15% of the original NPU backend on most benchmarks.

## Zero-Copy Optimization

Separated the combined kernel input into two independent DMA channels:
- **Data channel**: raw JSON data imported as an XRT buffer once (zero-copy from host memory)
- **Carry channel**: tiny 2KB carry-flags buffer per chunk

This required changes across three layers:
1. **AIE kernel** (`json_matrix_xdna2.cc`, `json_matrix_xdna1.cc`): now accepts `data_in` + `carry_in` instead of interleaved `in_buffer`
2. **MLIR design**: added separate data and carry `object_fifo` channels per column with row-level distribution via `object_fifo_link`
3. **Host kernel**: pre-reorders data once at construction time, eliminating per-chunk 8MB memcpy

### Benchmark Results

| Benchmark | Before (GB/s) | After Zero-Copy (GB/s) | Improvement | vs Baseline |
|-----------|--------------|----------------------|-------------|-------------|
| twitter T1 | 10.12 | 11.16 | 1.10x | 12.39 |
| twitter T2 | 10.11 | 10.51 | 1.04x | 12.53 |
| bestbuy B1 | 9.98 | 10.05 | 1.01x | 10.45 |
| bestbuy B2 | 10.54 | 11.16 | 1.06x | 11.79 |
| googlemaps G1 | 4.70 | 5.04 | 1.07x | 5.15 |
| googlemaps G2 | 6.87 | 8.08 | 1.18x | 7.08 |
| nspl N1 | 10.77 | 11.78 | 1.09x | 12.44 |
| nspl N2 | 5.33 | 5.56 | 1.04x | 5.60 |
| walmart W1 | 7.60 | 8.10 | 1.07x | 7.53 |
| wikipedia Wi | 9.55 | 9.67 | 1.01x | 9.74 |

### Post Zero-Copy Per-Chunk Breakdown (N1 trace)

| Task | Before (µs) | After (µs) |
|------|------------|-----------|
| prepare_kernel_input (8MB memcpy) | 422.4 | **0.4** |
| input/carry sync | 109.2 | **0.7** |
| npu_matrix_kernel_roundtrip | 485.9 | 498.6 |
| automaton_matrix_npu | 428.7 | 337.7 |
| read_kernel_output_matrix | 179.2 | 208.9 |

The indexing bottleneck is now the NPU hardware roundtrip (~499µs, 48%) and automaton query evaluation (~338µs, 32%).

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