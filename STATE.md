# STATE.md — npu-json Project State

## Project Overview

This project is a JSONPath query engine accelerated by AMD XDNA NPU for structural indexing. The core pipeline:

```
JSON bytes → Structural Indexing (find structural chars) → JSONPath Automaton Engine → Results
```

There are **4 backends**:

| Backend | Define | NPU/XRT | Description |
|---------|--------|---------|-------------|
| `npu` (original) | default | Yes | Two separate NPU kernels: `string_index` + `structural_character_index` |
| `cpu` | `-Dcpu_backend=true` | No | CPU SIMD (AVX-512) implementation of the same two-kernel pipeline |
| `matrix` | `-Dmatrix_backend=true` | No | CPU matrix ops — fuses both kernels into one pass |
| `npu-matrix` | `-Dnpu_matrix_backend=true` | Yes | NPU-accelerated combined kernel — fuses both kernels into a single AIE kernel |

## Architecture

### Structural Indexing Algorithm (all backends)

1. **Character classification** — Detect `{`, `}`, `[`, `]`, `:`, `,`, `"`, `\` (GEMM-equiv or SIMD compare)
2. **Escape detection** — Bit manipulation to find escaped backslashes (AND, NOT, XOR, shift)
3. **String detection** — Prefix-XOR (inclusive scan with XOR) on non-escaped quotes
4. **Structural mask** — OR the structural chars, AND NOT the string mask
5. **Compress** — Popcount + bit-scan to convert bitmask → position array

### NPU Matrix Backend Architecture

The combined AIE kernel (`combined_index`) runs on 2 rows × 8 columns of AIE cores. Each core processes blocks of 16KB in a loop:

- **Input per block**: 4 bytes carry flags (bit 0 = string carry, bit 1 = escape carry) + 16KB raw JSON data
- **Output per block**: string bitmask (uint64_t per 64-byte vector) + structural bitmask (uint64_t per 64-byte vector)
- **Carry propagation**: String carry is rectified on the host; escape carry is pre-computed per-block on the CPU

### Host-Side Processing (`matrix/npu/kernel.cpp`)

1. **`prepare_kernel_input`**: Compute per-block escape carry via `construct_escape_carry_index`, interleave carry+data per block, arrange in column-row-block layout for MLIR DMA
2. **`call`**: Launch kernel on NPU with ping-pong buffering; pre-compute escape carry for next chunk while waiting
3. **`read_kernel_output`**: Rectify string_index across all vectors (XOR with carry), compute `nonquoted_structural = structural & ~rectified_string_index`, compress to position array

### Key Design Decisions

1. **AIE outputs raw structurals, not `structurals & ~string_index`**: Per-block string_index starts with carry=0, so masking on AIE would produce wrong results at block boundaries. The host masks after rectification.

2. **String carry rectified on host, not AIE**: Each AIE block starts with `string_carry = 0`. The host XOR-rectifies across all vectors in the chunk (same as original NPU backend).

3. **Escape carry pre-computed on CPU**: `construct_escape_carry_index` checks block boundaries for odd backslash sequences. This is fast (O(blocks_per_chunk)) and ensures correct escape detection across blocks.

4. **4-buffer runtime sequence**: Input (carry+data), data (unused by cores, for DMA compatibility), string_index output, structural_index output. The data_buffer DMA is unused by the AIE cores but required for the 4-buffer XRT interface.

5. **MLIR data layout**: Input buffer uses per-block carry+data interleaved format. The `input_split_ty` is `input_block_size * num_rows = (4 + 16384) * 2 = 32776` bytes per shim-to-mem transfer.

## File Map

### AIE Kernels
- `src/aie/json_matrix_xdna2.cc` — Combined AIE kernel for XDNA2 (NPU2)
- `src/aie/json_matrix_xdna1.cc` — Combined AIE kernel for XDNA1 (NPU1)
- `src/aie/gen_mlir_matrix_design.py` — MLIR design for combined kernel (2 core rows, 4 DMA channels)

### NPU Matrix Backend (Host)
- `src/npu-json/matrix/npu/kernel.hpp` / `kernel.cpp` — NPUMatrixKernel with XRT ping-pong buffering
- `src/npu-json/matrix/npu/pipeline.hpp` / `pipeline.cpp` — NPUMatrixPipeline (StructuralIterator impl)

### Shared/Modified
- `src/npu-json/npu/iterator.hpp` — Abstract StructuralIterator interface
- `src/npu-json/engine.hpp` / `engine.cpp` — Uses `StructuralIterator*`, dispatches via `#ifdef`
- `src/npu-json/options.hpp` — Path constants for xclbin/insts files
- `meson.build` / `meson.options` — Build options for all backends
- `Justfile` — Build targets including `build-npu-matrix-hx370`

### Tests
- `test/unit/cpu_backend_test.cpp` — CPU backend unit tests
- `test/unit/matrix_backend_test.cpp` — Matrix backend unit tests (11 test cases, all pass)
- `test/unit/npu_matrix_backend_test.cpp` — NPU matrix tests (guarded by `NPU_JSON_NPU_MATRIX_BACKEND`, requires hardware)

## Build Commands

```bash
just build-cpu-only      # CPU SIMD backend
just build-cpu-matrix     # CPU matrix ops backend
just build-npu-matrix-hx370  # NPU matrix backend (hx370 defaults)
```

## Test Results

- **Small e2e tests**: PASS (matches `test/e2e/correct-small.log`)
- **Big e2e tests**: PASS (matches `test/e2e/correct-answer.log`)

## Git History (Recent)

1. Fix matrix insts path to match aiecc output location
2. Add STATE.md documenting project state and pending work
3. **Current**: Fix NPU matrix backend to pass e2e tests
   - Changed AIE kernels to output raw `structurals` instead of `structurals & ~string_index`
   - Rewrote MLIR design for carry+data input format with 4-buffer runtime
   - Rewrote host kernel code: per-block carry+data input, string_index rectification, structural masking on host
   - Fixed VECTORS_IN_CHUNK calculation bug
   - Added padded JSON buffer for escape_carry_index safety