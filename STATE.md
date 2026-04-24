# STATE.md — npu-json Project State

## Project Overview

This project is a JSONPath query engine accelerated by AMD XDNA NPU for index computing. The core pipeline:

```
JSON bytes → Structural Indexing (find structural chars) → JSONPath Automaton Engine → Results
```

There are now **4 backends**:

| Backend | Define | NPU/XRT | Description |
|---------|--------|---------|-------------|
| `npu` (original) | default | Yes | Two separate NPU kernels: `string_index` + `structural_character_index` |
| `cpu` | `-Dcpu_backend=true` | No | CPU SIMD (AVX-512) implementation of the same two-kernel pipeline |
| `matrix` | `-Dmatrix_backend=true` | No | CPU matrix ops (GEMM, step activation, prefix-XOR, compress) — fuses both kernels into one pass |
| `npu-matrix` | `-Dnpu_matrix_backend=true` | Yes | NPU-accelerated combined kernel — fuses both kernels into a single AIE kernel |

## Architecture

### Structural Indexing Algorithm (all backends)

1. **Character classification** — Detect `{`, `}`, `[`, `]`, `:`, `,`, `"`, `\` (GEMM-equiv or SIMD compare)
2. **Escape detection** — Bit manipulation to find escaped backslashes (AND, NOT, XOR, shift)
3. **String detection** — Prefix-XOR (inclusive scan with XOR) on non-escaped quotes
4. **Structural mask** — OR the structural chars, AND NOT the string mask
5. **Compress** — Popcount + bit-scan to convert bitmask → position array

### Engine

`src/npu-json/engine.cpp` — JSONPath automaton that walks structural characters. Uses `StructuralIterator` abstract interface to get structural characters from any backend.

### Key Abstractions

- `npu::StructuralIterator` (`src/npu-json/npu/iterator.hpp`) — Abstract base class with `get_next_structural_character()`, `setup()`, `reset()` etc.
- `npu::ChunkIndex` (`src/npu-json/npu/chunk-index.hpp`) — Holds structural character positions + string/escape index for a chunk
- `npu::Kernel` / `npu::PipelinedIterator` — Original NPU backend (two separate kernels)
- `matrix::MatrixKernel` / `matrix::MatrixPipeline` — CPU matrix backend
- `matrix::npu::NPUMatrixKernel` / `matrix::npu::NPUMatrixPipeline` — NPU matrix backend (new, **BROKEN**)

## File Map

### Matrix Backend (CPU — WORKING)
- `src/npu-json/matrix/matrix.hpp` — Matrix data structure
- `src/npu-json/matrix/ops.hpp` / `ops.cpp` — GEMM, step activation, prefix-XOR, character match, compress
- `src/npu-json/matrix/kernel.hpp` / `kernel.cpp` — CPU matrix kernel using matrix ops
- `src/npu-json/matrix/pipeline.hpp` / `pipeline.cpp` — CPU matrix pipeline (StructuralIterator)

### NPU Matrix Backend (NPU — BROKEN at runtime)
- `src/npu-json/matrix/npu/kernel.hpp` / `kernel.cpp` — **Host-side NPU kernel — NEEDS REWRITE**
- `src/npu-json/matrix/npu/pipeline.hpp` / `pipeline.cpp` — NPU matrix pipeline (StructuralIterator)
- `src/aie/json_matrix_xdna2.cc` / `json_matrix_xdna1.cc` — **Combined AIE kernel** (WORKS, compiles for AIE)
- `src/aie/gen_mlir_matrix_design.py` — MLIR design for combined kernel (2 core rows)

### Shared/Modified
- `src/npu-json/npu/iterator.hpp` — New abstract StructuralIterator interface
- `src/npu-json/engine.hpp` / `engine.cpp` — Uses `StructuralIterator*`, dispatches to backend via `#ifdef`
- `src/npu-json/options.hpp` — Path constants for xclbin/insts files
- `meson.build` / `meson.options` — Build system with all 4 backend options
- `Justfile` — Build targets including `build-npu-matrix-hx370`

### Tests
- `test/unit/cpu_backend_test.cpp` — CPU backend unit tests
- `test/unit/matrix_backend_test.cpp` — Matrix backend unit tests (11 test cases, all pass)
- `test/unit/npu_matrix_backend_test.cpp` — NPU matrix tests (guarded by `NPU_JSON_NPU_MATRIX_BACKEND`, requires hardware)

## Current Error / Blocker

### Runtime: `npu-matrix` produces 0 results

The `npu-matrix` backend compiles and the AIE xclbin builds successfully, but produces 0 results when running queries.

**Root cause:** The host-side `NPUMatrixKernel` (in `src/npu-json/matrix/npu/kernel.cpp`) was copied from the original `npu::Kernel` and still follows the **old two-kernel flow**:

1. `prepare_kernel_input()` — Still builds quote/backslash bitmaps on CPU (calls `construct_escape_carry_index`, builds dual character index). This is wrong for the combined kernel.
2. `read_kernel_output()` — Still rectifies string_index with carry states and computes `structural & ~string_index` on CPU. The combined kernel already produces `nonquoted_structural` directly.
3. `call()` — Still uses ping-pong buffering of the old `string_input`/`string_output`/`structural_output` buffers.

The **combined AIE kernel** (`json_matrix_xdna2.cc`) has a completely different I/O contract:

**Input:** 4 bytes carry flags (bit 0 = string carry, bit 1 = escape carry) + raw JSON data
```
[in_buffer]: [carry_flags(4 bytes)] [json_data(block_size bytes)]
```

**Output:** Two separate buffers:
```
[string_out]:   string_index bitmask (one uint64_t per 64-byte lane)
[structural_out]: non-quoted structural bitmask (one uint64_t per 64-byte lane, strings already masked out)
```

### What needs to be rewritten in `matrix/npu/kernel.cpp`

1. **`NPUMatrixKernel` constructor** — Change buffer layout:
   - Input: single buffer per chunk = `4 + CHUNK_SIZE` bytes (carry + JSON data)
   - Remove `string_input_buffers` and separate `build_dual_character_index`
   - Output: two separate buffers for string_index and structural_index
   - Keep ping-pong buffering for the input, but simpler layout

2. **`prepare_kernel_input()`** — Simplify to:
   ```cpp
   // Pack 4-byte carry flags + raw JSON chunk
   uint32_t carry_flags = 0;
   if (previous_string_carry) carry_flags |= 1;
   if (previous_escape_carry) carry_flags |= 2;
   memcpy(input_buf, &carry_flags, 4);
   memcpy(input_buf + 4, chunk, CHUNK_SIZE);
   ```

3. **`read_kernel_output()`** — Simplify to:
   - Read `string_index` directly from output buffer (still need carry propagation between chunks)
   - Read `structural_index` directly — it's already `nonquoted_structural` (strings masked out)
   - Use `write_structural_index` to compress the bitmask to position array
   - Remove the `structural_index_buf[pos] & ~index.string_index[pos]` computation (already done on NPU)

4. **`call()`** — Change kernel invocation:
   - 4 arguments: `data_buffer`, `input_buffer`, `string_output`, `structural_output`
   - The MLIR design uses `runtime_sequence(data_chunk_ty, string_chunk_ty, index_chunk_ty, index_chunk_ty)` with 4 buffers
   - Need to match the argument order from `gen_mlir_matrix_design.py`

### MLIR Design Data Flow

The `gen_mlir_matrix_design.py` uses 4 DMA channels per column (same as original):
1. `string_input_buffer` → NPU (combined carry+data input, but only carry+blank data is actually used by the combined kernel's input path via the `string_in` FIFO)
2. NPU → `string_index_buffer` (string bitmask output)
3. `data_buffer` → NPU (raw JSON data, via `structural_in` FIFO — **NOTE: this is for the structural path input which the combined kernel doesn't use, but it's needed for DMA channel allocation**)
4. NPU → `structural_index_buffer` (non-quoted structural bitmask output)

**IMPORTANT**: The combined kernel receives its data through `core_fifos_in` (carry+data) connected to `shim_fifos_in_string`. The `shim_fifos_in_structural` transfer of raw JSON data is currently connected to memory tiles but NOT fanned out to core tiles. The kernel only has 2 core rows (rows 0-1) which process carry+data and produce both outputs. The raw JSON data DMA transfer (`data_buffer`) still exists to satisfy the runtime sequence's 4-buffer interface, but the cores don't read from it.

This means the `data_buffer` DMA is essentially dead weight in the current design. The combined kernel gets all its data through the `combined_input_buffer` (carry flags + raw JSON). This is a simplification opportunity but also means we're wasting DMA bandwidth.

**Alternative approach**: Eliminate the unused `data_buffer` DMA and have only 3 buffers in the runtime sequence. This would require 3 DMA channels per column instead of 4, but the AIE shim tiles support at most 2 DMA channels per direction. So the 4-buffer approach (matching the original) is the safe choice.

## Build System

### Backend Selection
```bash
# Original NPU backend (default)
meson setup build

# CPU-only backend (no NPU/XRT)
meson setup build --reconfigure -Dcpu_backend=true

# CPU matrix backend (no NPU/XRT)
meson setup build --reconfigure -Dmatrix_backend=true

# NPU matrix backend (requires XRT + AIE tools)
meson setup build --reconfigure -Dnpu_matrix_backend=true
```

### Justfile Targets
```
just build-cpu-only      # CPU SIMD backend
just build-cpu-matrix     # CPU matrix ops backend
just build-npu-matrix-hx370  # NPU matrix backend (hx370 defaults)
```

### Test Commands
```bash
# Unit tests (works with matrix_backend, cpu_backend)
./build/test/unit_tests              # All tests
./build/test/unit_tests "matrix*"    # Matrix-specific tests
sh test/e2e/run_small_e2e.sh         # Small e2e tests
sh test/e2e/run_big_e2e.sh           # Big e2e tests
```

## Git History

Recent commits:
1. `5fe6a26` — Add cpu-only backend (original)
2. `a13444b` — Add matrix backend for JSONPath structural indexing (CPU matrix ops)
3. `4fb4147` — Add npu-matrix backend with combined AIE kernel
4. `6dc52d0` — Fix npu-matrix build: namespace, restrict, and MLIR design

All matrix CPU backend tests pass (11 test cases, 76 assertions, including correctness comparison with CPU backend). The CPU matrix and e2e tests are verified working.

## Key Design Decisions

1. **Matrix ops on CPU**: Uses simple CPU implementations of GEMM, prefix-XOR, etc. These are equivalent to SIMD/NPU ops and serve as the reference for correctness. When targeting NPU, the same algorithm runs on AIE cores using `aie::eq()` / `aie::broadcast()` for character classification.

2. **AIE kernel uses high-level API**: The combined kernel (`json_matrix_xdna2.cc`) uses `aie::eq()` and `aie::broadcast()` for character classification rather than a dense GEMM. This is correct because the weight matrix is extremely sparse (only 8 out of 256 rows have nonzero entries), making `aie::eq()` the idiomatic and efficient AIE operation.

3. **StructuralIterator abstraction**: All backends implement `npu::StructuralIterator`, allowing the engine to be backend-agnostic. The `#ifdef` in `engine.cpp` selects the backend at compile time.

4. **MLIR design uses 2 core rows**: The combined kernel only needs 2 rows (vs 4 for the original), but the DMA interface remains 4 channels to match the AIE shim tile architecture.

## Known Issues / TODOs

1. **NPU matrix kernel host code needs complete rewrite** — This is the main blocker. `matrix/npu/kernel.cpp` still has the old two-kernel flow. Need to rewrite constructor, `prepare_kernel_input`, `read_kernel_output`, and `call`.

2. **MLIR design has unused data DMA channel** — The `data_buffer` / `shim_fifos_in_structural` DMA transfer is allocated but not fanned out to core tiles. This wastes bandwidth. Could be optimized later.

3. **The `construct_escape_carry_index` in NPUMatrixKernel** — Should NOT be called. The combined kernel does escape detection internally. The host only needs to provide the carry-in state as a 4-byte flag.

4. **Carry propagation** — The combined kernel outputs `prev_in_string` as the sign bit of the last string_index vector. The host `read_kernel_output` needs to propagate this between chunks (similar to original).

5. **Ping-pong buffering** — The NPUMatrixKernel should maintain ping-pong buffers, but with the simpler combined input, the input buffer is much smaller (4 bytes carry + CHUNK_SIZE data vs. the original's complex dual character index).