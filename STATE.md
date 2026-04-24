# STATE.md — npu-json Project State

## Current Status

**The NPU matrix backend compiles and links successfully, but produces incorrect results at runtime.** The AIE xclbin generates warnings about core-level buffer allocation (16KB input blocks exceed individual bank size on AIE core tiles), and the structural index output does not contain expected characters (e.g., `{` at position 0 is missing).

The original NPU backend (`npu`) works correctly. The CPU matrix backend also works correctly. The issue is specific to the NPU-accelerated combined kernel.

## What was done in this session

1. **AIE kernel changes**: Changed `json_matrix_xdna2.cc` and `json_matrix_xdna1.cc` to output raw `structurals` instead of `structurals & ~string_index`, since per-block string_index starts with carry=0 and can't be masked on AIE.

2. **MLIR design rewrite** (`gen_mlir_matrix_design.py`):
   - Changed input format to carry+data interleaved per block (`input_block_size = 4 + data_block_size`)
   - Added 4-buffer runtime sequence (input_buffer, data_buffer, string_index, structural_index) matching original XRT interface
   - Added separate `shim_fifos_data_in` for unused data DMA
   - Used `input_split_ty = (input_block_size * num_rows,)` for shim-to-mem FIFOs
   - `object_fifo_link` offsets use `i * input_block_size` per row

3. **Host kernel rewrite** (`kernel.hpp`, `kernel.cpp`):
   - Per-block carry+data input format
   - Padded JSON buffer for `construct_escape_carry_index` safety
   - `read_kernel_output` rectifies string_index across entire chunk and masks with `structural & ~rectified_string_index`
   - 4-buffer XRT interface matching MLIR runtime (data, input, string_out, structural_out)

4. **Fixed VECTORS_IN_CHUNK** calculation (was `CHUNK_SIZE / 64 / 8`, corrected to `CHUNK_SIZE / 64`)

5. **Fixed padded_json buffer** for out-of-bounds access in `construct_escape_carry_index`

## Remaining Issue: NPU Matrix Backend Produces Wrong Results

### Symptoms
- `./build/nj people.json '$.people[*].name'` returns "Found 0 results" (should return 3)
- The AIE xclbin produces non-zero string_index and structural_index output, but the structural positions don't include expected characters (e.g., `{` at position 0)
- The AIE xclbin build produces warnings: "Failed to allocate buffer" for `input_block_size = 16388` on core tiles, falling back to basic sequential allocation

### Root Cause (Suspected)
The AIE core tiles have limited data memory (64KB in 8 banks of 8KB each). The combined kernel's input buffer (`input_block_size = carry_size + block_size = 4 + 16384 = 16388 bytes`) plus two output buffers (`2 × 2048 = 4096 bytes`) plus stack may cause the basic sequential allocator to place buffers at incorrect offsets, leading to data corruption.

### Potential Fixes
1. **Reduce block size**: Use `BLOCK_SIZE = 4096` or `2048` instead of `16384` to reduce AIE memory pressure. This requires regenerating the MLIR design and xclbin with different parameters.
2. **Split input into carry + data**: Keep carry separate (4 bytes) and send data block via a different DMA channel, reducing per-core memory to `data_block_size + 2 * index_block_size ≈ 20KB` which fits better.
3. **Use the original NPU backend's two-kernel approach for now** and optimize the combined kernel design later.
4. **Debug the AIE data path**: Add trace output to verify the AIE cores are receiving correct input data and producing correct output.

## Build Commands

```bash
just build-cpu-only          # CPU SIMD backend (working)
just build-cpu-matrix         # CPU matrix ops backend (working)
just build-npu-matrix-hx370   # NPU matrix backend (builds, wrong runtime results)
```

## Test Results

| Backend | Build | E2E Tests |
|---------|-------|-----------|
| `npu` (original) | ✅ | ✅ Found 3 results |
| `cpu` (SIMD) | ✅ | ✅ |
| `matrix` (CPU ops) | ✅ | ✅ |
| `npu-matrix` | ✅ | ❌ Found 0 results |

## Key Files

- `src/aie/json_matrix_xdna2.cc` / `json_matrix_xdna1.cc` — Combined AIE kernel
- `src/aie/gen_mlir_matrix_design.py` — MLIR design for combined kernel
- `src/npu-json/matrix/npu/kernel.hpp` / `kernel.cpp` — Host-side NPU matrix kernel
- `src/npu-json/matrix/npu/pipeline.hpp` / `pipeline.cpp` — NPU matrix pipeline
- `src/npu-json/options.hpp` — XCLBIN and insts paths