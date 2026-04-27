# Algorithm: Combined JSON Indexing Kernel

## Overview

The combined index kernel performs **structural indexing** of JSON in a single streaming pass over raw bytes. For each 64-byte vector it computes: character classification, escape sequence detection, string boundary detection, and structural character extraction. All operations are expressed as standard vector/bit operations — no branches, no lookup tables, no cross-lane dependencies beyond prefix scan.

## Algorithm Steps (per 64-byte vector)

### Step 1: Character Classification
```
quotes      = (data == '"').to_u64()
backslash   = (data == '\\').to_u64()
brace_open  = (data == '{').to_u64()
brace_close = (data == '}').to_u64()
bracket_open  = (data == '[').to_u64()
bracket_close = (data == ']').to_u64()
colon       = (data == ':').to_u64()
comma       = (data == ',').to_u64()
```

**GEMM equivalence**: This is `step(one_hot(data) ⊗ W)` where W is a 256×8 weight matrix. Each comparison computes one column of the output. On platforms with efficient GEMM or convolution, these 8 comparisons can be replaced with a single `matmul + relu` or small `conv2d`.

| Platform | Classification Implementation |
|----------|------------------------------|
| AMD AIE | `aie::eq(data, mask).to_u64()` (8× per vector) |
| CPU (AVX-512) | `_mm512_cmpeq_epu8_mask` (8× per 64B) |
| CPU (NEON) | `vceqq_u8` + movemask (8× per 16B) |
| GPU (CUDA) | `__vcmpeq4` or 1× `one_hot + gemm` |
| CNN Accel | 1× `conv2d(8,1×1)` + ReLU |
| RISC-V V | `vmseq.vv` (8× per vector) |

### Step 2: Escape Sequence Detection
```
potential_escape = backslash & ~prev_is_escaped
escaped = ((potential_escape << 1 | ODD_BITS) - potential_escape) ^ ODD_BITS ^ (backslash | prev_is_escaped)
escape  = escaped & backslash
prev_is_escaped = escape >> 63
```

This is the simdjson icelake algorithm. Uses only bit operations: AND, OR, XOR, shift-left, subtract, shift-right. No branches. Runs identically on any 64-bit ALU.

### Step 3: String Detection (Prefix-XOR Scan)
```
non_escaped_quotes = quotes & ~escaped
string_index = prefix_xor(non_escaped_quotes)
string_index ^= prev_in_string_carry
prev_in_string_carry = arith_shift_right(string_index, 63)
```

The prefix-XOR is an inclusive parallel scan in log₂(64) = 6 shift+XOR steps:
```
x ^= x << 1
x ^= x << 2
x ^= x << 4
x ^= x << 8
x ^= x << 16
x ^= x << 32
```

**Carry handling**: A static variable propagates the string-in/out state between vectors and between kernel calls (blocks), enabling streaming operation without host intervention.

### Step 4: Structural Mask Combination
```
braces   = brace_open | brace_close
brackets = bracket_open | bracket_close
colons_and_commas = colon | comma
structurals = braces | brackets | colons_and_commas
masked_structurals = structurals & ~string_index
```

Simple OR reduction (3 operations) followed by ANDN to exclude in-string characters.

## Portability Matrix

| Operation | Type | GPU | ARM NEON | SVE | RISC-V V | CNN Accel |
|-----------|------|-----|----------|-----|----------|-----------|
| vector == scalar | SIMD | ✓ | ✓ | ✓ | ✓ | ✓ (conv) |
| AND/OR/XOR | Bit ALU | ✓ | ✓ | ✓ | ✓ | ✓ |
| shift left/right | Bit ALU | ✓ | ✓ | ✓ | ✓ | ✓ |
| subtract | Int ALU | ✓ | ✓ | ✓ | ✓ | ✓ |
| prefix scan (log N) | Shift+XOR | ✓ | ✓ | ✓ | ✓ | ✗* |
| compress bits→index | TBL/SIMD | ✓ | ✓ | ✓ | ✓ | ✗ |
| broadcast | SIMD | ✓ | ✓ | ✓ | ✓ | ✓ |

\* Prefix scan on CNN accelerators can be emulated with matrix multiplication in O(log N) passes.

## Host-Side Processing

After the AIE kernel produces per-block output, the host performs:

1. **Between-block string carry** (512 ops/chunk): XOR the previous block's string carry into the first vector of each block's string_index, for `ends_in_string()` tracking
2. **Structural extraction** (131K vectors/chunk): scan non-quoted structural bitmasks, popcount non-zero vectors, compress bit positions to uint32_t arrays via AVX-512 `_mm512_maskz_compress_epi8`
3. **Escape carry pre-computation** (done once in constructor): scan block boundaries for backslash sequences

## Data Flow

```
Raw JSON (host memory, zero-copy XRT import)
    │
    ├──► NPU DMA ch 0: data blocks (16KB per block)
    ├──► NPU DMA ch 1: carry flags (4B per block: bit0=unused, bit1=escape_carry)
    │
    ▼
┌─────────────────────────────────────────┐
│ AIE Core (repeated for each of 32 blocks)│
│                                          │
│ For each 64B vector (256 per block):     │
│   load_v<64>(data)                       │
│   classify 8 char classes (aie::eq)      │
│   detect escape sequences (bit ops)      │
│   detect strings (prefix_xor)            │
│   combine structurals (OR reduction)     │
│   mask with string_index (ANDN)          │
│   store string_index, masked_structurals │
└─────────────────────────────────────────┘
    │
    ├──► NPU DMA ch 2: string_index (2KB per block)
    └──► NPU DMA ch 3: masked structural bitmasks (2KB per block)
              │
              ▼
         Host: compress bitmasks → uint32_t positions
              │
              ▼
         Automaton: walk positions, evaluate JSONPath query
```

## Key Design Decisions

1. **Zero-copy data**: JSON data imported as XRT buffer once, DMA'd per-chunk via sub-buffer offsets. No per-chunk memcpy.
2. **Static string carry**: AIE kernel uses a `static` variable for between-block string carry, eliminating host-side vector-level rectification (saves ~131K XORs/chunk).
3. **On-AIE masking**: Structurals are masked with string_index on the AIE, so host only extracts positions from pre-filtered bitmasks (saves ~131K ANDN ops/chunk).
4. **Per-block escape carry**: Escape carry is pre-computed per-block on the host and passed via the carry buffer, since escape sequences can span block boundaries and require backward scanning.
