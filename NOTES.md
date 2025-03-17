# NOTES

Notes related to the graduation project (i.e. evaluation) and implementation considerations.

## Queries

Datasets are collected in the `datesets` folder. It is important to consider which queries to evaluate on these datasets.

| Code | Dataset | Query | Notes |
|------|---------|-------|-------|
| T1 | `twitter` | `$.user.lang` | From GPJSON. Very simple query, only basic JSONPath support needed. |
| T2 | `twitter` | `$.user.lang[?(@ == 'nl')]` | From GPJSON. Modification of the basic query, but now with a filter. Filter is very selective, so few results. Could measure possible speedup with large skipping. |
| T3 | `twitter` | `$[*].entities.urls[*].url` | From JSONSKi and rsonpath. Uses the `*` for arrays. |
| B1 | `bestbuy` | `$.products[*].categoryPath[1:3].id` | From GPJSON and JSONSKi. Uses a slice (`1:3`) for arrays. |
| B2 | `bestbuy` | `$.products[*].videoChapters[*].chapter` | From JSONSKi and rsonpath. Does not add functionality over T3, but optional for extra data. |
| N1 | `nspl` | `$.meta.view.columns[*].name` | From JSONSKi and rsonpath. Does not add functionality over T3, but optional for extra data. |
| W1 | `walmart` | `$.items[*].bestMarketplacePrice.price` | From GPJSON. Does not add functionality over T3, but optional for extra data. |
| A1 | `ast` | `$..decl.name` | From rsonpath. Uses descendant (`..`) operator. Not sure if we will end up supporting this, but noting the query just in case. |
| A2 | `ast` | `$..inner..inner..type.qualType` | From rsonpath. Uses descendant (`..`) operator. Not sure if we will end up supporting this, but noting the query just in case. |

## Architecture

### Concept

- Iterate over JSON in chunks (10+ MB)
- Create indices for current chunk on NPU in parallel (alt. CPU) with kernels:
  1. Escape carry index for block ends
  2. String index
  3. Structural character index (pos. of opening/closing brackets/braces, colon, comma)
- Fixup string index if previous block ended in string (bitwise inverse)
- Update state (inside/outside string, escape character end) for next chunk
- Run query engine over structural character index (see next section)

The three kernels are required to be done in order, although 2 and 3 could be merged into a single kernel.

The kernels work on blocks, which are further splits of the chunk. (i.e. ~1024 bytes per block).

Inputs and outputs:

- `escape_carry_index(json: string) -> bitset(num_blocks)`
- `string_index(json: string, escape_carry: bool) -> bitset(BLOCK_INDEX_SIZE)`
- `structural_character_index(json: string, string_index: bitset(BLOCK_INDEX_SIZE)) -> 6 * bitset(BLOCK_INDEX_SIZE)`

Here `BLOCK_INDEX_SIZE` is the `BLOCK_DATA_SIZE` (i.e. 1024 bytes) divided by 8 (we go from bytes to bits).
For maximal throughput this is still ok, the output size is still smaller (7 indices total) than input string.

Due to implementation difficulties, an alternative approach utilizing smaller kernels that can be implemented seperately might be better. These allow the string index to not have to be rectified later, and successive kernels can be pipelined on the NPU. The kernels also often only take indices, not the full JSON data, meaning a higher theoretical bandwidth.

- Alternative kernels:
  0. backslash + quotes (json) on CPU [32+GB/s]
  1. escape carry index (backslash) on CPU [-]
  2. quote index (quotes, backslash, escape carry index) [128GB/s]
  3. quote carry index (quote index) on CPU [-]
  4. string index (quote index, quote carry index) [128GB/s]
  5. structural character index (json, string index) on CPU [?GB/s]

### rsonpath

JSONPath query engine works as follows:

- Main loop iterates over structural characters (opening braces, etc.)
  - String index is required to filter out structural characters inside a string
- It does this one block at a time (i.e. 64-bits of index or 64 characters)
- When opening character is hit:
  - Search for matches inside array or object
  - Tail-skip when query automaton is in rejecting state; fast forward to same-level closing character.
  - This still does SIMD cmp instructions linearly over the entire input file looking for the character
    basically charmatcher kernel (30+GB/s on CPU). However it could go faster with an index.
  - After this, increment depth and make automaton search further inside subtree.
- When colon is hit:
  - Record value as result if match
- When comma is hit:
  - Record previous value as result if match
- When closing character is hit:
  - Decrement depth and restore automaton state from stack

Ideas on how to accelerate with indices:

- NPU and CPU can work in parallel on chunks, when CPU automaton is iterating over indices, NPU can
  construct indices for next chunk in advance. The "state" (inside/outside string, escapes) can quickly
  be checked at the end of the current chunk before launching the kernel of the next query. This is a
  task level parallelization of their technique.
- String index is computationally expensive, so getting it for "free" from NPU will be a benefit.
- Structural iterator index: array(s) of numbers, which are indices/pointers into the original file.
  When looking for the next structural character, we don't need to look at the actual JSON string at all.
  In the case of skipping, this should now be extremely fast, just a few integer comparisons looking for
  the closing character at the correct depth, and then updating the index and continuing the automaton
  from that point. The (fast) creation of these number arrays is explained in simdjson. Challenge will
  be constructing it fast on NPU. Perhaps return bitsets (known size), and make array on CPU w/ simdjson impl.

Difficulties:
- Automaton will need to be able to be "paused" in the middle of anything when it hits the end of the chunk,
  and resume on the next chunk. Might be ugly in code. Coroutines? Exit out when end is hit and stay on last state?

### JSONSKi

Works largely similar to rsonpath, one "word" at a time. Different is the means of finding fast-forward
cases with strucutural intervals which are more complex, but achieve similar skipping mechanisms.

### simdjson tape

The On-Demand simdjson backend for simdjson works on the tape to access data on demand. You could imagine
a JSONPath query going over the tape. Problem is, tape might require more processing for elements that are
skipped anyways.
