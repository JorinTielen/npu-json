# NOTES

Notes related to the graduation project (i.e. evaluation) and implementation considerations.

## Queries

Datasets are collected in the `datesets` folder. It is important to consider which queries to evaluate on these datasets.

| Code | Dataset | Query | Notes |
|------|---------|-------|-------|
| T1 | `twitter` | `$[*].user.lang` | From GPJSON. Very simple query, only basic JSONPath support needed. |
| T3 | `twitter` | `$[*].entities.urls[*].url` | From JSONSKi and rsonpath. |
| B1 | `bestbuy` | `$.products[*].categoryPath[1:3].id` | From GPJSON and JSONSKi. Uses a slice (`1:3`) for arrays. |
| B2 | `bestbuy` | `$.products[*].videoChapters[*].chapter` | From GPJSON, JSONSKi and rsonpath. |
| G1 | `googlemaps` | `$[*].routes[*].legs[*].steps[*].distance.text` | From JSONSki and rsonpath. Google Maps dataset. Massive amount of results. |
| G2 | `googlemaps` | `$[*].available_travel_modes` | From JSONSki and rsonpath. Google Maps dataset. |
| N1 | `nspl` | `$.meta.view.columns[*].name` | From JSONSKi and rsonpath. NSPL dataset. |
| N2 | `nspl` | `$.data[*][*][*]` | From JSONSKi and rsonpath. Massive amount of results. |
| W1 | `walmart` | `$.items[*].bestMarketplacePrice.price` | From GPJSON, JSONSki and rsonpath. Walmart dataset. |
| Wi | `wikipedia` | `$[*].claims.P150[*].mainsnak.property` | From JSONSki and rsonpath. Wikipedia dataset. |

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

## Implementation findings

Fixed these by merging input buffer:

- Cannot utilize all tiles due to memory tile DMA output channel limit (when including carry)
  - 4 channels for input to compute tiles
  - 4 channels for carry to compute tiles
  - 1 channel for output to shim tile
- Cannot utilize multiple rows due to unknown issue w/ carry (chrash at runtime)

- Overhead of copying input buffer large (~1GB/s+ lost)
  - Solution: Allocate two xrt_bo's, ping pong between them, calling kernel with the right one.
  - Copy 2 indices (quote and backslash) instead (pros: less data too)
    - Good for string index, alhtough no benefit until optimatization. Full input still needed for structural

### JSONPath engine

- States should be non-reentrant:
  - Complicated implementation by technically being able to get abritrary tokens in all states
  - Cannot keep state during state easily. See: search_depth.
  - Simplify by inverting loop:
    - Main while loop is `while (query_executing)`
    - Each state has a loop eating structurals
    - Execution can end when the initial state does a return/fallback on the final closing
      - State boolean `query_executing` set to false when this happens
- Fallback is buggy / complicated
  - Simplify by splitting into 3 use cases:
    - Return: Exit the current state, without tail-skipping. Drops depth and ip by 1. (needed?)
    - Fallback: Exit the current state, with tail-skipping (single level).
    - Abort: "Exits" the current state, and passes the current structural character to the lower state to handle it.
  - By keeping the methods handling a only a single level: we simplify the logic.

#### Examples:

- OpenObject + ']' -> Abort:
  - The object was never opened, meaning we are already ending the parent level. We want to have the parent
    (WildCard) state handle the closing structural, so it can exit itself. Fallback would get stuck on the wildcard state.
    If there is no previous state (we are initial), error out with "Invalid JSON".
- RecordResult + ',' -> Fallback:
  - We just recorded a result, and want to exit the current object/array, because no duplicate keys. The fallback will
    make sure to skip to the end of the current object/array, as we are still on a comma at the moment.
- RecordResult + '}' -> Abort:
  - We just recorded a result, and want to exit the current object/array. We should clean up our own state and let the previous state handle this token as well.
- FindKey + '}' -> Abort:
  - We were looking for a key, but have reached the end of the current object. The closing of the object should be handled
    by the previous (OpenObject) state. This one could be stacked from "RecordResult + '}' -> Abort".
- OpenObject + '}' -> Return:
  - The object is empty, or we came from a higher state's Abort. Exit the current object and clean up our state.
- WildCard + ']' -> Return:
  - We have reached the end of the WildCard array, so we will clean up our own state and lower to the previous.
- FindKey + ',' -> ?:
  - If we have not yet found a key in the current object, we should continue, but if we have, we can tail skip here.
    - TODO: How to maintain this state? boolean on the stack?
- WildCard + ',' -> void:
  - We skip the comma and continue with the next key/array value.
- WildCard + ':' -> ?:
  - Important in the wildcard state to handle both objects and arrays correctly. This token can appear in objects.
    We should `advance()` so OpenObject can handle the following '{' if there is one. WildCard for array should
    `advance()` immediately after eating the first opening. Each time we come back, we can eat a comma and then
    `advance()` again, either immediately for array or after this ':' again for an object.
- WildCard + '}' -> Return:
  - We have reached the end of the WildCard object, so we will clean up our own state and lower to the previous.

## Measurements

for 50MB chunks.

escape_carry_index:      315'944ns   -> 158.3 GB/s
string_index (cpu):      1'912'817ns -> ~26.1 GB/s
string_index (overhead): 8'988'803ns -> ~5.5  GB/s
string_index (npu):      3'881'261ns -> ~12.9 GB/s
structural_index:        3'561'557ns -> ~14.0 GB/s
automaton:               7'758'391ns -> ~6.4  GB/s
automaton (random opts): 7'047'851ns -> ~7.1  GB/s

for 1MB chunks.

automaton:               434083ns    -> ~11.5 GB/s

### Ideas to go faster

- NPU
  - Only copy Quote + Backslash to NPU for string_index
  - Fastpath if no escapes
  - Vectorize bitmask operations
  - Better NPU (clockspeed + more compute tiles)
- Structural character index
  - Make `StructuralCharacter` struct a tagged 32-bit integer (padded struct is 128-bits?)
    - 3 bit tag for character type (6 options): '{', '}', '[', ']', ':', ','
    - 29 bits for position, this allows for chunks of max. ~500MB.
    - This should also speed up automaton which needs to fetch all these characters (cpu cache)
  - Problem: when moving to NPU:
    - 1/4th of the characters in a block are structural -> fits in input size
      - According to my statistics this has no overflows on bestbuy and walmart datasets i have
    - per block check for count, better to do optimistically: return boolean from npu in case of overflow
    - when overflow, redo that block on CPU with std::vector or something (slow path)
    - memory is more scattered, when iterating, need to jump from end of current block to next block if hit 0
- Automaton
  - Add tail-skip after first quote match
    - Tried this multiple times, somehow it doesn't give an improvement on benchmarked files
      - Maybe after cache issue it now does?
  - Add character type skipper to iterator/structural character index?
    - Avoids the automaton iterating, but more logic / branch in iterator
