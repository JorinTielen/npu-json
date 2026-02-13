#pragma once

#include <memory>
#include <stack>
#include <string>

#include <npu-json/jsonpath/byte-code.hpp>
#include <npu-json/jsonpath/query.hpp>
#include <npu-json/result-set.hpp>

// Forward declares
namespace npu {
class StructuralIndex;
class StructuralIndexer;
class PipelinedIterator;
}

namespace structural {
class Iterator;
}

// Global types
enum class StructureType {
  Object,
  Array
};


struct StructuralCharacter {
  char c;
  uint32_t pos;
};

struct StackFrame {
  size_t instruction_pointer; // The instruction being executed at this depth of the JSON (sub-)tree
  StructureType structure_type; // The structure type of the current object (at this depth)
  size_t depth; // The current depth of the JSON (sub-)tree we are executing on

  bool matched_key_at_depth = false; // Used for FindKey state tail-skip
  size_t array_position = 0; // Used for FindIndex & FindRange

  StackFrame(
    size_t instruction_pointer,
    StructureType structure_type,
    size_t depth,
    bool matched_key_at_depth,
    size_t array_position)
    : instruction_pointer(instruction_pointer)
    , structure_type(structure_type)
    , depth(depth)
    , matched_key_at_depth(matched_key_at_depth)
    , array_position(array_position) {}
};

// JSONPath engine
class Engine {
public:
  // Must be kept in sync with the DATA_CHUNK_SIZE AND DATA_BLOCK_SIZE in `src/aie/gen_mlir_design.py`.
  static constexpr size_t BLOCK_SIZE = 16 * 1024;
  static constexpr size_t BLOCKS_PER_CHUNK = 512;
  static constexpr size_t CHUNK_SIZE = BLOCKS_PER_CHUNK * BLOCK_SIZE;

  Engine(jsonpath::Query &query, std::string_view json);
  ~Engine();

  std::shared_ptr<ResultSet> run_query();
private:
  std::unique_ptr<jsonpath::ByteCode> byte_code;
  jsonpath::Instruction *instructions;
  std::unique_ptr<npu::PipelinedIterator> iterator;

  // Engine execution state
  bool executing_query = false;
  std::stack<StackFrame> stack;

  size_t current_instruction_pointer = 0;
  size_t current_depth = 0;
  StructureType current_structure_type;
  uint32_t *previous_structural;

  bool current_matched_key_at_depth = false;
  size_t current_array_position = 0;
  std::string_view json;

  // State implementations
  void handle_open_structure(StructureType structure_type);
  void handle_find_key(const std::string_view search_key);
  void handle_find_range(const size_t start, const size_t end);
  void handle_wildcard();
  void handle_record_result(ResultSet &result_set);

  // State movement functions
  void advance();
  void fallback();
  void abort(uint32_t* structural_character);
  void back();

  // Helper functions
  void enter(StructureType structure_type);
  void exit(StructureType structure_type);
  void restore_state_from_stack(StackFrame &frame);
  void pass_structural(uint32_t* structural_character);
  uint32_t *passed_previous_structural();
  size_t calculate_query_depth();

  uint32_t *skip_current_structure(StructureType structure_type);
};
