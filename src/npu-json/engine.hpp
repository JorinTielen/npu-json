#pragma once

#include <memory>
#include <stack>

#include <npu-json/jsonpath/byte-code.hpp>
#include <npu-json/jsonpath/query.hpp>
#include <npu-json/result-set.hpp>

// Forward declares
namespace npu {
class StructuralIndex;
class StructuralIndexer;
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
  size_t pos;
};

struct StackFrame {
  size_t instruction_pointer; // The instruction being executed at this depth of the JSON (sub-)tree
  StructureType structure_type; // The structure type of the current object (at this depth)
  // std::optional<std::string> search_key; // The key we are searching for (if state is FindKey && structure type is object)
  // std::optional<size_t> search_index; // The index we are searching for (if state is FindIndex && structure type is array)
  size_t depth; // The current depth of the JSON (sub-)tree we are executing on

  StackFrame(size_t instruction_pointer, StructureType structure_type, size_t depth)
    : instruction_pointer(instruction_pointer), structure_type(structure_type), depth(depth) {}
};

// JSONPath engine
class Engine {
public:
  // Must be kept in sync with the DATA_CHUNK_SIZE AND DATA_BLOCK_SIZE in `src/aie/gen_mlir_design.py`.
  static constexpr size_t BLOCK_SIZE = 1024;
  static constexpr size_t CHUNK_SIZE = 50 * 1000 * BLOCK_SIZE;

  Engine(jsonpath::Query &query);
  ~Engine();

  std::shared_ptr<ResultSet> run_query_on(std::string& json);
private:
  std::unique_ptr<jsonpath::ByteCode> byte_code;

  // Engine execution state
  bool executing_query = false;
  std::stack<StackFrame> stack;

  size_t current_instruction_pointer = 0;
  size_t current_depth = 0;
  StructureType current_structure_type;
  std::optional<StructuralCharacter> previous_structural;

  // State implementations
  void handle_open_structure(
    StructureType structure_type,
    structural::Iterator &iterator,
    std::optional<StructuralCharacter> initial_structural_character
  );
  void handle_find_key(
    std::string &json,
    std::string &search_key,
    structural::Iterator &iterator,
    std::optional<StructuralCharacter> initial_structural_character
  );
  void handle_wildcard(
    structural::Iterator &iterator,
    std::optional<StructuralCharacter> initial_structural_character
  );
  void handle_record_result(
    std::string &json,
    structural::Iterator &iterator,
    ResultSet &result_set,
    std::optional<StructuralCharacter> initial_structural_character
  );

  // State movement functions
  void advance();
  void fallback(structural::Iterator &iterator);
  void abort(StructuralCharacter structural_character);
  void back();

  // Helper functions
  void enter(StructureType structure_type);
  void exit(StructureType structure_type);
  void restore_state_from_stack(StackFrame &frame);
  void pass_structural(StructuralCharacter structural_character);
  std::optional<StructuralCharacter> passed_previous_structural();
  size_t calculate_query_depth();

  StructuralCharacter skip_current_structure(structural::Iterator &iterator, StructureType structure_type);
};
