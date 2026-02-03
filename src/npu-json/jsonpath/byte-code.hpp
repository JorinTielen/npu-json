#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <npu-json/jsonpath/query.hpp>

namespace jsonpath {

enum class Opcode {
  OpenObject,
  OpenArray,
  FindKey,
  FindIndex,
  FindRange,
  WildCard,
  RecordResult
};

struct Instruction {
  Opcode opcode;
  std::optional<std::string> search_key;
  std::optional<size_t> search_index;
  std::optional<std::tuple<size_t,size_t>> search_range;

  Instruction(Opcode opcode) : opcode(opcode) {};
  Instruction(Opcode opcode, std::string search_key) : opcode(opcode) {
    this->search_key = std::optional<std::string>(search_key);
  };
  Instruction(Opcode opcode, size_t search_index) : opcode(opcode) {
    this->search_index = std::optional<size_t>(search_index);
  };
  Instruction(Opcode opcode, size_t start, size_t end) : opcode(opcode) {
    this->search_range = std::optional<std::tuple<size_t,size_t>>(std::tuple(start, end));
  };
};

class ByteCode {
public:
  void compile_from_query(Query &query);

  std::vector<Instruction> instructions = {};
  std::vector<int> query_instruction_depth = {};
private:
  void calculate_query_depth();
};

} // namespace jsonpath
