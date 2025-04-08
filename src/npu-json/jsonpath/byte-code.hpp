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
  WildCard,
  RecordResult
};

struct Instruction {
  Opcode opcode;
  std::optional<std::string> search_key;
  std::optional<size_t> search_index;

  Instruction(Opcode opcode) : opcode(opcode) {};
  Instruction(Opcode opcode, std::string search_key) : opcode(opcode) {
    this->search_key = std::optional<std::string>(search_key);
  };
};

class ByteCode {
public:
  void compile_from_query(Query &query);

  std::vector<Instruction> instructions = {};
};

} // namespace jsonpath