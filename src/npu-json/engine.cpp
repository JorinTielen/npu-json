#include <cassert>
#include <cstring>
#include <iostream>
#include <stack>
#include <variant>
#include <memory>

#include <npu-json/jsonpath/byte-code.hpp>
#include <npu-json/jsonpath/query.hpp>
#include <npu-json/npu/indexer.hpp>
#include <npu-json/structural/iterator.hpp>
#include <npu-json/util/debug.hpp>
#include <npu-json/engine.hpp>
#include <npu-json/error.hpp>
#include "engine.hpp"

Engine::Engine(jsonpath::Query &query) {
  byte_code = std::make_unique<jsonpath::ByteCode>();
  byte_code->compile_from_query(query);
  stack = std::stack<StackFrame>();
}

Engine::~Engine() {}

std::shared_ptr<ResultSet> Engine::run_query_on(std::string &json) {
  auto structural_iterator = std::make_unique<structural::Iterator>(json);
  auto result_set = std::make_shared<ResultSet>();

  // $.statuses[*].user.lang
  // OPEN_OBJECT, FIND_KEY "statuses", OPEN_OBJECT, WILDCARD, FIND_KEY "user", OPEN_OBJECT, FIND_KEY "lang", RECORD_RESULT

  while (auto structural_character = structural_iterator->get_next_structural_character()) {
    auto current_instruction = byte_code->instructions[current_instruction_pointer];

    using jsonpath::Opcode;
    switch (current_instruction.opcode) {
      case Opcode::OpenObject: {
        handle_open_structure(
          structural_character.value(),
          StructureType::Object,
          *structural_iterator.get()
        );
        break;
      }
      case Opcode::OpenArray: {
        handle_open_structure(
          structural_character.value(),
          StructureType::Array,
          *structural_iterator.get()
        );
        break;
      }
      case Opcode::FindKey: {
        assert(current_instruction.search_key.has_value());
        handle_find_key(
          structural_character.value(),
          json,
          current_instruction.search_key.value(),
          *structural_iterator.get()
        );
        break;
      }
      case Opcode::WildCard: {
        handle_wildcard(
          structural_character.value(),
          *structural_iterator.get()
        );
        break;
      }
      case Opcode::RecordResult: {
        handle_record_result(
          structural_character.value(),
          json,
          *structural_iterator.get(),
          *result_set.get()
        );
        break;
      }
      default:
        throw std::logic_error("Unimplemented opcode");
    }
  }

  return result_set;
}

void Engine::handle_open_structure(StructuralCharacter structural_character,
                                   StructureType structure_type, structural::Iterator &iterator) {
  if ((structural_character.c == '{' && structure_type == StructureType::Object) ||
      (structural_character.c == '[' && structure_type == StructureType::Array)) {
    current_depth++;
    current_structure_type = structure_type;
    stack.emplace(current_instruction_pointer, current_structure_type, current_depth);
    advance();
  } else {
    fallback(iterator);
  }
}

bool check_key_match(std::string &json, size_t colon_position, std::string &search_key) {
  // {"a" : 1 }
  // 0123456789
  //    ^
  size_t current_position = colon_position;
  while (json[current_position] != '"') {
    assert(current_position > 0);
    current_position--;
  }

  if (current_position + 1 < search_key.length()) return false;

  size_t start_position = current_position - search_key.length();

  // std::cout << "check_key_match" << std::endl;
  // std::cout << std::string(json.c_str() + start_position, search_key.length()) << std::endl;

  auto match = memmem(
    json.c_str() + start_position,
    search_key.length(),
    search_key.c_str(),
    search_key.length()
  );

  return match != nullptr;
}

void Engine::handle_find_key(StructuralCharacter structural_character, std::string &json,
                             std::string &search_key, structural::Iterator &iterator) {
  size_t search_depth = current_depth;
  switch (structural_character.c) {
    case ':': {
      // Only check keys at the current depth
      if (current_depth == search_depth) {
        // Match the key before the colon
        auto matched = check_key_match(json, structural_character.pos, search_key);
        if (matched) {
          possible_result_start_position = structural_character.pos + 1;
          advance();
        }
      }
    }
    case ',':
      // Last key did not match, we are still in the FindKey state, so skip the comma.
      break;
    case '{':
    case '[':
      current_depth++;
      break;
    case '}':
    case ']': {
      current_depth--;
      if (current_depth == search_depth) {
        // We matched no keys and reached the end of the object, so we fall back.
        fallback(iterator, false);
        return;
      }
    }
  }
}

void Engine::handle_wildcard(StructuralCharacter structural_character, structural::Iterator &iterator) {
  if (structural_character.c == '{') {
    current_depth++;
    current_structure_type = StructureType::Object;
    stack.emplace(current_instruction_pointer, current_structure_type, current_depth);
    advance();
  } else if (structural_character.c == '[') {
    current_depth++;
    current_structure_type = StructureType::Array;
    stack.emplace(current_instruction_pointer, current_structure_type, current_depth);
    advance();
  } else if (structural_character.c == '}' || structural_character.c == ']') {
    // End of wildcard, fall back to a previous one if there is one.
    fallback(iterator);
  } else if (structural_character.c == ',') {
    // Skip the comma on wildcards
    return;
  } else {
    throw EngineError("Unexpected JSON structural character");
  }
}

void Engine::handle_record_result(StructuralCharacter structural_character, std::string &json,
                                  structural::Iterator &iterator, ResultSet &result_set) {
  std::cout << "record result: " << possible_result_start_position << ", " << structural_character.pos << std::endl;
  std::cout << std::string(json, possible_result_start_position, structural_character.pos - possible_result_start_position) << std::endl;
  result_set.record_result(possible_result_start_position, structural_character.pos);

  if ((structural_character.c == '}' && current_structure_type == StructureType::Object) ||
      (structural_character.c == ']' && current_structure_type == StructureType::Array)) {
    fallback(iterator, false);
  } else if (structural_character.c == ',') {
    fallback(iterator);
  } else {
    throw EngineError("Unexpected JSON structural character");
  }
}

void Engine::advance() {
  assert(current_instruction_pointer < byte_code->instructions.size());

  current_instruction_pointer++;
}

// Unwinds engine bytecode execution towards the nearest Wildcard instruction
// if there is one, tail-skipping JSON structures as depth is reduced.
void Engine::fallback(structural::Iterator &iterator, bool skip_first) {
  std::cout << "fallback" << std::endl;
  std::cout << skip_first << std::endl;
  std::cout << stack.size() << std::endl;
  std::cout << current_depth << std::endl;
  std::cout << current_instruction_pointer << std::endl;
  auto skipped_first = false;
  // TODO: While should be other way around -> pop stack until wildcard reached -> that gives depth & ip we want
  while (current_instruction_pointer > 0) {
    if (!stack.empty()) {
      auto frame = stack.top();
      restore_state_from_stack(frame);
      if (skip_first || skipped_first) {
        skip_current_structure(iterator, frame.structure_type);
        skipped_first = true;
      }
      stack.pop();
    }

    current_depth--;
    current_instruction_pointer--;

    auto current_instr = byte_code->instructions[current_instruction_pointer];
    if (current_instr.opcode == jsonpath::Opcode::WildCard) break;
  }

  current_instruction_pointer++;
}

void Engine::restore_state_from_stack(StackFrame &frame) {
  current_structure_type = frame.structure_type;
}

void Engine::skip_current_structure(structural::Iterator &iterator, StructureType structure_type) {
  size_t skip_depth = current_depth;

  std::optional<StructuralCharacter> structural_character;
  while (skip_depth >= current_depth) {
    structural_character = iterator.get_next_structural_character();
    if (!structural_character.has_value()) throw EngineError("Unexpected end of JSON");

    switch (structural_character.value().c) {
      case '{':
        skip_depth++;
        break;
      case '}':
        skip_depth--;
        break;
      case '[':
        skip_depth++;
        break;
      case ']':
        skip_depth--;
        break;
      case ':':
        break;
      case ',':
        break;
      default:
        __builtin_unreachable();
    }
  }

  // TODO: Remove check if slow
  if ((structure_type == StructureType::Object && structural_character.value().c != '}') ||
      (structure_type == StructureType::Array  && structural_character.value().c != ']')) {
    throw EngineError("Unbalanced JSON structures");
  }
}
