#include <cassert>
#include <cstring>
#include <iostream>
#include <stack>
#include <variant>
#include <memory>

#include <npu-json/jsonpath/byte-code.hpp>
#include <npu-json/jsonpath/query.hpp>
#include <npu-json/npu/pipeline.hpp>
#include <npu-json/util/debug.hpp>
#include <npu-json/error.hpp>

#include <npu-json/engine.hpp>

Engine::Engine(jsonpath::Query &query, std::string &json) {
  byte_code = std::make_unique<jsonpath::ByteCode>();
  byte_code->compile_from_query(query);
  stack = std::stack<StackFrame>();

  iterator = std::make_unique<npu::PipelinedIterator>(json);
}

Engine::~Engine() {}

std::shared_ptr<ResultSet> Engine::run_query_on(const std::string *const json) {
  auto result_set = std::make_shared<ResultSet>();

  iterator->setup(json);

  executing_query = true;

  // print_byte_code(byte_code->instructions);

  while (executing_query) {
    auto current_instruction = byte_code->instructions[current_instruction_pointer];

    using jsonpath::Opcode;
    switch (current_instruction.opcode) {
      case Opcode::OpenObject: {
        handle_open_structure(json->c_str(), StructureType::Object);
        break;
      }
      case Opcode::OpenArray: {
        handle_open_structure(json->c_str(), StructureType::Array);
        break;
      }
      case Opcode::FindIndex: {
        assert(current_instruction.search_index.has_value());
        auto start = current_instruction.search_index.value();
        handle_find_range(json->c_str(), start, start + 1);
        break;
      }
      case Opcode::FindRange: {
        assert(current_instruction.search_range.has_value());
        auto [start, end] = current_instruction.search_range.value();
        handle_find_range(json->c_str(), start, end);
        break;
      }
      case Opcode::FindKey: {
        assert(current_instruction.search_key.has_value());
        handle_find_key(json->c_str(), current_instruction.search_key.value());
        break;
      }
      case Opcode::WildCard: {
        handle_wildcard(json->c_str());
        break;
      }
      case Opcode::RecordResult: {
        handle_record_result(json->c_str(), *result_set.get());
        break;
      }
      default:
        throw std::logic_error("Unimplemented opcode");
    }
  }

  // For finishing the last automaton trace.
  iterator->get_next_structural_character();

  iterator->reset();

  return result_set;
}

void Engine::handle_open_structure(const char *const json, StructureType structure_type) {
  auto initial_structural_character = passed_previous_structural();
  auto structural_character = initial_structural_character.has_value()
    ? initial_structural_character.value()
    : iterator->get_next_structural_character();

  if (structural_character == nullptr) {
    throw EngineError("Unexpected end of JSON");
  }

  auto query_depth = calculate_query_depth();

  auto structurals_end = iterator->get_chunk_structural_index_end_ptr();

  while (structural_character != nullptr) {
    switch (json[*(structural_character)]) {
      case '{': {
        enter(StructureType::Object);
        if (structure_type == StructureType::Object) {
          advance();
        } else {
          fallback(json);
        }
        iterator->set_chunk_structural_pos(structural_character);
        return;
      }
      case '[': {
        enter(StructureType::Array);
        if (structure_type == StructureType::Array) {
          auto next_opcode = byte_code->instructions[current_instruction_pointer + 1].opcode;
          if (next_opcode == jsonpath::Opcode::FindIndex || next_opcode == jsonpath::Opcode::FindRange) {
            // We might need to handle the first element in the array for FindIndex and FindRange.
            // Therefore, we pass the opening structural in this case.
            pass_structural(structural_character);
          }
          advance();
        } else {
          fallback(json);
        }
        iterator->set_chunk_structural_pos(structural_character);
        return;
      }
      case '}':
      case ']': {
        // What we need to do depends on the depth at which we are searching.
        // If we exit the parent structure, we abort. If the closing structural
        // is at the expected depth we can close our own structure normally.
        assert(current_depth >= query_depth - 1);
        assert(current_depth <= query_depth);
        if (current_depth == query_depth) {
          exit(json[*(structural_character)] == '}' ? StructureType::Object : StructureType::Array);
          back();
        } else {
          abort(structural_character);
        }
        iterator->set_chunk_structural_pos(structural_character);
        return;
      }
      case ':': {
        // Ignore colon if it came from a previous FindKey.
        if (current_depth == query_depth) {
          // There should never be a colon at this level in this state.
          throw EngineError("Unexpected colon");
        }
        break;
      }
      case ',': {
        auto previous_opcode = byte_code->instructions[current_instruction_pointer - 1].opcode;
        if (previous_opcode == jsonpath::Opcode::FindIndex || previous_opcode == jsonpath::Opcode::FindRange) {
          // The closing comma of the result could also be the starting comma of the next result.
          // Therefore, we need to pass it back.
          pass_structural(structural_character);
        }
        back();
        iterator->set_chunk_structural_pos(structural_character);
        return;
      }
      default:
        __builtin_unreachable();
    }

    if (structural_character < structurals_end - 1) {
      structural_character++;
    } else {
      iterator->set_chunk_structural_pos(structurals_end);
      structural_character = iterator->get_next_structural_character();
      if (structural_character != nullptr) {
        structurals_end = iterator->get_chunk_structural_index_end_ptr();
      }
    }
  }
}

bool check_key_match(const char *const json, size_t colon_position, const std::string_view search_key) {
  // {"a" : 1 }
  // 0123456789
  //    ^
  size_t current_position = colon_position - 1;
  while (json[current_position] != '"') {
    assert(current_position > 0);
    current_position--;
  }

  if (current_position < search_key.length() + 2) return false;

  size_t start_position = current_position - search_key.length();

  if (!(json[start_position - 1] == '"' && json[start_position - 2] != '\\')) return false;

  if (json[start_position] != search_key[0]) return false;

  auto match = memcmp(
    json + start_position,
    search_key.data(),
    search_key.length()
  );

  return match == 0;
}

bool is_closing_structural(char structural) {
  switch (structural) {
    case '}':
    case ']':
      return true;
    default:
      return false;
  }
}

void Engine::handle_find_key(const char *const json, const std::string_view search_key) {
  auto initial_structural_character = passed_previous_structural();
  auto structural_character = initial_structural_character.has_value()
    ? initial_structural_character.value()
    : iterator->get_next_structural_character();

  if (structural_character == nullptr) {
    throw EngineError("Unexpected end of JSON");
  }

  auto query_depth = calculate_query_depth();

  auto structurals_end = iterator->get_chunk_structural_index_end_ptr();

  // Make sure we didn't come back through an abort when tail-skipping
  if (current_matched_key_at_depth && initial_structural_character.has_value()) {
    // If already matched a key and we're not already on the closing structural, we can tail-skip.
    if (!is_closing_structural(json[*(structural_character)])) {
      fallback(json);
      return;
    }
  }

  while (structural_character != nullptr) {
    switch (json[*(structural_character)]) {
      case '{':
      case '[':
        current_depth++;
        break;
      case '}':
      case ']': {
        if (current_depth == query_depth) {
          // We matched no keys and reached the end of the object, so we abort.
          abort(structural_character);
          iterator->set_chunk_structural_pos(structural_character);
          return;
        } else {
          current_depth--;
        }
        break;
      }
      case ':': {
        // Only check keys at the correct depth
        if (current_depth == query_depth) {
          // Match the key before the colon
          auto matched = check_key_match(json, size_t(*structural_character), search_key);
          if (matched) {
            current_matched_key_at_depth = true;
            pass_structural(structural_character);
            advance();
            iterator->set_chunk_structural_pos(structural_character);
            return;
          }
        }
        break;
      }
      case ',':
        // Last key did not match, we are still in the FindKey state, so skip the comma.
        break;
      default:
        __builtin_unreachable();
    }

    if (structural_character < structurals_end - 1) {
      structural_character++;
    } else {
      iterator->set_chunk_structural_pos(structurals_end);
      structural_character = iterator->get_next_structural_character();
      if (structural_character != nullptr) {
        structurals_end = iterator->get_chunk_structural_index_end_ptr();
      }
    }
  }
}

void Engine::handle_find_range(const char *const json, const size_t start, const size_t end) {
  auto initial_structural_character = passed_previous_structural();
  auto structural_character = initial_structural_character.has_value()
    ? initial_structural_character.value()
    : iterator->get_next_structural_character();

  if (structural_character == nullptr) {
    throw EngineError("Unexpected end of JSON");
  }

  auto query_depth = calculate_query_depth();

  auto structurals_end = iterator->get_chunk_structural_index_end_ptr();

  if (initial_structural_character.has_value() && json[*(structural_character)] == '[') {
    if (current_array_position >= start && current_array_position < end) {
      current_array_position++;
      pass_structural(structural_character);
      advance();
      iterator->set_chunk_structural_pos(structural_character);
      return;
    }
    current_array_position++;

    if (structural_character < structurals_end - 1) {
      structural_character++;
    } else {
      iterator->set_chunk_structural_pos(structurals_end);
      structural_character = iterator->get_next_structural_character();
      if (structural_character != nullptr) {
        structurals_end = iterator->get_chunk_structural_index_end_ptr();
      }
    }
  }

  while (structural_character != nullptr) {
    switch (json[*(structural_character)]) {
      case '{':
      case '[':
        current_depth++;
        break;
      case '}':
      case ']': {
        if (current_depth == query_depth) {
          abort(structural_character);
          iterator->set_chunk_structural_pos(structural_character);
          return;
        } else {
          current_depth--;
        }
        break;
      }
      case ':':
        break;
      case ',':
        if (current_depth == query_depth) {
          if (current_array_position >= start && current_array_position < end) {
            current_array_position++;
            // pass_structural(structural_character);
            advance();
            iterator->set_chunk_structural_pos(structural_character);
            return;
          }
          current_array_position++;
        }
        break;
      default:
        __builtin_unreachable();
    }

    if (structural_character < structurals_end - 1) {
      structural_character++;
    } else {
      iterator->set_chunk_structural_pos(structurals_end);
      structural_character = iterator->get_next_structural_character();
      if (structural_character != nullptr) {
        structurals_end = iterator->get_chunk_structural_index_end_ptr();
      }
    }
  }
}

void Engine::handle_wildcard(const char *const json) {
  auto initial_structural_character = passed_previous_structural();
  auto structural_character = initial_structural_character.has_value()
    ? initial_structural_character.value()
    : iterator->get_next_structural_character();

  if (structural_character == nullptr) {
    throw EngineError("Unexpected end of JSON");
  }

  auto query_depth = calculate_query_depth();

  auto structurals_end = iterator->get_chunk_structural_index_end_ptr();

  while (structural_character != nullptr) {
    switch (json[*(structural_character)]) {
      case '{': {
        enter(StructureType::Object);
        break;
      }
      case '[': {
        enter(StructureType::Array);
        auto next_opcode = byte_code->instructions[current_instruction_pointer + 1].opcode;
        if (next_opcode == jsonpath::Opcode::RecordResult) {
          // If the next opcode is RecordResult, we want to record every value in the array.
          // The first value in the array needs its span to start from the opening bracket,
          // therefore we pass it.
          pass_structural(structural_character);
        }
        advance();
        iterator->set_chunk_structural_pos(structural_character);
        return;
      }
      case '}':
      case ']': {
        // What we need to do depends on the depth at which we are searching.
        // If we exit the parent structure, we abort. If the closing structural
        // is at the expected depth we can close our own structure normally.
        assert(current_depth >= query_depth);
        assert(current_depth <= query_depth + 1);
        if (current_depth == query_depth) {
          exit(json[*(structural_character)] == '{' ? StructureType::Object : StructureType::Array);
          back();
        } else {
          abort(structural_character);
        }
        iterator->set_chunk_structural_pos(structural_character);
        return;
      }
      case ':': {
        // Ignore the colon if it came from a previous Findkey.
        if (current_depth == query_depth + 1) {
          if (current_structure_type == StructureType::Object) {
            advance();
            iterator->set_chunk_structural_pos(structural_character);
            return;
          } else {
            throw EngineError("Unexpected colon in array");
          }
        }
        break;
      }
      case ',': {
        // For wildcards, we want to skip comma's and enter all child-structures.
        // Therefore, when we are an array, we should advance for the second, third, etc. element
        // after a comma at the correct depth.
        if (current_depth == query_depth) {
          if (current_structure_type == StructureType::Array) {
            auto next_opcode = byte_code->instructions[current_instruction_pointer + 1].opcode;
            if (next_opcode == jsonpath::Opcode::RecordResult) {
              // If the next opcode is RecordResult, we want to record every value in the array.
              // The value in between this comma and the next must therefore be recorded.
              pass_structural(structural_character);
            }
            advance();
            iterator->set_chunk_structural_pos(structural_character);
            return;
          }
        }
        break;
      }
      default:
        __builtin_unreachable();
    }

    if (structural_character < structurals_end - 1) {
      structural_character++;
    } else {
      iterator->set_chunk_structural_pos(structurals_end);
      structural_character = iterator->get_next_structural_character();
      if (structural_character != nullptr) {
        structurals_end = iterator->get_chunk_structural_index_end_ptr();
      }
    }
  }
}

void Engine::handle_record_result(const char *const json, ResultSet &result_set) {
  auto initial_structural_character = passed_previous_structural();
  auto structural_character = initial_structural_character.has_value()
    ? initial_structural_character.value()
    : iterator->get_next_structural_character();

  if (structural_character == nullptr) {
    throw EngineError("Unexpected end of JSON");
  }

  auto start_pos = size_t(*structural_character);
  auto query_depth = calculate_query_depth();

  auto structurals_end = iterator->get_chunk_structural_index_end_ptr();
  auto previous_opcode = byte_code->instructions[current_instruction_pointer - 1].opcode;

  while (structural_character != nullptr) {
    switch (json[*(structural_character)]) {
      case '{':
      case '[':
        // For complex results, we want to record the entire structure.
        if (initial_structural_character.has_value() &&
            initial_structural_character.value() == structural_character) {
          initial_structural_character.reset();
        } else {
          current_depth++;
        }
        break;
      case '}':
      case ']': {
        assert(current_depth > query_depth - 1);
        if (current_depth == query_depth) {
          // If this was the last key or value in the array, this closing marks the end of the result value.
          result_set.record_result(start_pos + 1, size_t(*structural_character) - 1);
          abort(structural_character);
          iterator->set_chunk_structural_pos(structural_character);
          return;
        } else {
          current_depth--;
        }
        break;
      }
      case ':': {
        // Either passed as the initial structural by FindKey, or a nested colon.
        break;
      }
      case ',': {
        if (current_depth == query_depth && size_t(*structural_character) != start_pos) {
          // Record a result at this position
          result_set.record_result(start_pos + 1, size_t(*structural_character) - 1);
          if (previous_opcode == jsonpath::Opcode::FindIndex ||
              previous_opcode == jsonpath::Opcode::FindRange) {
            // The closing comma of the result could also be the starting comma of the next result.
            // Therefore, we need to pass it back.
            pass_structural(structural_character);
          } else if (previous_opcode == jsonpath::Opcode::WildCard) {
            // HACK: Since the previous is a wildcard, we record all sub-structures.
            //       To avoid ping-ponging these passbacks we record everything without
            //       switching states.
            start_pos = size_t(*structural_character);
            break;
          }
          back();
          iterator->set_chunk_structural_pos(structural_character);
          return;
        }
        break;
      }
      default:
        __builtin_unreachable();
    }

    if (structural_character < structurals_end - 1) {
      structural_character++;
    } else {
      iterator->set_chunk_structural_pos(structurals_end);
      structural_character = iterator->get_next_structural_character();
      if (structural_character != nullptr) {
        structurals_end = iterator->get_chunk_structural_index_end_ptr();
      }
    }
  }
}

// Advance to the next state.
void Engine::advance() {
  assert(current_instruction_pointer < byte_code->instructions.size());

  stack.emplace(
    current_instruction_pointer,
    current_structure_type,
    current_depth,
    current_matched_key_at_depth,
    current_array_position
  );

  current_instruction_pointer++;
}

// Exit the current state, tail-skipping to to end of the structure.
void Engine::fallback(const char *const json) {
  assert(!stack.empty());

  auto last_structural = skip_current_structure(json, current_structure_type);
  pass_structural(last_structural);

  back();
}

// Exit the current state, allowing the previous state to handle the current token.
void Engine::abort(uint32_t* structural_character) {
  pass_structural(structural_character);

  back();
}

// Exit the current state, without tail-skipping.
void Engine::back() {
  if (stack.empty()) {
    // We exited the top state, therefore ending query execution.
    executing_query = false;
    return;
  }

  auto frame = stack.top();
  restore_state_from_stack(frame);
  stack.pop();
}

// Enters a JSON structure.
void Engine::enter(StructureType structure_type) {
  current_depth++;
  current_structure_type = structure_type;
}

// Exits a JSON structure.
// Does not restore any state from stack, make sure to call back,abort,fallback after.
void Engine::exit(StructureType structure_type) {
  if (current_depth == 0) throw EngineError("Invalid JSON");
  if (structure_type != current_structure_type) throw EngineError("Unbalanced JSON structures");

  current_depth--;
}

// Restore the engine state from a stack frame.
void Engine::restore_state_from_stack(StackFrame &frame) {
  current_depth = frame.depth;
  current_structure_type = frame.structure_type;
  current_instruction_pointer = frame.instruction_pointer;
  current_matched_key_at_depth = frame.matched_key_at_depth;
  current_array_position = frame.array_position;
}

// Pass a structural character to the next engine state.
void Engine::pass_structural(uint32_t* structural_character) {
  previous_structural = std::optional<uint32_t*>(structural_character);
}

// Retrieve the passed structural character from the previous state, if there is one.
std::optional<uint32_t*> Engine::passed_previous_structural() {
  if (previous_structural.has_value()) {
    auto s = previous_structural.value();
    previous_structural = std::optional<uint32_t*>();
    return std::optional<uint32_t*>(s);
  } else {
    return std::optional<uint32_t*>();
  }
}

size_t Engine::calculate_query_depth() {
  auto depth = 0;
  for (size_t i = 0; i <= current_instruction_pointer; i++) {
    auto instruction = byte_code->instructions[i];
    using jsonpath::Opcode;
    switch (instruction.opcode) {
      case Opcode::OpenArray:
      case Opcode::OpenObject:
      case Opcode::WildCard:
        depth++;
        break;
      default:
        break;
    }
  }

  return depth;
}

// Skip the current JSON structure.
uint32_t* Engine::skip_current_structure(const char *const json, StructureType structure_type) {
  size_t skip_depth = current_depth;

  uint32_t* structural_character = iterator->get_next_structural_character();
  auto structurals_end = iterator->get_chunk_structural_index_end_ptr();

  if (structural_character == nullptr) {
    throw EngineError("Unexpected end of JSON");
  }

  while (skip_depth >= current_depth) {
    switch (json[*(structural_character)]) {
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

    if (structural_character < structurals_end - 1) {
      structural_character++;
    } else {
      iterator->set_chunk_structural_pos(structurals_end);
      structural_character = iterator->get_next_structural_character();
      if (structural_character != nullptr) {
        structurals_end = iterator->get_chunk_structural_index_end_ptr();
      } else {
        throw EngineError("Unexpected end of JSON");
      }
    }
  }


  // TODO: Remove check if slow
  if ((structure_type == StructureType::Object && json[*(structural_character)] != '}') ||
      (structure_type == StructureType::Array  && json[*(structural_character)] != ']')) {
    throw EngineError("Unbalanced JSON structures");
  }

  return structural_character;
}
