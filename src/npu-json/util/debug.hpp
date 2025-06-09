#pragma once

#include <algorithm>
#include <cstdint>
#include <bitset>
#include <iostream>
#include <vector>

#include <npu-json/jsonpath/byte-code.hpp>

inline void print_input_and_index(const char* input, const uint64_t *index, const size_t at = 0) {
  auto index_bitset = std::bitset<64>(index[at]);
  auto index_bitset_str = index_bitset.to_string();
  std::reverse(index_bitset_str.begin(), index_bitset_str.end());
  auto input_str = std::string(input + at * 64, 64);
  std::replace(input_str.begin(), input_str.end(), '\n', ' ');
  std::replace(input_str.begin(), input_str.end(), '\r', ' ');
  std::replace(input_str.begin(), input_str.end(), '\t', ' ');
  std::cout << "data (64 bytes at position " << at << "):" << std::endl;
  std::cout << "input: |" << input_str << "|" << std::endl;
  std::cout << "index: |" << index_bitset_str << "|" << std::endl;
}

inline void print_structural_classifier_block(uint32_t structural) {
  auto structural_bitset = std::bitset<32>(structural);
  auto structural_bitset_str = structural_bitset.to_string();
  std::reverse(structural_bitset_str.begin(), structural_bitset_str.end());
  std::cout << "structural block (32bits):" << std::endl;
  std::cout << "block: |" << structural_bitset_str << "|" << std::endl;
}

inline void print_carry_index(const uint32_t *index, const size_t at = 0) {
  auto index_bitset = std::bitset<8>();
  for (size_t i = 0; i < 8; i++) {
    index_bitset[i] = index[i];
  }
  auto index_bitset_str = index_bitset.to_string();
  std::reverse(index_bitset_str.begin(), index_bitset_str.end());
  std::cout << "carries (8 bits at position " << at << "):" << std::endl;
  std::cout << "carry: |" << index_bitset_str << "|" << std::endl;
}

inline void print_structural_character_index(const std::vector<StructuralCharacter> &index) {
  for (auto c : index) {
    std::cout << "{ char: '" << c.c << "', pos: " << c.pos << " }" << std::endl;
  }
}

inline void print_byte_code(const std::vector<jsonpath::Instruction> &instructions) {
  size_t ip = 0;
  for (auto instruction : instructions) {
    switch (instruction.opcode) {
      case jsonpath::Opcode::FindIndex:
        std::cout << ip << ": " << "FindIndex" << std::endl; break;
      case jsonpath::Opcode::FindRange:
        std::cout << ip << ": " << "FindRange" << std::endl; break;
      case jsonpath::Opcode::FindKey:
        std::cout << ip << ": " << "FindKey" << std::endl; break;
      case jsonpath::Opcode::OpenArray:
        std::cout << ip << ": " << "OpenArray" << std::endl; break;
      case jsonpath::Opcode::OpenObject:
        std::cout << ip << ": " << "OpenObject" << std::endl; break;
      case jsonpath::Opcode::WildCard:
        std::cout << ip << ": " << "WildCard" << std::endl; break;
      case jsonpath::Opcode::RecordResult:
        std::cout << ip << ": " << "RecordResult" << std::endl; break;
      default:
        __builtin_unreachable();
    }

    ip++;
  }
}
