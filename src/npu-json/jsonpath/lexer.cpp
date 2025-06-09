#include <cctype>
#include <format>

#include <npu-json/error.hpp>

#include <npu-json/jsonpath/lexer.hpp>

namespace jsonpath {

bool Lexer::is_at_end() {
  return pos >= input.length();
}

Token Lexer::consume() {
  if (peeked_token.has_value()) {
    auto token = peeked_token.value();
    peeked_token.reset();
    return token;
  }

  auto token = next_token();
  return token;
}

Token Lexer::peek() {
  if (peeked_token.has_value()) {
    return peeked_token.value();
  }
  auto token = next_token();
  peeked_token.emplace(token);
  return peeked_token.value();
}

Token Lexer::next_token() {
  while (pos < input.length()) {
    switch (input[pos]) {
      case ' ': [[fallthrough]];
      case '\t': {
        advance();
        break;
      }
      case '$': return single_character_token(TokenType::Root);
      case '.': {
        if ((pos + 1 < input.length()) && input[pos + 1] == '.') {
          return two_character_token(TokenType::Descendant);
        } else {
          return single_character_token(TokenType::Member);
        }
      }
      case '[': return single_character_token(TokenType::OpenBracket);
      case ']': return single_character_token(TokenType::CloseBracket);
      case '*': return single_character_token(TokenType::Wildcard);
      case ':': return single_character_token(TokenType::Colon);
      default: {
        if (std::isalpha(input[pos])) {
          size_t start = pos;

          while (pos < input.length() && std::isalnum(input[pos]) || input[pos] == '_')
            advance();

          return build_token(TokenType::Name, start, pos);
        }

        if (std::isdigit(input[pos])) {
          size_t start = pos;

          while (pos < input.length() && std::isdigit(input[pos])) advance();

          return build_token(TokenType::Number, start, pos);
        }

        throw QueryError(std::format("Unexpected character '{}' at {}", input[pos], pos));
      }
    }
  }

  throw QueryError("Unexpected end of query");
}

Token Lexer::single_character_token(TokenType type) {
  auto start = pos;
  pos += 1;
  return Token { type, start, input.substr(start, 1) };
}

Token Lexer::two_character_token(TokenType type) {
  auto start = pos;
  pos += 2;
  return Token { type, start, input.substr(start, 2) };
}

Token Lexer::build_token(TokenType type, size_t start, size_t end) {
  return Token { type, start, input.substr(start, end - start) };
}

void Lexer::advance() {
  pos++;
}

} // namespace jsonpath
