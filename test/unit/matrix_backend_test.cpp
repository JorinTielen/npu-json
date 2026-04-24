#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_all.hpp>

#include <npu-json/engine.hpp>
#include <npu-json/jsonpath/parser.hpp>
#include <npu-json/npu/chunk-index.hpp>
#include <npu-json/matrix/ops.hpp>
#include <npu-json/matrix/kernel.hpp>
#include <npu-json/matrix/pipeline.hpp>

#ifdef NPU_JSON_MATRIX_BACKEND

namespace {

bool is_structural(char c) {
  switch (c) {
    case '{':
    case '}':
    case '[':
    case ']':
    case ':':
    case ',':
      return true;
    default:
      return false;
  }
}

std::vector<uint32_t> build_reference_structural_index(std::string_view input) {
  std::vector<uint32_t> structurals;
  structurals.reserve(input.size() / 8);

  bool in_string = false;
  bool escaped = false;

  for (size_t i = 0; i < input.size(); i++) {
    auto c = input[i];

    if (in_string) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (c == '\\') {
        escaped = true;
        continue;
      }
      if (c == '"') {
        in_string = false;
      }
      continue;
    }

    if (c == '"') {
      in_string = true;
      continue;
    }

    if (is_structural(c)) {
      structurals.push_back(static_cast<uint32_t>(i));
    }
  }

  return structurals;
}

std::vector<uint32_t> collect_chunk_structurals(const npu::ChunkIndex &index) {
  auto count = index.block.structural_characters_count;
  return std::vector<uint32_t>(
    index.block.structural_characters.begin(),
    index.block.structural_characters.begin() + count
  );
}

size_t run_query_count(std::string_view json, std::string_view query_source) {
  auto parser = jsonpath::Parser();
  auto query = parser.parse(std::string(query_source));
  auto engine = Engine(*query, json);
  auto result_set = engine.run_query();
  return result_set->get_result_count();
}

} // namespace

// --- Matrix Operation Tests ---

TEST_CASE("matrix gemm computes correct product") {
  matrix::Matrix A(2, 3, 0.0f);
  A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f;
  A(1, 0) = 4.0f; A(1, 1) = 5.0f; A(1, 2) = 6.0f;

  matrix::Matrix B(3, 2, 0.0f);
  B(0, 0) = 7.0f;  B(0, 1) = 8.0f;
  B(1, 0) = 9.0f;  B(1, 1) = 10.0f;
  B(2, 0) = 11.0f; B(2, 1) = 12.0f;

  matrix::Matrix C(2, 2, 0.0f);
  matrix::gemm(A, B, C);

  // [1,2,3] * [7,8]   = [58, 64]
  // [4,5,6]   [9,10]    [139,154]
  //           [11,12]
  REQUIRE(C(0, 0) == Catch::Approx(58.0f));
  REQUIRE(C(0, 1) == Catch::Approx(64.0f));
  REQUIRE(C(1, 0) == Catch::Approx(139.0f));
  REQUIRE(C(1, 1) == Catch::Approx(154.0f));
}

TEST_CASE("matrix gemm with alpha and beta") {
  matrix::Matrix A(1, 2, 0.0f);
  A(0, 0) = 2.0f; A(0, 1) = 3.0f;

  matrix::Matrix B(2, 1, 0.0f);
  B(0, 0) = 5.0f;
  B(1, 0) = 7.0f;

  matrix::Matrix C(1, 1, 0.0f);
  C(0, 0) = 10.0f;

  matrix::gemm(A, B, C, 0.5f, 2.0f);

  // C = 0.5 * (2*5 + 3*7) + 2*10 = 0.5*31 + 20 = 15.5 + 20 = 35.5
  REQUIRE(C(0, 0) == Catch::Approx(35.5f));
}

TEST_CASE("matrix step activation thresholds correctly") {
  matrix::Matrix M(2, 3, 0.0f);
  M(0, 0) = 0.0f; M(0, 1) = 0.7f; M(0, 2) = 0.51f;
  M(1, 0) = -1.0f; M(1, 1) = 2.0f; M(1, 2) = 0.49f;

  matrix::step_activation(M, 0.5f);

  REQUIRE(M(0, 0) == 0.0f);
  REQUIRE(M(0, 1) == 1.0f);
  REQUIRE(M(0, 2) == 1.0f);
  REQUIRE(M(1, 0) == 0.0f);
  REQUIRE(M(1, 1) == 1.0f);
  REQUIRE(M(1, 2) == 0.0f);
}

TEST_CASE("matrix prefix xor produces correct result") {
  // Single bit at position 0: all bits set to 1
  REQUIRE(matrix::prefix_xor(0b1) == 0xFFFFFFFFFFFFFFFFULL);
  // Single bit at position 63
  uint64_t mask63 = 1ULL << 63;
  REQUIRE(matrix::prefix_xor(mask63) == mask63);

  // Two bits at positions 10 and 20
  uint64_t two_bits = (1ULL << 10) | (1ULL << 20);
  uint64_t result = matrix::prefix_xor(two_bits);
  // Before position 10: all 0
  for (int i = 0; i < 10; i++) {
    REQUIRE(((result >> i) & 1) == 0);
  }
  // From position 10 to 19: all 1 (XOR with the first bit)
  for (int i = 10; i < 20; i++) {
    REQUIRE((result >> i) & 1);
  }
  // From position 20 onward: all 0 (XOR toggled back)
  for (int i = 20; i < 40; i++) {
    REQUIRE(((result >> i) & 1) == 0);
  }
}

TEST_CASE("matrix pack column to bitmask") {
  uint8_t block[8] = {'{', '"', 'a', '"', ':', '1', '}', ' '};

  matrix::Matrix W = matrix::build_weight_matrix();

  uint64_t masks[matrix::CHAR_CLASS_COUNT];
  matrix::character_match(reinterpret_cast<const char*>(block), 8, W, masks);

  REQUIRE(masks[static_cast<size_t>(matrix::CharClass::BraceOpen)] == 0b00000001);
  REQUIRE(masks[static_cast<size_t>(matrix::CharClass::BraceClose)] == 0b01000000);
  REQUIRE(masks[static_cast<size_t>(matrix::CharClass::Colon)] == 0b00010000);
  REQUIRE(masks[static_cast<size_t>(matrix::CharClass::Quote)] == 0b00001010);
}

TEST_CASE("matrix compress bitmask extracts positions") {
  uint64_t mask = (1ULL << 0) | (1ULL << 4) | (1ULL << 6);
  uint32_t tail[3];

  auto count = matrix::compress_bitmask(tail, mask, 100);

  REQUIRE(count == 3);
  REQUIRE(tail[0] == 100);
  REQUIRE(tail[1] == 104);
  REQUIRE(tail[2] == 106);
}

TEST_CASE("matrix character match detects all structural characters") {
  std::string json = R"({"a":1,"b":"x, {[]}:","c":[true,{"d":"\\\"z"}]})";
  std::string padded(json);
  padded.resize(64, ' ');

  matrix::Matrix W = matrix::build_weight_matrix();
  uint64_t masks[matrix::CHAR_CLASS_COUNT];
  matrix::character_match(padded.data(), 64, W, masks);

  uint64_t brace_open = masks[static_cast<size_t>(matrix::CharClass::BraceOpen)];
  uint64_t brace_close = masks[static_cast<size_t>(matrix::CharClass::BraceClose)];
  uint64_t quote = masks[static_cast<size_t>(matrix::CharClass::Quote)];
  uint64_t backslash = masks[static_cast<size_t>(matrix::CharClass::Backslash)];

  REQUIRE(brace_open != 0);
  REQUIRE(brace_close != 0);
  REQUIRE(quote != 0);
  REQUIRE(backslash != 0);
}

// --- Matrix Kernel Tests ---

TEST_CASE("matrix kernel indexes one chunk correctly") {
  auto json = std::string(R"({"a":1,"b":"x, {[]}:","c":[true,{"d":"\\\"z"}]})");

  auto kernel = std::make_unique<matrix::MatrixKernel>(json);
  auto chunk_index = std::make_unique<npu::ChunkIndex>();

  kernel->call(chunk_index.get(), 0, []{});

  auto expected = build_reference_structural_index(json);
  auto actual = collect_chunk_structurals(*chunk_index);
  REQUIRE(actual == expected);
}

TEST_CASE("matrix kernel handles chunk carries across boundaries") {
  auto json = std::string(Engine::CHUNK_SIZE + 64, ' ');
  json[0] = '{';
  json[1] = '"';
  json[2] = 'a';
  json[3] = '"';
  json[4] = ':';

  auto boundary = Engine::CHUNK_SIZE;
  json[boundary - 3] = '"';
  json[boundary - 2] = 'x';
  json[boundary - 1] = '\\';
  json[boundary] = '"';
  json[boundary + 1] = 'y';
  json[boundary + 2] = '"';
  json[boundary + 3] = ':';
  json[boundary + 4] = '1';
  json[boundary + 5] = ',';
  json[boundary + 6] = '[';
  json[boundary + 7] = ']';
  json[boundary + 8] = '}';

  auto kernel = std::make_unique<matrix::MatrixKernel>(json);
  auto first_chunk_index = std::make_unique<npu::ChunkIndex>();
  auto second_chunk_index = std::make_unique<npu::ChunkIndex>();

  kernel->call(first_chunk_index.get(), 0, []{});
  kernel->call(second_chunk_index.get(), Engine::CHUNK_SIZE, []{});

  REQUIRE(first_chunk_index->ends_with_escape());
  REQUIRE(first_chunk_index->ends_in_string());
  REQUIRE_FALSE(second_chunk_index->ends_with_escape());
  REQUIRE_FALSE(second_chunk_index->ends_in_string());

  auto expected = build_reference_structural_index(json);

  auto actual_first = collect_chunk_structurals(*first_chunk_index);
  auto actual_second = collect_chunk_structurals(*second_chunk_index);
  actual_first.insert(actual_first.end(), actual_second.begin(), actual_second.end());

  REQUIRE(actual_first == expected);
}

TEST_CASE("matrix kernel matches CPU kernel on representative inputs") {
  auto json = std::string(R"({"a":1,"b":"x, {[]}:","c":[true,{"d":"\\\"z"}]})");

  auto matrix_kernel = std::make_unique<matrix::MatrixKernel>(json);
  auto cpu_kernel = std::make_unique<npu::Kernel>(json);

  auto matrix_index = std::make_unique<npu::ChunkIndex>();
  auto cpu_index = std::make_unique<npu::ChunkIndex>();

  matrix_kernel->call(matrix_index.get(), 0, []{});
  cpu_kernel->call(cpu_index.get(), 0, []{});

  auto matrix_structurals = collect_chunk_structurals(*matrix_index);
  auto cpu_structurals = collect_chunk_structurals(*cpu_index);

  REQUIRE(matrix_structurals == cpu_structurals);
}

// --- End-to-End Query Tests ---

TEST_CASE("matrix backend matches expected counts on representative queries") {
  auto people_json = std::string(R"({
  "people": [
    {
      "name": "Ann",
      "age": 30,
      "tags": ["dev", "ops"]
    },
    {
      "name": "Bob",
      "age": 25,
      "tags": ["dev"]
    },
    {
      "name": "Cara",
      "age": 40,
      "tags": []
    }
  ],
  "active": true
})");

  auto store_json = std::string(R"({
  "store": {
    "book": [
      {
        "category": "fiction",
        "title": "A",
        "price": 8
      },
      {
        "category": "history",
        "title": "B",
        "price": 12
      }
    ],
    "bicycle": {
      "color": "red",
      "price": 19
    }
  },
  "ids": [1, 2, 3, 4]
})");

  auto events_json = std::string(R"([
  {
    "type": "click",
    "ok": true,
    "meta": {
      "x": 1,
      "y": 2
    }
  },
  {
    "type": "scroll",
    "ok": false,
    "meta": {
      "x": 3,
      "y": 4
    }
  },
  {
    "type": "click",
    "ok": true,
    "meta": {
      "x": 5,
      "y": 6
    }
  }
])");

  REQUIRE(run_query_count(people_json, "$.people[*].name") == 3);
  REQUIRE(run_query_count(store_json, "$.store.book[1:2].price") == 1);
  REQUIRE(run_query_count(store_json, "$.ids[1:3]") == 2);
  REQUIRE(run_query_count(events_json, "$[*].meta.x") == 3);
}

#else

TEST_CASE("matrix backend tests are skipped for non-matrix builds") {
  SUCCEED("Build without NPU_JSON_MATRIX_BACKEND");
}

#endif