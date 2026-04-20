#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_all.hpp>

#include <npu-json/engine.hpp>
#include <npu-json/jsonpath/parser.hpp>
#include <npu-json/npu/chunk-index.hpp>
#include <npu-json/npu/kernel.hpp>
#include <npu-json/npu/pipeline.hpp>

#ifdef NPU_JSON_CPU_BACKEND

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

TEST_CASE("cpu simd kernel indexes one chunk correctly") {
  auto json = std::string(R"({"a":1,"b":"x, {[]}:","c":[true,{"d":"\\\"z"}]})");

  auto kernel = std::make_unique<npu::Kernel>(json);
  auto indexer = std::make_unique<npu::PipelinedIndexer>(*kernel, json);
  auto chunk_index = std::make_unique<npu::ChunkIndex>();

  bool callback_hit = false;
  indexer->index_chunk(chunk_index.get(), [&callback_hit] {
    callback_hit = true;
  });
  indexer->wait_for_last_chunk();

  REQUIRE(callback_hit);

  auto expected = build_reference_structural_index(json);
  auto actual = collect_chunk_structurals(*chunk_index);
  REQUIRE(actual == expected);
}

TEST_CASE("cpu simd kernel handles chunk carries across boundaries") {
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

  auto kernel = std::make_unique<npu::Kernel>(json);
  auto indexer = std::make_unique<npu::PipelinedIndexer>(*kernel, json);
  auto first_chunk_index = std::make_unique<npu::ChunkIndex>();
  auto second_chunk_index = std::make_unique<npu::ChunkIndex>();

  size_t callbacks = 0;
  indexer->index_chunk(first_chunk_index.get(), [&callbacks] {
    callbacks++;
  });
  indexer->index_chunk(second_chunk_index.get(), [&callbacks] {
    callbacks++;
  });
  indexer->wait_for_last_chunk();

  REQUIRE(callbacks == 2);

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

TEST_CASE("cpu backend matches expected counts on representative queries") {
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

TEST_CASE("cpu backend tests are skipped for npu builds") {
  SUCCEED("Build without NPU_JSON_CPU_BACKEND");
}

#endif
