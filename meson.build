project(
  'npu-json',
  ['cpp'],
  version : '0.1.0',
  default_options : ['warning_level=3', 'cpp_std=c++20'],
)

subdir('src/aie')

project_source_files = [
  'src/npu-json/jsonpath/byte-code.cpp',
  'src/npu-json/jsonpath/lexer.cpp',
  'src/npu-json/jsonpath/parser.cpp',
  'src/npu-json/npu/kernel.cpp',
  'src/npu-json/npu/pipeline.cpp',
  'src/npu-json/structural/classifier.cpp',
  'src/npu-json/util/tracer.cpp',
  'src/npu-json/engine.cpp',
  'src/npu-json/result-set.cpp',
]

project_dependencies = [
  dependency('xrt'),
]

project_include_directories = [
  'src'
]

build_args = []

add_project_arguments(['-march=native'], language: 'cpp')

npu_json_lib = shared_library(
  'npu_json_lib',
  project_source_files,
  include_directories : project_include_directories,
  dependencies : project_dependencies,
  install : true
)

executable(
  'nj',
  ['src/npu-json/main.cpp'] + project_source_files,
  include_directories : project_include_directories,
  dependencies : project_dependencies,
  c_args : build_args,
)

catch2_project = subproject('catch2', required: true)
catch2_dep= catch2_project.get_variable('catch2_with_main_dep')
subdir('test')
