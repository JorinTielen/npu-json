project(
  'npu-json',
  ['cpp'],
  version : '0.1.0',
  default_options : ['warning_level=3'],
)

subdir('src/aie')

project_source_files = [
  'src/npu-json/main.cpp',
  'src/npu-json/engine.cpp',
  'src/npu-json/npu/indexer.cpp'
]

project_dependencies = [
  dependency('xrt'),
]

build_args = []

add_project_arguments(['-O3', '-march=native'], language: 'cpp')

executable(
  'nj',
  project_source_files,
  include_directories : 'src',
  dependencies : project_dependencies,
  c_args : build_args,
)
