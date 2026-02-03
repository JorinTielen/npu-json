#!/usr/bin/env just --justfile

set dotenv-filename := "vitis.env"
set dotenv-load

_venv:
  [ -d .venv ] || python3 -m venv .venv

# setup required tooling
setup-python: _venv
  # Install MLIR-AIE Python required packages
  .venv/bin/pip install -r requirements.txt
  # INstall MLIR-AIE dependencies
  .venv/bin/pip install -r requirements-mlir.txt
  # Install extras package seperately due to package prefix environment variable
  HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie .venv/bin/pip install -r requirements-extras.txt

# Setup meson build directory
setup-meson:
  meson setup build

# build main application
build:
  meson compile -C build

# run nj on a specic JSON file
run json *args:
  ./build/nj datasets/{{json}}.json {{args}}

# run unit tests
test-unit:
  ninja -C build test

# run npu tests
test-npu:
  ./build/test/npu_tests build/src/aie/json.xclbin build/src/aie/json-insts.txt

# clean build directory
clean:
  rm -rf ./build
  meson setup build