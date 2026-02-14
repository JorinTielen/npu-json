#!/usr/bin/env just --justfile

set dotenv-filename := "vitis.env"
set dotenv-load

_venv:
  [ -d .venv ] || python3 -m venv .venv

# setup required tooling
setup-python: _venv
  # Install MLIR-AIE Python required packages
  # .venv/bin/pip install -r requirements.txt
  ./install-iron.sh
  # INstall MLIR-AIE dependencies
  .venv/bin/pip install -r requirements-mlir.txt
  # Install extras package seperately due to package prefix environment variable
  HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie .venv/bin/pip install -r requirements-extras.txt

# Setup meson build directory
setup-meson:
  meson setup build

# setup meson build directory with NPU parameters
setup-meson-params block_size blocks_per_chunk npu_cols npu_device aie_target:
  meson setup build --reconfigure \
    -Dnpu_block_size={{block_size}} \
    -Dnpu_blocks_per_chunk={{blocks_per_chunk}} \
    -Dnpu_num_cols={{npu_cols}} \
    -Dnpu_device={{npu_device}} \
    -Daie_target_triple={{aie_target}}

# build main application
build *args:
  meson compile -C build {{args}}

# build with hx370 defaults (xdna2)
build-hx370:
  just setup-meson-params 16384 512 8 npu2 aie2p-none-unknown-elf
  just build

# build with 7940hs defaults (xdna1)
build-7940hs:
  just setup-meson-params 16384 1024 4 npu1 aie2-none-unknown-elf
  just build

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
