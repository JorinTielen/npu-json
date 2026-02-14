# npu-json

A JSON parser and JSONPath query engine accelerated on the AMD XDNA NPU
architecture. Developed as part of my Master Embedded Systems Graduation
Project "Accelerating JSON Processing on SIMD-based NPU Architectures".

## Build instructions

### 0. MLIR-AIE development prerequisites

Before you are able to build and run the AI Engine application, it is
required to setup the required tooling such as the XDNA™ Driver.

See the [README in the MLIR-AIE repository](https://github.com/Xilinx/mlir-aie/blob/main/README.md)
for details.

### 1. Setup your environment

Source the required script to use the XRT tooling such as `xrt-smi`:

```sh
source /opt/xilinx/xrt/setup.sh
```

### 2. Setup Python development environment

Now we can set up the Python virtual environment:

```sh
just setup-python
```

We need to source MLIR-AIE and LLVM (AIE fork) tools which were downloaded
as Python dependencies. There is a script for this in the repository:

```sh
source env.sh
```

Repeat this step every time you are working working in a new shell.

### 3. Build the project

Now we should be able to build with Meson. Choose one of the presets:

```sh
just build-hx370
```

```sh
just build-7940hs
```

You can also configure the parameters manually:

```sh
just setup-meson-params <block_size> <blocks_per_chunk> <npu_cols> <npu_device> <aie_target>
just build
```

Example (xdna2):

```sh
just setup-meson-params 16384 512 8 npu2 aie2p-none-unknown-elf
just build
```

Example (xdna1):

```sh
just setup-meson-params 16384 1024 4 npu1 aie2-none-unknown-elf
just build
```

## Usage

First download the datasets with the instructions in the `datasets/` directory.

We can run a query on one of the benchmark datasets as follows:

```sh
just run twitter "\$[*].user.lang"
```
