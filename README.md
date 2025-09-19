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

Now we should be able to build with Meson:

```sh
just setup-meson
just build
```

## Usage

First download the datasets with the instructions in the `datasets/` directory.

We can run a query on one of the benchmark datasets as follows:

```sh
just run twitter "\$[*].user.lang"
```
