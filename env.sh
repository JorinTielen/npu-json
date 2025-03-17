#!/usr/bin/env bash

# TODO: Add AIE_TOOLS_ROOT to path as well (for outside Just?)

if (pip show mlir_aie &>/dev/null) && (pip show llvm-aie &>/dev/null); then
  MLIR_AIE_INSTALL_DIR="$(pip show mlir_aie | grep ^Location: | awk '{print $2}')/mlir_aie"
  LLVM_AIE_INSTALL_DIR="$(pip show llvm-aie | grep ^Location: | awk '{print $2}')/llvm-aie"
fi

# MLIR-AIE still does not expose these binaries through the virtual env and have to be added to the PATH manually.
if [ -v LLVM_AIE_INSTALL_DIR ] && [ -v MLIR_AIE_INSTALL_DIR ]; then
  export PATH=${LLVM_AIE_INSTALL_DIR}/bin:${MLIR_AIE_INSTALL_DIR}/bin:${PATH}
  export PYTHONPATH=${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH}
  export LD_LIBRARY_PATH=${LLVM_AIE_INSTALL_DIR}/bin:${MLIR_AIE_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
  export PEANO_INSTALL_DIR=${LLVM_AIE_INSTALL_DIR}
fi

# Add `pth` file pointing to MLIR-AIE python package, to support tooling such as Pylance.
venv_site_packages=`python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'`
[ -f $venv_site_packages/mlir-aie.pth ] || echo ${MLIR_AIE_INSTALL_DIR}/python > $venv_site_packages/mlir-aie.pth
