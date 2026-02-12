# Get the latest release version
latest_tag_with_v=$(curl -s "https://api.github.com/repos/Xilinx/mlir-aie/releases/latest" | jq -r '.tag_name')
latest_tag="${latest_tag_with_v#v}"

# Install IRON library and mlir-aie from the latest stable release
.venv/bin/python3 -m pip install mlir_aie==${latest_tag} -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/${latest_tag_with_v}
git checkout $latest_tag_with_v

# Install Peano from llvm-aie wheel
.venv/bin/python3 -m pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
