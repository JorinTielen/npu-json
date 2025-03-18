xrt_flags = -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -luuid -lxrt_coreutil

.PHONY = all clean

all: build/multi_kernel.xclbin build/test

build/kernels.cc.o: kernels.cc
	mkdir -p ${@D}
	clang++ -O2 -std=c++23 --target=aie2-none-unknown-elf \
		-Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-deprecated-declarations \
		-Wno-logical-not-parentheses -DNDEBUG -I/opt/Xilinx/ryzen_ai-1.3.0/vitis_aie_essentials/include \
		-c $< -o $@

build/vector_scalar_add.mlir: gen_mlir_design.py build/kernels.cc.o
	mkdir -p ${@D}
	python $< --kernel vector_scalar_add --out $@

build/vector_scalar_mul.mlir: gen_mlir_design.py build/kernels.cc.o
	mkdir -p ${@D}
	python $< --kernel vector_scalar_mul --out $@

build/vector_scalar_add.xclbin: build/vector_scalar_add.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --xclbin-kernel-name=VECTORSCALARADD --xclbin-kernel-id=0x901 \
		--xclbin-instance-name=VECTORSCALARADDINST --no-aiesim --aie-generate-npu --no-xchessc --no-xbridge \
		--aie-generate-cdo --no-compile-host --xclbin-name=${@F} --npu-insts-name=insts.txt $(<:%=../%)

build/multi_kernel.xclbin: build/vector_scalar_mul.mlir build/vector_scalar_add.xclbin
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --xclbin-kernel-name=VECTORSCALARMUL --xclbin-kernel-id=0x902 --no-xchessc --no-xbridge \
		--xclbin-instance-name=VECTORSCALARMULINST --no-aiesim --aie-generate-cdo --no-compile-host \
		--aie-generate-npu --xclbin-input=vector_scalar_add.xclbin --aie-generate-cdo \
		--xclbin-name=${@F} --npu-insts-name=insts.txt $(<:%=../%)

build/test: test.cpp
	clang++ -g $< -o $@ ${test_cflags} ${xrt_flags} -lrt -lstdc++ ${test_utils_flags}
