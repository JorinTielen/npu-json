import argparse
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_


# The kernels supported by this NPU design.
KERNELS = [
    "string_index",
    "structural_character_index"
]


# Must be kept in sync with the CHUNK_SIZE and BLOCK_SIZE in `src/npu-json/engine.hpp`.
# The 4 extra bytes are for the carry index
BLOCKS_PER_CHUNK = 4 * 1000
DATA_BLOCK_SIZE = 1024
DATA_CHUNK_SIZE = BLOCKS_PER_CHUNK * DATA_BLOCK_SIZE

INDEX_CHUNK_SIZE = DATA_CHUNK_SIZE // 8
INDEX_BLOCK_SIZE = DATA_BLOCK_SIZE // 8

CARRY_CHUNK_SIZE = DATA_CHUNK_SIZE // DATA_BLOCK_SIZE * 4
CARRY_BLOCK_SIZE = 4

# We assume 1/4th of chars are structural at 32-bits per structural character for the NPU fast-path for now.
# TODO: Add overflow bool at end
STRUCTURAL_CHARACTER_CHUNK_SIZE = INDEX_CHUNK_SIZE
STRUCTURAL_CHARACTER_BLOCK_SIZE = INDEX_BLOCK_SIZE

# AI Engine structural design function
def string_index_design(kernel_obj: str):
    num_cols = 2
    num_rows = 4

    # Device declaration - aie2 device NPU
    @device(AIEDevice.npu1_2col)
    def device_body():
        data_chunk_ty = np.ndarray[(DATA_CHUNK_SIZE + CARRY_CHUNK_SIZE,), np.dtype[np.uint8]]
        data_block_ty = np.ndarray[(DATA_BLOCK_SIZE + CARRY_BLOCK_SIZE,), np.dtype[np.uint8]]
        data_split_ty = np.ndarray[((DATA_BLOCK_SIZE + CARRY_BLOCK_SIZE) * num_rows,), np.dtype[np.uint8]]
        index_chunk_ty = np.ndarray[(INDEX_CHUNK_SIZE,), np.dtype[np.uint8]]
        index_block_ty = np.ndarray[(INDEX_BLOCK_SIZE,), np.dtype[np.uint8]]
        index_split_ty = np.ndarray[(INDEX_BLOCK_SIZE * num_rows,), np.dtype[np.uint8]]

        tiles = [
            [tile(col, row) for col in range(0, num_cols)] for row in range(0, num_rows + 2)
        ]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        kernel = external_func(
            "string_index", inputs=[data_block_ty, index_block_ty, np.int32]
        )

        # Setup FIFOs for input data
        shim_fifos_in = [None] * num_cols
        core_fifos_in = [
            [None for _ in range(0, num_cols)] for _ in range(0, num_rows)
        ]
        for col in range(0, num_cols):
            shim_fifos_in[col] = object_fifo(
                f"in_c{col}_mem",
                shim_tiles[col],
                mem_tiles[col],
                2,
                data_split_ty
            )
            for row in range(0, num_rows):
                core_fifos_in[row][col] = object_fifo(
                    f"in_c{col}_r{row}",
                    mem_tiles[col],
                    core_tiles[row][col],
                    2,
                    data_block_ty
                )
            object_fifo_link(
                shim_fifos_in[col],
                [core_fifos_in[row][col] for row in range(0, num_rows)],
                [],
                [i * (DATA_BLOCK_SIZE + CARRY_BLOCK_SIZE) for i in range(0, num_rows)],
            )

        # Setup FIFOs for output index
        shim_fifos_out = [None] * num_cols
        core_fifos_out = [
            [None for _ in range(0, num_cols)] for _ in range(0, num_rows)
        ]
        for col in range(0, num_cols):
            shim_fifos_out[col] = object_fifo(
                f"out_c{col}_mem",
                mem_tiles[col],
                shim_tiles[col],
                2,
                index_split_ty
            )
            for row in range(0, num_rows):
                core_fifos_out[row][col] = object_fifo(
                    f"out_c{col}_r{row}",
                    core_tiles[row][col],
                    mem_tiles[col],
                    2,
                    index_block_ty
                )
            object_fifo_link(
                [core_fifos_out[row][col] for row in range(0, num_rows)],
                shim_fifos_out[col],
                [i * INDEX_BLOCK_SIZE for i in range(0, num_rows)],
                []
            )

        # Compute tile core definitions for running the kernel
        assert (DATA_CHUNK_SIZE / DATA_BLOCK_SIZE / num_cols / num_rows).is_integer(), \
            "Data sizes do not evenly divide for splitting across tiles"
        for col in range(0, num_cols):
            for row in range(0, num_rows):
                @core(core_tiles[row][col], kernel_obj)
                def core_body():
                    for _ in range_(0, sys.maxsize):
                        for _ in range_(DATA_CHUNK_SIZE // DATA_BLOCK_SIZE // num_cols // num_rows):
                            of_in = core_fifos_in[row][col]
                            of_out = core_fifos_out[row][col]

                            elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                            elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                            kernel(elem_in, elem_out, DATA_BLOCK_SIZE)
                            of_in.release(ObjectFifoPort.Consume, 1)
                            of_out.release(ObjectFifoPort.Produce, 1)

        # Host side data-flow movement
        @runtime_sequence(data_chunk_ty, index_chunk_ty)
        def sequence(a, b):
            for col in range(0, num_cols):
                input_chunk_size = DATA_CHUNK_SIZE + CARRY_CHUNK_SIZE
                npu_dma_memcpy_nd(
                    metadata=shim_fifos_in[col],
                    bd_id=col * 2,
                    mem=a,
                    sizes=[1, 1, 1, input_chunk_size // num_cols],
                    offsets=[0, 0, 0, (input_chunk_size // num_cols) * col],
                    issue_token=True
                )
                npu_dma_memcpy_nd(
                    metadata=shim_fifos_out[col],
                    bd_id=col * 2 + 1,
                    mem=b,
                    sizes=[1, 1, 1, INDEX_CHUNK_SIZE // num_cols],
                    offsets=[0, 0, 0, (INDEX_CHUNK_SIZE // num_cols) * col],
                    issue_token=True
                )
            dma_wait(*shim_fifos_out)


# AI Engine structural design function
def structural_character_index_design(kernel_obj: str):
    num_cols = 2
    num_rows = 4

    # Device declaration - aie2 device NPU
    @device(AIEDevice.npu1_2col)
    def device_body():
        data_chunk_ty = np.ndarray[(DATA_CHUNK_SIZE + INDEX_CHUNK_SIZE,), np.dtype[np.uint8]]
        data_block_ty = np.ndarray[(DATA_BLOCK_SIZE + INDEX_BLOCK_SIZE,), np.dtype[np.uint8]]
        data_split_ty = np.ndarray[((DATA_BLOCK_SIZE + INDEX_BLOCK_SIZE) * num_rows,), np.dtype[np.uint8]]
        structural_character_chunk_ty = np.ndarray[(STRUCTURAL_CHARACTER_CHUNK_SIZE,), np.dtype[np.uint8]]
        structural_character_block_ty = np.ndarray[(STRUCTURAL_CHARACTER_BLOCK_SIZE,), np.dtype[np.uint8]]
        structural_character_split_ty = np.ndarray[(STRUCTURAL_CHARACTER_BLOCK_SIZE * num_rows,), np.dtype[np.uint8]]

        tiles = [
            [tile(col, row) for col in range(0, num_cols)] for row in range(0, num_rows + 2)
        ]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        kernel = external_func(
            "structural_character_index", inputs=[data_block_ty, structural_character_block_ty, np.int32]
        )

        # Setup FIFOs for input data
        shim_fifos_in = [None] * num_cols
        core_fifos_in = [
            [None for _ in range(0, num_cols)] for _ in range(0, num_rows)
        ]
        for col in range(0, num_cols):
            shim_fifos_in[col] = object_fifo(
                f"in_c{col}_mem",
                shim_tiles[col],
                mem_tiles[col],
                2,
                data_split_ty
            )
            for row in range(0, num_rows):
                core_fifos_in[row][col] = object_fifo(
                    f"in_c{col}_r{row}",
                    mem_tiles[col],
                    core_tiles[row][col],
                    2,
                    data_block_ty
                )
            object_fifo_link(
                shim_fifos_in[col],
                [core_fifos_in[row][col] for row in range(0, num_rows)],
                [],
                [i * (DATA_BLOCK_SIZE + INDEX_BLOCK_SIZE) for i in range(0, num_rows)],
            )

        # Setup FIFOs for output index
        shim_fifos_out = [None] * num_cols
        core_fifos_out = [
            [None for _ in range(0, num_cols)] for _ in range(0, num_rows)
        ]
        for col in range(0, num_cols):
            shim_fifos_out[col] = object_fifo(
                f"out_c{col}_mem",
                mem_tiles[col],
                shim_tiles[col],
                2,
                structural_character_split_ty
            )
            for row in range(0, num_rows):
                core_fifos_out[row][col] = object_fifo(
                    f"out_c{col}_r{row}",
                    core_tiles[row][col],
                    mem_tiles[col],
                    2,
                    structural_character_block_ty
                )
            object_fifo_link(
                [core_fifos_out[row][col] for row in range(0, num_rows)],
                shim_fifos_out[col],
                [i * (STRUCTURAL_CHARACTER_BLOCK_SIZE) for i in range(0, num_rows)],
                []
            )

        # Compute tile core definitions for running the kernel
        assert (DATA_CHUNK_SIZE / DATA_BLOCK_SIZE / num_cols / num_rows).is_integer(), \
            "Data sizes do not evenly divide for splitting across tiles"
        assert (STRUCTURAL_CHARACTER_CHUNK_SIZE / STRUCTURAL_CHARACTER_BLOCK_SIZE / num_cols / num_rows).is_integer(), \
            "Structural character index sizes do not evenly divide for splitting across tiles"
        for col in range(0, num_cols):
            for row in range(0, num_rows):
                @core(core_tiles[row][col], kernel_obj)
                def core_body():
                    for _ in range_(0, sys.maxsize):
                        for _ in range_(DATA_CHUNK_SIZE // DATA_BLOCK_SIZE // num_cols // num_rows):
                            of_in = core_fifos_in[row][col]
                            of_out = core_fifos_out[row][col]

                            elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                            elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                            kernel(elem_in, elem_out, DATA_BLOCK_SIZE)
                            of_in.release(ObjectFifoPort.Consume, 1)
                            of_out.release(ObjectFifoPort.Produce, 1)

        # Host side data-flow movement
        @runtime_sequence(data_chunk_ty, structural_character_chunk_ty)
        def sequence(a, b):
            for col in range(0, num_cols):
                input_chunk_size = DATA_CHUNK_SIZE + INDEX_CHUNK_SIZE
                npu_dma_memcpy_nd(
                    metadata=shim_fifos_in[col],
                    bd_id=col * 2,
                    mem=a,
                    sizes=[1, 1, 1, input_chunk_size // num_cols],
                    offsets=[0, 0, 0, (input_chunk_size // num_cols) * col],
                    # issue_token=True
                )
                npu_dma_memcpy_nd(
                    metadata=shim_fifos_out[col],
                    bd_id=col * 2 + 1,
                    mem=b,
                    sizes=[1, 1, 1, STRUCTURAL_CHARACTER_CHUNK_SIZE // num_cols],
                    offsets=[0, 0, 0, (STRUCTURAL_CHARACTER_CHUNK_SIZE // num_cols) * col],
                    # issue_token=True
                )
            dma_wait(*shim_fifos_out)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--kernel", type=str, help="kernel to build", required=True, choices=KERNELS)
    arg_parser.add_argument("--kernel-obj", type=str, help="kernel object file")
    arg_parser.add_argument("--out", type=str, help="output file", default="aie.mlir")
    args = arg_parser.parse_args()

    with mlir_mod_ctx() as ctx:
        match args.kernel:
            case "string_index":
                string_index_design(args.kernel_obj)
            case "structural_character_index":
                structural_character_index_design(args.kernel_obj)
            case _:
                print("Unsupported kernel", sys.stderr)
                exit(1)
        res = ctx.module.operation.verify()
        if not res:
            print(res, sys.stderr)
            exit(1)
        with open(args.out, 'w') as f:
            f.write(str(ctx.module))


if __name__ == '__main__':
    main()
