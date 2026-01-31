import argparse
import sys

import numpy as np

from aie.iron import *
from aie.iron.controlflow import range_

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
# from aie.helpers.dialects.ext.scf import _for as range_


# Must be kept in sync with the CHUNK_SIZE and BLOCK_SIZE in `src/npu-json/engine.hpp`.
# The 4 extra bytes are for the carry index
BLOCKS_PER_CHUNK = 16 * 1000
DATA_BLOCK_SIZE = 1024
DATA_CHUNK_SIZE = BLOCKS_PER_CHUNK * DATA_BLOCK_SIZE

INDEX_CHUNK_SIZE = DATA_CHUNK_SIZE // 8
INDEX_BLOCK_SIZE = DATA_BLOCK_SIZE // 8

CARRY_CHUNK_SIZE = DATA_CHUNK_SIZE // DATA_BLOCK_SIZE * 4
CARRY_BLOCK_SIZE = 4

# AI Engine structural design function
def aie_design(kernel_obj: str):
    num_cols = 8
    num_rows = 4

    # Device declaration - aie2 device NPU
    @device(AIEDevice.npu2)
    def device_body():
        data_chunk_ty = np.ndarray[(DATA_CHUNK_SIZE,), np.dtype[np.uint8]]
        data_block_ty = np.ndarray[(DATA_BLOCK_SIZE,), np.dtype[np.uint8]]
        data_split_ty = np.ndarray[((DATA_BLOCK_SIZE) * (num_rows // 2),), np.dtype[np.uint8]]
        string_chunk_ty = np.ndarray[(INDEX_CHUNK_SIZE * 2 + CARRY_CHUNK_SIZE,), np.dtype[np.uint8]]
        string_block_ty = np.ndarray[(INDEX_BLOCK_SIZE * 2 + CARRY_BLOCK_SIZE,), np.dtype[np.uint8]]
        string_split_ty = np.ndarray[((INDEX_BLOCK_SIZE * 2 + CARRY_BLOCK_SIZE) * (num_rows // 2),), np.dtype[np.uint8]]
        index_chunk_ty = np.ndarray[(INDEX_CHUNK_SIZE,), np.dtype[np.uint8]]
        index_block_ty = np.ndarray[(INDEX_BLOCK_SIZE,), np.dtype[np.uint8]]
        index_split_ty = np.ndarray[(INDEX_BLOCK_SIZE * (num_rows // 2),), np.dtype[np.uint8]]

        tiles = [
            [tile(col, row) for col in range(0, num_cols)] for row in range(0, num_rows + 2)
        ]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        string_kernel = external_func(
            "string_index", inputs=[string_block_ty, index_block_ty, np.int32]
        )

        structural_kernel = external_func(
            "structural_character_index", inputs=[data_block_ty, index_block_ty, np.int32]
        )

        shim_fifos_in_string = [None] * num_cols
        shim_fifos_in_structural = [None] * num_cols
        core_fifos_in = [
            [None for _ in range(0, num_cols)] for _ in range(0, num_rows)
        ]

        shim_fifos_out_string = [None] * num_cols
        shim_fifos_out_structural = [None] * num_cols
        core_fifos_out = [
            [None for _ in range(0, num_cols)] for _ in range(0, num_rows)
        ]

        # Setup FIFOs for string kernel input data (first half of rows)
        for col in range(0, num_cols):
            shim_fifos_in_string[col] = object_fifo(
                f"string_in_c{col}_mem",
                shim_tiles[col],
                mem_tiles[col],
                2,
                string_split_ty
            )
            for row in range(0, num_rows // 2):
                core_fifos_in[row][col] = object_fifo(
                    f"string_in_c{col}_r{row}",
                    mem_tiles[col],
                    core_tiles[row][col],
                    2,
                    string_block_ty
                )
            object_fifo_link(
                shim_fifos_in_string[col],
                [core_fifos_in[row][col] for row in range(0, num_rows // 2)],
                [],
                [i * (INDEX_BLOCK_SIZE * 2 + CARRY_BLOCK_SIZE) for i in range(0, num_rows // 2)],
            )

        # Setup FIFOs for structural kernel input data (second half of rows)
        for col in range(0, num_cols):
            shim_fifos_in_structural[col] = object_fifo(
                f"structural_in_c{col}_mem",
                shim_tiles[col],
                mem_tiles[col],
                2,
                data_split_ty
            )
            for row in range(num_rows // 2, num_rows):
                core_fifos_in[row][col] = object_fifo(
                    f"structural_in_c{col}_r{row}",
                    mem_tiles[col],
                    core_tiles[row][col],
                    2,
                    data_block_ty
                )
            object_fifo_link(
                shim_fifos_in_structural[col],
                [core_fifos_in[row][col] for row in range(num_rows // 2, num_rows)],
                [],
                [i * DATA_BLOCK_SIZE for i in range(0, num_rows // 2)],
            )

        # Setup FIFOs for string kernel output (first half of rows)
        for col in range(0, num_cols):
            shim_fifos_out_string[col] = object_fifo(
                f"string_out_c{col}_mem",
                mem_tiles[col],
                shim_tiles[col],
                2,
                index_split_ty
            )
            for row in range(0, num_rows // 2):
                core_fifos_out[row][col] = object_fifo(
                    f"string_out_c{col}_r{row}",
                    core_tiles[row][col],
                    mem_tiles[col],
                    2,
                    index_block_ty
                )
            object_fifo_link(
                [core_fifos_out[row][col] for row in range(0, num_rows // 2)],
                shim_fifos_out_string[col],
                [i * INDEX_BLOCK_SIZE for i in range(0, num_rows // 2)],
                []
            )

        # Setup FIFOs for structural kernel output (second half of rows)
        for col in range(0, num_cols):
            shim_fifos_out_structural[col] = object_fifo(
                f"structural_out_c{col}_mem",
                mem_tiles[col],
                shim_tiles[col],
                2,
                index_split_ty
            )
            for row in range(num_rows // 2, num_rows):
                core_fifos_out[row][col] = object_fifo(
                    f"structural_out_c{col}_r{row}",
                    core_tiles[row][col],
                    mem_tiles[col],
                    2,
                    index_block_ty
                )
            object_fifo_link(
                [core_fifos_out[row][col] for row in range(num_rows // 2, num_rows)],
                shim_fifos_out_structural[col],
                [i * INDEX_BLOCK_SIZE for i in range(0, num_rows // 2)],
                []
            )

        # Basic validation of data sizes to avoid hard-crashing the NPU.
        assert (DATA_CHUNK_SIZE / DATA_BLOCK_SIZE / num_cols / num_rows).is_integer(), \
            "Data sizes do not evenly divide for splitting across tiles"
        assert (INDEX_CHUNK_SIZE / INDEX_BLOCK_SIZE / num_cols / num_rows).is_integer(), \
            "Index sizes do not evenly divide for splitting across tiles"

        # Compute tile core definitions for running the string kernel (first half of rows)
        for col in range(0, num_cols):
            for row in range(0, num_rows // 2):
                @core(core_tiles[row][col], kernel_obj)
                def core_body():
                    for _ in range_(0, sys.maxsize):
                        for _ in range_(DATA_CHUNK_SIZE // DATA_BLOCK_SIZE // num_cols // (num_rows // 2)):
                            of_in = core_fifos_in[row][col]
                            of_out = core_fifos_out[row][col]

                            elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                            elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                            string_kernel(elem_in, elem_out, DATA_BLOCK_SIZE)
                            of_in.release(ObjectFifoPort.Consume, 1)
                            of_out.release(ObjectFifoPort.Produce, 1)

        # Compute tile core definitions for running the structuralkernel (second half of rows)
        for col in range(0, num_cols):
            for row in range(num_rows // 2, num_rows):
                @core(core_tiles[row][col], kernel_obj)
                def core_body():
                    for _ in range_(0, sys.maxsize):
                        for _ in range_(DATA_CHUNK_SIZE // DATA_BLOCK_SIZE // num_cols // (num_rows // 2)):
                            of_in = core_fifos_in[row][col]
                            of_out = core_fifos_out[row][col]

                            elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                            elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                            structural_kernel(elem_in, elem_out, DATA_BLOCK_SIZE)
                            of_in.release(ObjectFifoPort.Consume, 1)
                            of_out.release(ObjectFifoPort.Produce, 1)

        # Host side data-flow movement
        @runtime_sequence(data_chunk_ty, string_chunk_ty, index_chunk_ty, index_chunk_ty)
        def sequence(data_buffer, string_input_buffer, string_index_buffer, structural_index_buffer):
            for col in range(0, num_cols):
                string_input_chunk_size = INDEX_CHUNK_SIZE * 2 + CARRY_CHUNK_SIZE
                npu_dma_memcpy_nd(
                    metadata=shim_fifos_in_string[col],
                    # bd_id=col * 4,
                    bd_id=0,
                    mem=string_input_buffer,
                    sizes=[1, 1, 1, string_input_chunk_size // num_cols],
                    offsets=[0, 0, 0, (string_input_chunk_size // num_cols) * col],
                    issue_token=True
                )
                npu_dma_memcpy_nd(
                    metadata=shim_fifos_out_string[col],
                    # bd_id=col * 4 + 1,
                    bd_id=1,
                    mem=string_index_buffer,
                    sizes=[1, 1, 1, INDEX_CHUNK_SIZE // num_cols],
                    offsets=[0, 0, 0, (INDEX_CHUNK_SIZE // num_cols) * col],
                    issue_token=True
                )
                npu_dma_memcpy_nd(
                    metadata=shim_fifos_in_structural[col],
                    # bd_id=col * 4 + 2,
                    bd_id=2,
                    mem=data_buffer,
                    sizes=[1, 1, 1, DATA_CHUNK_SIZE // num_cols],
                    offsets=[0, 0, 0, (DATA_CHUNK_SIZE // num_cols) * col],
                    issue_token=True
                )
                npu_dma_memcpy_nd(
                    metadata=shim_fifos_out_structural[col],
                    # bd_id=col * 4 + 3,
                    bd_id=3,
                    mem=structural_index_buffer,
                    sizes=[1, 1, 1, INDEX_CHUNK_SIZE // num_cols],
                    offsets=[0, 0, 0, (INDEX_CHUNK_SIZE // num_cols) * col],
                    issue_token=True
                )
            dma_wait(*shim_fifos_out_string)
            dma_wait(*shim_fifos_out_structural)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--kernel-obj", type=str, help="kernel object file")
    arg_parser.add_argument("--out", type=str, help="output file", default="aie.mlir")
    args = arg_parser.parse_args()

    with mlir_mod_ctx() as ctx:
        aie_design(args.kernel_obj)
        res = ctx.module.operation.verify()
        if not res:
            print(res, sys.stderr)
            exit(1)
        with open(args.out, 'w') as f:
            f.write(str(ctx.module))


if __name__ == '__main__':
    main()
