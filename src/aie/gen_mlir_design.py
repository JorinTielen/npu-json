import argparse
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_


DATA_CHUNK_SIZE = 10 * 1024 * 1024
DATA_BLOCK_SIZE = 1024

INDEX_CHUNK_SIZE = DATA_CHUNK_SIZE // 8
INDEX_BLOCK_SIZE = DATA_BLOCK_SIZE // 8

# AI Engine structural design function
def mlir_aie_design(kernel_obj: str):
    num_cols = 1
    num_rows = 4

    # Device declaration - aie2 device NPU
    @device(AIEDevice.npu1_1col)
    def device_body():
        data_chunk_ty = np.ndarray[(DATA_CHUNK_SIZE,), np.dtype[np.uint8]]
        data_block_ty = np.ndarray[(DATA_BLOCK_SIZE,), np.dtype[np.uint8]]
        data_split_ty = np.ndarray[(DATA_BLOCK_SIZE * num_rows,), np.dtype[np.uint8]]
        index_chunk_ty = np.ndarray[(INDEX_CHUNK_SIZE,), np.dtype[np.uint8]]
        index_block_ty = np.ndarray[(INDEX_BLOCK_SIZE,), np.dtype[np.uint8]]
        index_split_ty = np.ndarray[(INDEX_BLOCK_SIZE * num_rows,), np.dtype[np.uint8]]

        tiles = [
            [tile(col, row) for col in range(0, num_cols)] for row in range(0, num_rows + 2)
        ]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        kernel_name = "stringindexer"

        kernel = external_func(
            kernel_name, inputs=[data_block_ty, index_block_ty, np.int32]
        )

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
                [i * DATA_BLOCK_SIZE for i in range(0, num_rows)],
            )

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

        @runtime_sequence(data_chunk_ty, index_chunk_ty)
        def sequence(a, b):
            for col in range(0, num_cols):
                npu_dma_memcpy_nd(
                    metadata=shim_fifos_in[col],
                    bd_id=col * 2,
                    mem=a,
                    sizes=[1, 1, 1, DATA_CHUNK_SIZE // num_cols],
                    offsets=[0, 0, 0, (DATA_CHUNK_SIZE // num_cols) * col],
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


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--kernel-obj", type=str, help="kernel object file")
    arg_parser.add_argument("--out", type=str, help="output file", default="aie.mlir")
    args = arg_parser.parse_args()

    with mlir_mod_ctx() as ctx:
        mlir_aie_design(args.kernel_obj)
        res = ctx.module.operation.verify()
        if not res:
            print(res, sys.stderr)
            exit(1)
        with open(args.out, 'w') as f:
            f.write(str(ctx.module))


if __name__ == '__main__':
    main()
