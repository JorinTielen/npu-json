import argparse
import sys

import numpy as np

from aie.iron import *
from aie.iron.controlflow import range_

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx


def aie_design(
    kernel_obj: str,
    block_size: int,
    blocks_per_chunk: int,
    num_cols: int,
    aie_device: AIEDevice,
):
    num_rows = 2

    data_block_size = block_size
    data_chunk_size = blocks_per_chunk * data_block_size

    index_chunk_size = data_chunk_size // 8
    index_block_size = data_block_size // 8

    carry_block_size = 64
    carry_chunk_size = blocks_per_chunk * carry_block_size

    # Combined kernel input per block: 64-byte aligned carry section + raw data
    input_block_size = carry_block_size + data_block_size
    input_chunk_size = blocks_per_chunk * input_block_size

    blocks_per_row = data_chunk_size // data_block_size // num_cols // num_rows

    @device(aie_device)
    def device_body():
        input_chunk_ty = np.ndarray[
            (input_chunk_size,), np.dtype[np.uint8]
        ]
        input_block_ty = np.ndarray[
            (input_block_size,), np.dtype[np.uint8]
        ]
        input_split_ty = np.ndarray[
            (input_block_size * num_rows,), np.dtype[np.uint8],
        ]
        index_chunk_ty = np.ndarray[(index_chunk_size,), np.dtype[np.uint8]]
        index_block_ty = np.ndarray[(index_block_size,), np.dtype[np.uint8]]
        index_split_ty = np.ndarray[
            (index_block_size * num_rows,), np.dtype[np.uint8],
        ]

        tiles = [
            [tile(col, row) for col in range(0, num_cols)]
            for row in range(0, num_rows + 2)
        ]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        combined_kernel = external_func(
            "combined_index",
            inputs=[input_block_ty, index_block_ty, index_block_ty, np.int32],
            link_with=kernel_obj,
        )

        shim_fifos_in = [None] * num_cols
        core_fifos_in = [[None for _ in range(0, num_cols)] for _ in range(0, num_rows)]

        shim_fifos_out_string = [None] * num_cols
        shim_fifos_out_structural = [None] * num_cols
        core_fifos_out_string = [
            [None for _ in range(0, num_cols)] for _ in range(0, num_rows)
        ]
        core_fifos_out_structural = [
            [None for _ in range(0, num_cols)] for _ in range(0, num_rows)
        ]

        # Combined kernel input: carry flags + raw JSON data
        for col in range(0, num_cols):
            shim_fifos_in[col] = object_fifo(
                f"in_c{col}_mem",
                shim_tiles[col],
                mem_tiles[col],
                2,
                input_split_ty,
            )
            for row in range(0, num_rows):
                core_fifos_in[row][col] = object_fifo(
                    f"in_c{col}_r{row}",
                    mem_tiles[col],
                    core_tiles[row][col],
                    2,
                    input_block_ty,
                )
            object_fifo_link(
                shim_fifos_in[col],
                [core_fifos_in[row][col] for row in range(0, num_rows)],
                [],
                [i * input_block_size for i in range(0, num_rows)],
            )

        # String index output
        for col in range(0, num_cols):
            shim_fifos_out_string[col] = object_fifo(
                f"string_out_c{col}_mem",
                mem_tiles[col],
                shim_tiles[col],
                2,
                index_split_ty,
            )
            for row in range(0, num_rows):
                core_fifos_out_string[row][col] = object_fifo(
                    f"string_out_c{col}_r{row}",
                    core_tiles[row][col],
                    mem_tiles[col],
                    2,
                    index_block_ty,
                )
            object_fifo_link(
                [core_fifos_out_string[row][col] for row in range(0, num_rows)],
                shim_fifos_out_string[col],
                [i * index_block_size for i in range(0, num_rows)],
                [],
            )

        # Structural index output
        for col in range(0, num_cols):
            shim_fifos_out_structural[col] = object_fifo(
                f"structural_out_c{col}_mem",
                mem_tiles[col],
                shim_tiles[col],
                2,
                index_split_ty,
            )
            for row in range(0, num_rows):
                core_fifos_out_structural[row][col] = object_fifo(
                    f"structural_out_c{col}_r{row}",
                    core_tiles[row][col],
                    mem_tiles[col],
                    2,
                    index_block_ty,
                )
            object_fifo_link(
                [core_fifos_out_structural[row][col] for row in range(0, num_rows)],
                shim_fifos_out_structural[col],
                [i * index_block_size for i in range(0, num_rows)],
                [],
            )

        # Validation
        assert (data_chunk_size / data_block_size / num_cols / num_rows).is_integer(), (
            "Data sizes do not evenly divide for splitting across tiles"
        )
        assert (
            index_chunk_size / index_block_size / num_cols / num_rows
        ).is_integer(), "Index sizes do not evenly divide for splitting across tiles"

        # Core definitions: all tiles run the combined kernel
        for col in range(0, num_cols):
            for row in range(0, num_rows):

                @core(core_tiles[row][col])
                def core_body():
                    for _ in range_(0, sys.maxsize):
                        for _ in range_(
                            data_chunk_size
                            // data_block_size
                            // num_cols
                            // num_rows
                        ):
                            of_in = core_fifos_in[row][col]
                            of_out_str = core_fifos_out_string[row][col]
                            of_out_st = core_fifos_out_structural[row][col]

                            elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                            elem_out_str = of_out_str.acquire(ObjectFifoPort.Produce, 1)
                            elem_out_st = of_out_st.acquire(ObjectFifoPort.Produce, 1)
                            combined_kernel(elem_in, elem_out_str, elem_out_st, data_block_size)
                            of_in.release(ObjectFifoPort.Consume, 1)
                            of_out_str.release(ObjectFifoPort.Produce, 1)
                            of_out_st.release(ObjectFifoPort.Produce, 1)

        # Host side data-flow movement (3 DMA operations per column)
        @runtime_sequence(
            input_chunk_ty, index_chunk_ty, index_chunk_ty
        )
        def sequence(
            input_buffer,
            string_index_buffer,
            structural_index_buffer,
        ):
            for col in range(0, num_cols):
                npu_dma_memcpy_nd(
                    metadata=shim_fifos_in[col],
                    bd_id=0,
                    mem=input_buffer,
                    sizes=[1, 1, 1, input_chunk_size // num_cols],
                    offsets=[0, 0, 0, (input_chunk_size // num_cols) * col],
                    issue_token=True,
                )
                npu_dma_memcpy_nd(
                    metadata=shim_fifos_out_string[col],
                    bd_id=1,
                    mem=string_index_buffer,
                    sizes=[1, 1, 1, index_chunk_size // num_cols],
                    offsets=[0, 0, 0, (index_chunk_size // num_cols) * col],
                    issue_token=True,
                )
                npu_dma_memcpy_nd(
                    metadata=shim_fifos_out_structural[col],
                    bd_id=2,
                    mem=structural_index_buffer,
                    sizes=[1, 1, 1, index_chunk_size // num_cols],
                    offsets=[0, 0, 0, (index_chunk_size // num_cols) * col],
                    issue_token=True,
                )
            dma_wait(*shim_fifos_out_string)
            dma_wait(*shim_fifos_out_structural)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--kernel-obj", type=str, help="kernel object file")
    arg_parser.add_argument("--out", type=str, help="output file", default="aie.mlir")
    arg_parser.add_argument(
        "--block-size", type=int, default=16 * 1024, help="input block size in bytes"
    )
    arg_parser.add_argument(
        "--blocks-per-chunk", type=int, default=512, help="blocks per chunk"
    )
    arg_parser.add_argument(
        "--num-cols", type=int, default=8, help="number of NPU columns"
    )
    arg_parser.add_argument(
        "--aie-device",
        type=str,
        default="npu2",
        choices=["npu1", "npu2"],
        help="AIE device",
    )
    args = arg_parser.parse_args()

    aie_device_map = {
        "npu1": AIEDevice.npu1,
        "npu2": AIEDevice.npu2,
    }

    aie_device = aie_device_map[args.aie_device]

    with mlir_mod_ctx() as ctx:
        aie_design(
            args.kernel_obj,
            args.block_size,
            args.blocks_per_chunk,
            args.num_cols,
            aie_device,
        )
        res = ctx.module.operation.verify()
        if not res:
            print(res, sys.stderr)
            exit(1)
        with open(args.out, "w") as f:
            f.write(str(ctx.module))


if __name__ == "__main__":
    main()