import argparse
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_


VECTOR_WIDTH = 50 * 1000 * 1024
TILE_WIDTH = 1024


def vector_scalar_add():
    @device(AIEDevice.npu1_1col)
    def device_body():
        # Datatype definition for FIFO buffers
        vector_ty = np.ndarray[(VECTOR_WIDTH,), np.dtype[np.int32]]
        tile_ty = np.ndarray[(TILE_WIDTH,), np.dtype[np.int32]]

        # Tile declarations
        shim_tile = tile(0, 0)
        compute_tile = tile(0, 2)

        # AIE-array data movement with object fifos
        of_in = object_fifo("in", shim_tile, compute_tile, 2, tile_ty)
        of_out = object_fifo("out", compute_tile, shim_tile, 2, tile_ty)

        # Setup optimized C++ kernel function
        vector_scalar_add_fn = external_func(
            "vector_scalar_add", inputs=[tile_ty, tile_ty]
        )

        # Compute tile implementation
        @core(compute_tile, "kernels.cc.o")
        def core_body():
            for _ in range_(sys.maxsize):
                for _ in range_(VECTOR_WIDTH // TILE_WIDTH):
                    elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    vector_scalar_add_fn(elem_in, elem_out)
                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(vector_ty, vector_ty)
        def sequence(a, b):
            in_task = shim_dma_single_bd_task(
                of_in, a, sizes=[1, 1, 1, VECTOR_WIDTH], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out, b, sizes=[1, 1, 1, VECTOR_WIDTH], issue_token=True
            )

            dma_start_task(in_task, out_task)
            dma_await_task(in_task, out_task)


def vector_scalar_mul():
    @device(AIEDevice.npu1_1col)
    def device_body():
        # Datatype definition for FIFO buffers
        vector_ty = np.ndarray[(VECTOR_WIDTH,), np.dtype[np.int32]]
        tile_ty = np.ndarray[(TILE_WIDTH,), np.dtype[np.int32]]

        # Tile declarations
        shim_tile = tile(0, 0)
        compute_tile = tile(0, 2)

        # AIE-array data movement with object fifos
        of_in = object_fifo("in", shim_tile, compute_tile, 2, tile_ty)
        of_out = object_fifo("out", compute_tile, shim_tile, 2, tile_ty)

        # Setup optimized C++ kernel function
        vector_scalar_mul_fn = external_func(
            "vector_scalar_mul", inputs=[tile_ty, tile_ty]
        )

        # Compute tile implementation
        @core(compute_tile, "kernels.cc.o")
        def core_body():
            for _ in range_(sys.maxsize):
                for _ in range_(VECTOR_WIDTH // TILE_WIDTH):
                    elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    vector_scalar_mul_fn(elem_in, elem_out)
                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(vector_ty, vector_ty)
        def sequence(a, b):
            in_task = shim_dma_single_bd_task(
                of_in, a, sizes=[1, 1, 1, VECTOR_WIDTH], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out, b, sizes=[1, 1, 1, VECTOR_WIDTH], issue_token=True
            )

            dma_start_task(in_task, out_task)
            dma_await_task(in_task, out_task)

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--kernel", type=str, help="which kernel to emit")
    arg_parser.add_argument("--out", type=str, help="output file", default="aie.mlir")
    args = arg_parser.parse_args()

    with mlir_mod_ctx() as ctx:
        if args.kernel == "vector_scalar_add":
            vector_scalar_add()
        elif args.kernel == "vector_scalar_mul":
            vector_scalar_mul()
        else:
            print("Unsupported kernel type", sys.stderr)
            exit(1)

        res = ctx.module.operation.verify()
        if not res:
            print(res, sys.stderr)
            exit(1)
        with open(args.out, 'w') as f:
            f.write(str(ctx.module))


if __name__ == '__main__':
    main()
