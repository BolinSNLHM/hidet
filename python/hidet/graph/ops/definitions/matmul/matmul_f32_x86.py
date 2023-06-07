# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Union
from hidet.ir.dtypes import float32, int32
from hidet.ir.expr import cast
from hidet.ir.func import IRModule
from hidet.ir.compute import TensorNode
from hidet.ir.primitives import avx_malloc, avx_free
from hidet.ir.primitives.cpu import x86_memset, avx_f32x8_setzero
from hidet.ir.stmt import DeclareScope
from hidet.ir.task import Task
from hidet.ir.compute import compute, reduce
from hidet.graph.ops.definitions.utils import input_like, broadcast_shape, can_mutually_broadcast
from hidet.graph.ops.definitions.utils import tune
from hidet.graph.operator import Operator, Tensor
from hidet.graph.ops.definitions.utils import broadcast_indices


class MatmulF32Taskx86(Task):
    def __init__(self, a: TensorNode, b: TensorNode):
        a_shape = a.const_shape
        b_shape = b.const_shape

        if not a.type.dtype == float32 or not b.type.dtype == float32:
            raise ValueError('Both inputs must be float32 tensors')

        if len(a_shape) < 2 or len(b_shape) < 2:
            raise ValueError('Matrix multiplication expect at least 2D tensor, got {} and {}'.format(a_shape, b_shape))
        if a_shape[-1] != b_shape[-2]:
            raise ValueError(
                'Matrix multiplication expect tensor A and B with shape [..., M, K] and [..., K, N]'
                ', got {} and {}'.format(a_shape, b_shape)
            )
        if not can_mutually_broadcast(a_shape[:-2], b_shape[:-2]):
            raise ValueError(
                'Matrix multiplication expect tensor A and B with compatible broadcast shape, '
                'got {} and {}'.format(a_shape, b_shape)
            )

        k_size = a_shape[-1]
        c_shape = broadcast_shape(a_shape[:-2], b_shape[:-2]) + [a_shape[-2], b_shape[-1]]

        c = compute(
            name='c',
            shape=c_shape,
            fcompute=lambda *indices: reduce(
                shape=[k_size],
                fcompute=lambda k: a[broadcast_indices(indices[:-2], a_shape[:-2], c_shape[1:-2]) + [indices[-2], k]]
                                   * b[broadcast_indices(indices[:-2], b_shape[:-2], c_shape[1:-2]) + [k, indices[-1]]],
                reduce_type='sum',
            ),
        )

        super().__init__(
            name='matmul_f32_x86',
            inputs=[a, b],
            outputs=[c],
            attributes={'m_size': a_shape[-2], 'n_size': b_shape[-1], 'k_size': a_shape[-1]},
        )

    def allow_epilogue(self) -> bool:
        return True

    def allow_prologue(self) -> bool:
        return False

    def implement_cpu(self, working_dir: str) -> Union[IRModule, List[IRModule]]:
        return tune.extract_ir_modules(self.schedule_matmulf32_x86)

    # @tune.space(2, 'block_m', [2016])
    # @tune.space(2, 'block_n', [96, 192, 384, 512, 576, 896])
    # @tune.space(2, 'block_k', [96, 256, 384, 512, 784])
    @tune.space(2, 'nthreads', [1, 2, 4, 8, 16, 32, 64])
    @tune.space(2, 'nthreads_packing', [1, 2, 4, 8, 16, 32, 64])
    # @tune.space(1, 'block_m', [2016])
    # @tune.space(1, 'block_n', [384, 512, 896])
    # @tune.space(1, 'block_k', [384, 512, 560])
    # @tune.space(1, 'nthreads', [8, 16])
    def schedule_matmulf32_x86(
            self, block_m=2016, block_n=896, block_k=512, micro_ker=(6, 16),
            nthreads=16, nthreads_packing=4
    ) -> IRModule:
        import hidet
        from hidet.ir.type import tensor_type
        from hidet.lang import tensor, grid, as_tensor_pointer
        from hidet.lang.layout import row_layout, col_layout
        from hidet.lang.cpu import avx_f32x8_store, avx_f32x8_fmadd, avx_f32x8_load, avx_f32x8_broadcast
        from hidet.lang.cpu import avx_f32x4_broadcast, avx_f32x4_fmadd, avx_f32x4_load, avx_f32x4_store
        from hidet.lang.cpu import avx_f32x8_store_aligned, avx_f32x8_load_aligned
        from hidet.lang.cpu import avx_f32x8_unpacklo, avx_f32x8_unpackhi
        from hidet.lang.cpu import avx_f32x8_insert_f32x4, avx_f32x8_permute2f32x4
        from hidet.lang.cpu import avx_f32x8_shuffle, avx_f32x8_cast_f32x4

        node_a, node_b = self.inputs[0], self.inputs[1]
        a_shape = node_a.const_shape
        b_shape = node_b.const_shape
        m_size, n_size, k_size = a_shape[-2], b_shape[-1], a_shape[-1]

        tile_m, tile_n = micro_ker

        supported_microkers = ((6, 16), (4, 8), (8, 8))
        tune.check(micro_ker in supported_microkers, "The size of the micro-kernel is not supported")

        tune.check(block_m % tile_m == block_n % tile_n == 0, 'Tile size must divide the corresponding block size')
        tune.check(nthreads > 0 and nthreads_packing > 0, 'Number of threads must be positive')

        # if block_m is too big: round it down to the nearest multiple of 6 greater than m_size
        block_m = min(block_m, (m_size // 6) * 6)
        # if block_n is too big: round it down to the nearest multiple of 16 greater than n_size
        block_n = min(block_n, (n_size // 16) * 16)
        # if block_k is too big: round it down to the nearest multiple of 16 greater than k_size
        block_k = min(block_k, (k_size // 16) * 16)

        if block_n == 0:
            block_n = 16
        if block_m == 0:
            block_m = 6
        if block_k == 0:
            block_k = 16

        packed_a_type = tensor_type('float32', layout=row_layout(block_m // tile_m, 1) * col_layout(tile_m, block_k))
        packed_b_type = tensor_type('float32', layout=row_layout(1, block_n // tile_n) * row_layout(block_k, tile_n))

        aip_outer_rows = block_m // tile_m
        bip_outer_cols = block_n // tile_n

        # if block_m is too big: round it down to the nearest multiple of 6 greater than m_size
        block_m = min(block_m, (m_size // 6) * 6)

        parallel_packing_attr = 'p' + str(nthreads_packing)
        if nthreads_packing == 1:
            parallel_packing_attr = None

        parallel_attr = 'p' + str(nthreads)
        if nthreads == 1:
            parallel_attr = None

        with hidet.script_module() as module:

            @hidet.script
            def matmul_kernel_x86(a: float32[m_size, k_size], b: float32[k_size, n_size], c: float32[m_size, n_size]):

                # allocate a global region to collaboratively pack b
                packed_b_alloc = avx_malloc(k_size * n_size * 4, 32)
                packed_b_ptr = cast(packed_b_alloc, ~float32)
                packed_b = as_tensor_pointer(
                    packed_b_ptr, dtype=float32,
                    layout=row_layout(1, n_size // tile_n) * row_layout(k_size, tile_n),
                )
                width8_panels = k_size // 8  # TODO: this '8' should be replaced by
                w8_panel_size = k_size * 8
                width8_remainder = k_size % 8
                assert width8_remainder == 0  # TODO: handle this case later.
                for panel_idx in grid(width8_panels, attrs=parallel_packing_attr):
                    panel_start_ptr = packed_b_ptr + (panel_idx * w8_panel_size)
                    for panel_row in range(k_size):
                        v0 = avx_f32x8_load(~b[panel_row, panel_idx * 8])
                        avx_f32x8_store_aligned(panel_start_ptr + panel_row * 8, v0)

                ntasks = k_size // 8
                task_nouter_iterations = k_size // 8

                a_ptr = cast(a, ~float32)
                b_ptr = cast(b, ~float32)
                c_ptr = cast(c, ~float32)

                k_outer_iters = k_size // 4
                for task_idx in grid(ntasks, attrs=parallel_attr):
                    for i_outer_inner in range(task_nouter_iterations):
                        v0 = avx_f32x8_setzero()
                        v1 = avx_f32x8_setzero()
                        v2 = avx_f32x8_setzero()
                        v3 = avx_f32x8_setzero()
                        v4 = avx_f32x8_setzero()
                        v5 = avx_f32x8_setzero()
                        v6 = avx_f32x8_setzero()
                        v7 = avx_f32x8_setzero()

                        for k_outer in range(k_outer_iters):
                            cse_var_5 = task_idx * w8_panel_size + k_outer * 32  # TODO: the meaning of this "32"?
                            cse_var_4 = i_outer_inner * w8_panel_size + k_outer * 4  # TODO: the meaning of this "4"?
                            cse_var_3 = cse_var_5 + 8  # TODO: The meaning of this "8"?
                            cse_var_2 = cse_var_5 + 24  # TODO: The meaning of this "24"?
                            cse_var_1 = cse_var_5 + 16  # TODO: The meaning of this "16"?

                            bb = avx_f32x8_load_aligned(packed_b_ptr + cse_var_5)
                            v0 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4), bb, v0)
                            v1 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 128), bb, v1)
                            v2 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 256), bb, v2)
                            v3 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 384), bb, v3)
                            v4 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 512), bb, v4)
                            v5 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 640), bb, v5)
                            v6 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 768), bb, v6)
                            v7 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 896), bb, v7)

                            bb = avx_f32x8_load_aligned(packed_b_ptr + cse_var_3)
                            v0 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 1), bb, v0)
                            v1 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 129), bb, v1)
                            v2 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 257), bb, v2)
                            v3 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 385), bb, v3)
                            v4 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 513), bb, v4)
                            v5 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 641), bb, v5)
                            v6 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 769), bb, v6)
                            v7 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 897), bb, v7)

                            bb = avx_f32x8_load_aligned(packed_b_ptr + cse_var_1)
                            v0 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 2), bb, v0)
                            v1 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 130), bb, v1)
                            v2 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 258), bb, v2)
                            v3 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 386), bb, v3)
                            v4 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 514), bb, v4)
                            v5 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 642), bb, v5)
                            v6 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 770), bb, v6)
                            v7 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 898), bb, v7)

                            bb = avx_f32x8_load_aligned(packed_b_ptr + cse_var_2)
                            v0 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 3), bb, v0)
                            v1 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 131), bb, v1)
                            v2 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 259), bb, v2)
                            v3 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 387), bb, v3)
                            v4 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 515), bb, v4)
                            v5 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 643), bb, v5)
                            v6 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 771), bb, v6)
                            v7 = avx_f32x8_fmadd(avx_f32x8_broadcast(a_ptr + cse_var_4 + 899), bb, v7)
                        avx_f32x8_store(c_ptr + (i_outer_inner * 1024 + 0 * 128 + task_idx * 8), v0)
                        avx_f32x8_store(c_ptr + (i_outer_inner * 1024 + 1 * 128 + task_idx * 8), v1)
                        avx_f32x8_store(c_ptr + (i_outer_inner * 1024 + 2 * 128 + task_idx * 8), v2)
                        avx_f32x8_store(c_ptr + (i_outer_inner * 1024 + 3 * 128 + task_idx * 8), v3)
                        avx_f32x8_store(c_ptr + (i_outer_inner * 1024 + 4 * 128 + task_idx * 8), v4)
                        avx_f32x8_store(c_ptr + (i_outer_inner * 1024 + 5 * 128 + task_idx * 8), v5)
                        avx_f32x8_store(c_ptr + (i_outer_inner * 1024 + 6 * 128 + task_idx * 8), v6)
                        avx_f32x8_store(c_ptr + (i_outer_inner * 1024 + 7 * 128 + task_idx * 8), v7)

        assert isinstance(matmul_kernel_x86, hidet.ir.Function)
        matmul_kernel_x86.kind = "cpu_kernel"
        ir_module = module.ir_module()
        return ir_module


class Matmulx86Op(Operator):
    def __init__(self, a: Tensor, b: Tensor):
        if not (len(a.shape) == len(b.shape) == 2 and a.shape[1] == b.shape[0]):
            raise ValueError('Matrix multiplication: incompatible sizes: {} and {}'.format(a.shape, b.shape))
        task = MatmulF32Taskx86(input_like(a, 'a'), input_like(b, 'b'))
        super().__init__(inputs=[a, b], attributes={}, task=task)


def matmul_x86(a: Tensor, b: Tensor) -> Tensor:
    return Matmulx86Op(a, b).get_output(0)
