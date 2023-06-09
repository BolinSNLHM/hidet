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

    @tune.space(2, 'block_m', [2016])
    @tune.space(2, 'block_n', [144, 192, 256, 384, 512, 592, 544, 576, 896])
    @tune.space(2, 'block_k', [96, 256, 384, 512, 560, 784])
    @tune.space(2, 'nthreads', [4, 8, 16, 32, 64])
    @tune.space(2, 'nthreads_packing', [1, 2, 4, 8, 16, 32, 64])
    @tune.space(1, 'block_m', [2016])
    @tune.space(1, 'block_n', [384, 512, 896])
    @tune.space(1, 'block_k', [384, 512, 560])
    @tune.space(1, 'nthreads', [8, 16])
    def schedule_matmulf32_x86(
            self, block_m=2016, block_n=896, block_k=512, micro_ker=(6, 16),
            nthreads=16, nthreads_packing=2
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


        packing_parallel_attr = 'p' + str(nthreads_packing)
        if nthreads_packing == 1:
            packing_parallel_attr = None
        parallel_attr = 'p' + str(nthreads)
        if nthreads == 1:
            parallel_attr = None

        with hidet.script_module() as module:
            @hidet.script
            def matmul_kernel_x86(a: float32[m_size, k_size], b: float32[k_size, n_size], c: float32[m_size, n_size]):
                packed_b_ptr = cast(avx_malloc(k_size * n_size * 32, 32), ~float32)

                a_ptr = cast(a, ~float32)
                b_ptr = cast(b, ~float32)
                c_ptr = cast(c, ~float32)

                # packing
                for ax0_ax1_fused_ax2_fused in grid(48, attrs=packing_parallel_attr):
                    for ax4, ax6 in grid(64, 12):
                        b_idx = ax4 * 9216 + ax6 * 768 + ax0_ax1_fused_ax2_fused * 16

                        v0 = avx_f32x8_load(b_ptr + b_idx)
                        v8 = avx_f32x8_load(b_ptr + (b_idx + 8))

                        avx_f32x8_store_aligned(packed_b_ptr +
                                                (ax0_ax1_fused_ax2_fused * 12288 + ax4 * 192 + ax6 * 16), v0)
                        avx_f32x8_store_aligned(packed_b_ptr +
                                                (ax0_ax1_fused_ax2_fused * 12288 + ax4 * 192 + ax6 * 16 + 8), v8)

                # Main computation
                for i_outer_outer_j_outer_outer_fused_i_outer_inner_fused in grid(512, attrs=parallel_attr):
                    for j_outer_inner in range(3):
                        v0 = avx_f32x8_setzero()
                        v08 = avx_f32x8_setzero()
                        v1 = avx_f32x8_setzero()
                        v18 = avx_f32x8_setzero()
                        v2 = avx_f32x8_setzero()
                        v28 = avx_f32x8_setzero()
                        v3 = avx_f32x8_setzero()
                        v38 = avx_f32x8_setzero()

                        for k_outer in range(64):
                            cse_var_13 = i_outer_outer_j_outer_outer_fused_i_outer_inner_fused % 32 * 3072 + k_outer * 12
                            cse_var_12 = i_outer_outer_j_outer_outer_fused_i_outer_inner_fused // 32 * 36864 + j_outer_inner * 12288 + k_outer * 192
                            cse_var_11 = cse_var_12 + 96
                            cse_var_10 = cse_var_12 + 80
                            cse_var_9 = cse_var_12 + 64
                            cse_var_8 = cse_var_12 + 48
                            cse_var_7 = cse_var_12 + 32
                            cse_var_6 = cse_var_12 + 176
                            cse_var_5 = cse_var_12 + 160
                            cse_var_4 = cse_var_12 + 16
                            cse_var_3 = cse_var_12 + 144
                            cse_var_2 = cse_var_12 + 128
                            cse_var_1 = cse_var_12 + 112

                            bb0 = avx_f32x8_load(packed_b_ptr + cse_var_12)
                            bb8 = avx_f32x8_load(packed_b_ptr + (cse_var_12 + 8))
                            v0 = avx_f32x8_fmadd(v0, avx_f32x8_broadcast(a_ptr + cse_var_13),
                                                 bb0)
                            v08 = avx_f32x8_fmadd(v08, avx_f32x8_broadcast(a_ptr + cse_var_13),
                                                    bb8)
                            v1 = avx_f32x8_fmadd(v1, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 768)),
                                                    bb0)
                            v18 = avx_f32x8_fmadd(v18, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 768)),
                                                    bb8)
                            v2 = avx_f32x8_fmadd(v2, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1536)),
                                                    bb0)
                            v28 = avx_f32x8_fmadd(v28, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1536)),
                                                    bb8)
                            v3 = avx_f32x8_fmadd(v3, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2304)),
                                                    bb0)
                            v38 = avx_f32x8_fmadd(v38, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2304)),
                                                    bb8)

                            bb0 = avx_f32x8_load(packed_b_ptr + cse_var_4)
                            bb8 = avx_f32x8_load(packed_b_ptr + (cse_var_4 + 8))
                            v0 = avx_f32x8_fmadd(v0, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1)),
                                                    bb0)
                            v08 = avx_f32x8_fmadd(v08, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1)),
                                                    bb8)
                            v1 = avx_f32x8_fmadd(v1, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 769)),
                                                    bb0)
                            v18 = avx_f32x8_fmadd(v18, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 769)),
                                                    bb8)
                            v2 = avx_f32x8_fmadd(v2, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1537)),
                                                    bb0)
                            v28 = avx_f32x8_fmadd(v28, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1537)),
                                                    bb8)
                            v3 = avx_f32x8_fmadd(v3, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2305)),
                                                    bb0)
                            v38 = avx_f32x8_fmadd(v38, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2305)),
                                                    bb8)

                            bb0 = avx_f32x8_load(packed_b_ptr + cse_var_7)
                            bb8 = avx_f32x8_load(packed_b_ptr + (cse_var_7 + 8))
                            v0 = avx_f32x8_fmadd(v0, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2)),
                                                    bb0)
                            v08 = avx_f32x8_fmadd(v08, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2)),
                                                    bb8)
                            v1 = avx_f32x8_fmadd(v1, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 770)),
                                                    bb0)
                            v18 = avx_f32x8_fmadd(v18, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 770)),
                                                    bb8)
                            v2 = avx_f32x8_fmadd(v2, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1538)),
                                                    bb0)
                            v28 = avx_f32x8_fmadd(v28, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1538)),
                                                    bb8)
                            v3 = avx_f32x8_fmadd(v3, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2306)),
                                                    bb0)
                            v38 = avx_f32x8_fmadd(v38, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2306)),
                                                    bb8)

                            bb0 = avx_f32x8_load(packed_b_ptr + cse_var_8)
                            bb8 = avx_f32x8_load(packed_b_ptr + (cse_var_8 + 8))
                            v0 = avx_f32x8_fmadd(v0, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 3)),
                                                    bb0)
                            v08 = avx_f32x8_fmadd(v08, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 3)),
                                                    bb8)
                            v1 = avx_f32x8_fmadd(v1, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 771)),
                                                    bb0)
                            v18 = avx_f32x8_fmadd(v18, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 771)),
                                                    bb8)
                            v2 = avx_f32x8_fmadd(v2, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1539)),
                                                    bb0)
                            v28 = avx_f32x8_fmadd(v28, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1539)),
                                                    bb8)
                            v3 = avx_f32x8_fmadd(v3, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2307)),
                                                    bb0)
                            v38 = avx_f32x8_fmadd(v38, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2307)),
                                                    bb8)

                            bb0 = avx_f32x8_load(packed_b_ptr + cse_var_9)
                            bb8 = avx_f32x8_load(packed_b_ptr + (cse_var_9 + 8))
                            v0 = avx_f32x8_fmadd(v0, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 4)),
                                                    bb0)
                            v08 = avx_f32x8_fmadd(v08, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 4)),
                                                    bb8)
                            v1 = avx_f32x8_fmadd(v1, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 772)),
                                                    bb0)
                            v18 = avx_f32x8_fmadd(v18, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 772)),
                                                    bb8)
                            v2 = avx_f32x8_fmadd(v2, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1540)),
                                                    bb0)
                            v28 = avx_f32x8_fmadd(v28, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1540)),
                                                    bb8)
                            v3 = avx_f32x8_fmadd(v3, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2308)),
                                                    bb0)
                            v38 = avx_f32x8_fmadd(v38, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2308)),
                                                    bb8)

                            bb0 = avx_f32x8_load(packed_b_ptr + cse_var_10)
                            bb8 = avx_f32x8_load(packed_b_ptr + cse_var_10 + 8)
                            v0 = avx_f32x8_fmadd(v0, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 5)),
                                                    bb0)
                            v08 = avx_f32x8_fmadd(v08, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 5)),
                                                    bb8)
                            v1 = avx_f32x8_fmadd(v1, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 773)),
                                                    bb0)
                            v18 = avx_f32x8_fmadd(v18, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 773)),
                                                    bb8)
                            v2 = avx_f32x8_fmadd(v2, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1541)),
                                                    bb0)
                            v28 = avx_f32x8_fmadd(v28, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1541)),
                                                    bb8)
                            v3 = avx_f32x8_fmadd(v3, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2309)),
                                                    bb0)
                            v38 = avx_f32x8_fmadd(v38, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2309)),
                                                    bb8)

                            bb0 = avx_f32x8_load(packed_b_ptr + cse_var_11)
                            bb8 = avx_f32x8_load(packed_b_ptr + (cse_var_11 + 8))
                            v0 = avx_f32x8_fmadd(v0, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 6)),
                                                    bb0)
                            v08 = avx_f32x8_fmadd(v08, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 6)),
                                                    bb8)
                            v1 = avx_f32x8_fmadd(v1, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 774)),
                                                    bb0)
                            v18 = avx_f32x8_fmadd(v18, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 774)),
                                                    bb8)
                            v2 = avx_f32x8_fmadd(v2, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1542)),
                                                    bb0)
                            v28 = avx_f32x8_fmadd(v28, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1542)),
                                                    bb8)
                            v3 = avx_f32x8_fmadd(v3, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2310)),
                                                    bb0)
                            v38 = avx_f32x8_fmadd(v38, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2310)),
                                                    bb8)

                            bb0 = avx_f32x8_load(packed_b_ptr + cse_var_1)
                            bb8 = avx_f32x8_load(packed_b_ptr + (cse_var_1 + 8))
                            v0 = avx_f32x8_fmadd(v0, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 7)),
                                                    bb0)
                            v08 = avx_f32x8_fmadd(v08, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 7)),
                                                    bb8)
                            v1 = avx_f32x8_fmadd(v1, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 775)),
                                                    bb0)
                            v18 = avx_f32x8_fmadd(v18, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 775)),
                                                    bb8)
                            v2 = avx_f32x8_fmadd(v2, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1543)),
                                                    bb0)
                            v28 = avx_f32x8_fmadd(v28, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1543)),
                                                    bb8)
                            v3 = avx_f32x8_fmadd(v3, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2311)),
                                                    bb0)
                            v38 = avx_f32x8_fmadd(v38, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2311)),
                                                    bb8)

                            bb0 = avx_f32x8_load(packed_b_ptr + cse_var_2)
                            bb8 = avx_f32x8_load(packed_b_ptr + (cse_var_2 + 8))
                            v0 = avx_f32x8_fmadd(v0, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 8)),
                                                    bb0)
                            v08 = avx_f32x8_fmadd(v08, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 8)),
                                                    bb8)
                            v1 = avx_f32x8_fmadd(v1, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 776)),
                                                    bb0)
                            v18 = avx_f32x8_fmadd(v18, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 776)),
                                                    bb8)
                            v2 = avx_f32x8_fmadd(v2, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1544)),
                                                    bb0)
                            v28 = avx_f32x8_fmadd(v28, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1544)),
                                                    bb8)
                            v3 = avx_f32x8_fmadd(v3, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2312)),
                                                    bb0)
                            v38 = avx_f32x8_fmadd(v38, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2312)),
                                                    bb8)

                            bb0 = avx_f32x8_load(packed_b_ptr + cse_var_3)
                            bb8 = avx_f32x8_load(packed_b_ptr + (cse_var_3 + 8))
                            v0 = avx_f32x8_fmadd(v0, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 9)),
                                                    bb0)
                            v08 = avx_f32x8_fmadd(v08, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 9)),
                                                    bb8)
                            v1 = avx_f32x8_fmadd(v1, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 777)),
                                                    bb0)
                            v18 = avx_f32x8_fmadd(v18, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 777)),
                                                    bb8)
                            v2 = avx_f32x8_fmadd(v2, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1545)),
                                                    bb0)
                            v28 = avx_f32x8_fmadd(v28, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1545)),
                                                    bb8)
                            v3 = avx_f32x8_fmadd(v3, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2313)),
                                                    bb0)
                            v38 = avx_f32x8_fmadd(v38, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2313)),
                                                    bb8)

                            bb0 = avx_f32x8_load(packed_b_ptr + cse_var_5)
                            bb8 = avx_f32x8_load(packed_b_ptr + (cse_var_5 + 8))
                            v0 = avx_f32x8_fmadd(v0, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 10)),
                                                    bb0)
                            v08 = avx_f32x8_fmadd(v08, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 10)),
                                                    bb8)
                            v1 = avx_f32x8_fmadd(v1, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 778)),
                                                    bb0)
                            v18 = avx_f32x8_fmadd(v18, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 778)),
                                                    bb8)
                            v2 = avx_f32x8_fmadd(v2, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1546)),
                                                    bb0)
                            v28 = avx_f32x8_fmadd(v28, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1546)),
                                                    bb8)
                            v3 = avx_f32x8_fmadd(v3, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2314)),
                                                    bb0)
                            v38 = avx_f32x8_fmadd(v38, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2314)),
                                                    bb8)

                            bb0 = avx_f32x8_load(packed_b_ptr + cse_var_6)
                            bb8 = avx_f32x8_load(packed_b_ptr + (cse_var_6 + 8))
                            v0 = avx_f32x8_fmadd(v0, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 11)),
                                                    bb0)
                            v08 = avx_f32x8_fmadd(v08, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 11)),
                                                    bb8)
                            v1 = avx_f32x8_fmadd(v1, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 779)),
                                                    bb0)
                            v18 = avx_f32x8_fmadd(v18, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 779)),
                                                    bb8)
                            v2 = avx_f32x8_fmadd(v2, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1547)),
                                                    bb0)
                            v28 = avx_f32x8_fmadd(v28, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 1547)),
                                                    bb8)
                            v3 = avx_f32x8_fmadd(v3, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2315)),
                                                    bb0)
                            v38 = avx_f32x8_fmadd(v38, avx_f32x8_broadcast(a_ptr + (cse_var_13 + 2315)),
                                                    bb8)

                        avx_f32x8_store(c_ptr + (i_outer_outer_j_outer_outer_fused_i_outer_inner_fused % 32 * 3072 + 0 * 768 + i_outer_outer_j_outer_outer_fused_i_outer_inner_fused // 32 * 48 + j_outer_inner * 16), v0)
                        avx_f32x8_store(c_ptr + (i_outer_outer_j_outer_outer_fused_i_outer_inner_fused % 32 * 3072 + 0 * 768 + i_outer_outer_j_outer_outer_fused_i_outer_inner_fused // 32 * 48 + j_outer_inner * 16 + 8), v08)
                        avx_f32x8_store(c_ptr + (i_outer_outer_j_outer_outer_fused_i_outer_inner_fused % 32 * 3072 + 1 * 768 + i_outer_outer_j_outer_outer_fused_i_outer_inner_fused // 32 * 48 + j_outer_inner * 16), v1)
                        avx_f32x8_store(c_ptr + (i_outer_outer_j_outer_outer_fused_i_outer_inner_fused % 32 * 3072 + 1 * 768 + i_outer_outer_j_outer_outer_fused_i_outer_inner_fused // 32 * 48 + j_outer_inner * 16 + 8), v18)
                        avx_f32x8_store(c_ptr + (i_outer_outer_j_outer_outer_fused_i_outer_inner_fused % 32 * 3072 + 2 * 768 + i_outer_outer_j_outer_outer_fused_i_outer_inner_fused // 32 * 48 + j_outer_inner * 16), v2)
                        avx_f32x8_store(c_ptr + (i_outer_outer_j_outer_outer_fused_i_outer_inner_fused % 32 * 3072 + 2 * 768 + i_outer_outer_j_outer_outer_fused_i_outer_inner_fused // 32 * 48 + j_outer_inner * 16 + 8), v28)
                        avx_f32x8_store(c_ptr + (i_outer_outer_j_outer_outer_fused_i_outer_inner_fused % 32 * 3072 + 3 * 768 + i_outer_outer_j_outer_outer_fused_i_outer_inner_fused // 32 * 48 + j_outer_inner * 16), v3)
                        avx_f32x8_store(c_ptr + (i_outer_outer_j_outer_outer_fused_i_outer_inner_fused % 32 * 3072 + 3 * 768 + i_outer_outer_j_outer_outer_fused_i_outer_inner_fused // 32 * 48 + j_outer_inner * 16 + 8), v38)


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
