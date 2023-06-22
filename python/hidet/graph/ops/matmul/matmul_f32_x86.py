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
    @tune.space(2, 'block_n', [144, 192, 256, 384, 512, 592, 544, 576, 896, 1024])
    @tune.space(2, 'block_k', [96, 256, 384, 512, 560, 672, 784, 544])
    @tune.space(2, 'nthreads', [4, 8, 16, 32, 64])
    @tune.space(2, 'nthreads_packing', [1, 2, 4, 8, 16])
    @tune.space(1, 'block_m', [2016])
    @tune.space(1, 'block_n', [384, 512, 896])
    @tune.space(1, 'block_k', [384, 512, 560])
    @tune.space(1, 'nthreads', [8, 16])
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

        packed_a_type = tensor_type('float32', layout=row_layout(block_m // tile_m, 1) * col_layout(tile_m, block_k))
        packed_b_type = tensor_type('float32', layout=row_layout(1, block_n // tile_n) * row_layout(block_k, tile_n))

        aip_outer_rows = block_m // tile_m
        bip_outer_cols = block_n // tile_n

        attr_packing = 'p' + str(nthreads_packing)
        if nthreads_packing == 1:
            attr_packing = None

        # Shrink the block size of m if there's no more than 2016 rows...
        # block_m = min(block_m, m_size)

        with hidet.script_module() as module:
            # Add a micro_kernel_4x16 just as the 6x16 one below
            @hidet.script
            def micro_kernel_4x16(
                    a: packed_a_type, b: packed_b_type, c_ptr: ~float32, pb: int32, msize: int32, nsize: int32,
                    is_first: bool
            ):
                c = as_tensor_pointer(c_ptr, dtype=float32, shape=[msize, nsize])
                c0 = avx_f32x8_load(~c[0, 0])
                c08 = avx_f32x8_load(~c[0, 8])
                c1 = avx_f32x8_load(~c[1, 0])
                c18 = avx_f32x8_load(~c[1, 8])
                c2 = avx_f32x8_load(~c[2, 0])
                c28 = avx_f32x8_load(~c[2, 8])
                c3 = avx_f32x8_load(~c[3, 0])
                c38 = avx_f32x8_load(~c[3, 8])

                if is_first:
                    c0 = avx_f32x8_setzero()
                    c08 = avx_f32x8_setzero()
                    c1 = avx_f32x8_setzero()
                    c18 = avx_f32x8_setzero()
                    c2 = avx_f32x8_setzero()
                    c28 = avx_f32x8_setzero()
                    c3 = avx_f32x8_setzero()
                    c38 = avx_f32x8_setzero()

                a_ptr = cast(a, ~float32)
                b_ptr = cast(b, ~float32)

                for _ in range(pb):
                    bb0to7 = avx_f32x8_load_aligned(b_ptr)
                    bb8to15 = avx_f32x8_load_aligned(b_ptr + 8)
                    b_ptr = b_ptr + 16

                    aa = avx_f32x8_broadcast(a_ptr)
                    c0 = avx_f32x8_fmadd(aa, bb0to7, c0)
                    c08 = avx_f32x8_fmadd(aa, bb8to15, c08)

                    aa = avx_f32x8_broadcast(a_ptr + 1)
                    c1 = avx_f32x8_fmadd(aa, bb0to7, c1)
                    c18 = avx_f32x8_fmadd(aa, bb8to15, c18)

                    aa = avx_f32x8_broadcast(a_ptr + 2)
                    c2 = avx_f32x8_fmadd(aa, bb0to7, c2)
                    c28 = avx_f32x8_fmadd(aa, bb8to15, c28)

                    aa = avx_f32x8_broadcast(a_ptr + 3)
                    c3 = avx_f32x8_fmadd(aa, bb0to7, c3)
                    c38 = avx_f32x8_fmadd(aa, bb8to15, c38)

                    a_ptr = a_ptr + 4
                avx_f32x8_store(~c[0, 0], c0)
                avx_f32x8_store(~c[0, 8], c08)
                avx_f32x8_store(~c[1, 0], c1)
                avx_f32x8_store(~c[1, 8], c18)
                avx_f32x8_store(~c[2, 0], c2)
                avx_f32x8_store(~c[2, 8], c28)
                avx_f32x8_store(~c[3, 0], c3)
                avx_f32x8_store(~c[3, 8], c38)


            @hidet.script
            def micro_kernel_6x16(
                    a: packed_a_type, b: packed_b_type, c_ptr: ~float32, pb: int32, msize: int32, nsize: int32,
                    is_first: bool
            ):
                c = as_tensor_pointer(c_ptr, dtype=float32, shape=[msize, nsize])
                c0 = avx_f32x8_load(~c[0, 0])
                c08 = avx_f32x8_load(~c[0, 8])
                c1 = avx_f32x8_load(~c[1, 0])
                c18 = avx_f32x8_load(~c[1, 8])
                c2 = avx_f32x8_load(~c[2, 0])
                c28 = avx_f32x8_load(~c[2, 8])
                c3 = avx_f32x8_load(~c[3, 0])
                c38 = avx_f32x8_load(~c[3, 8])
                c4 = avx_f32x8_load(~c[4, 0])
                c48 = avx_f32x8_load(~c[4, 8])
                c5 = avx_f32x8_load(~c[5, 0])
                c58 = avx_f32x8_load(~c[5, 8])

                if is_first:
                    c0 = avx_f32x8_setzero()
                    c08 = avx_f32x8_setzero()
                    c1 = avx_f32x8_setzero()
                    c18 = avx_f32x8_setzero()
                    c2 = avx_f32x8_setzero()
                    c28 = avx_f32x8_setzero()
                    c3 = avx_f32x8_setzero()
                    c38 = avx_f32x8_setzero()
                    c4 = avx_f32x8_setzero()
                    c48 = avx_f32x8_setzero()
                    c5 = avx_f32x8_setzero()
                    c58 = avx_f32x8_setzero()

                a_ptr = cast(a, ~float32)
                b_ptr = cast(b, ~float32)

                for _ in range(pb):
                    bb0to7 = avx_f32x8_load_aligned(b_ptr)
                    bb8to15 = avx_f32x8_load_aligned(b_ptr + 8)
                    b_ptr = b_ptr + 16

                    aa = avx_f32x8_broadcast(a_ptr)
                    c0 = avx_f32x8_fmadd(aa, bb0to7, c0)
                    c08 = avx_f32x8_fmadd(aa, bb8to15, c08)

                    aa = avx_f32x8_broadcast(a_ptr + 1)
                    c1 = avx_f32x8_fmadd(aa, bb0to7, c1)
                    c18 = avx_f32x8_fmadd(aa, bb8to15, c18)

                    aa = avx_f32x8_broadcast(a_ptr + 2)
                    c2 = avx_f32x8_fmadd(aa, bb0to7, c2)
                    c28 = avx_f32x8_fmadd(aa, bb8to15, c28)

                    aa = avx_f32x8_broadcast(a_ptr + 3)
                    c3 = avx_f32x8_fmadd(aa, bb0to7, c3)
                    c38 = avx_f32x8_fmadd(aa, bb8to15, c38)

                    aa = avx_f32x8_broadcast(a_ptr + 4)
                    c4 = avx_f32x8_fmadd(aa, bb0to7, c4)
                    c48 = avx_f32x8_fmadd(aa, bb8to15, c48)

                    aa = avx_f32x8_broadcast(a_ptr + 5)
                    c5 = avx_f32x8_fmadd(aa, bb0to7, c5)
                    c58 = avx_f32x8_fmadd(aa, bb8to15, c58)

                    a_ptr = a_ptr + 6
                avx_f32x8_store(~c[0, 0], c0)
                avx_f32x8_store(~c[0, 8], c08)
                avx_f32x8_store(~c[1, 0], c1)
                avx_f32x8_store(~c[1, 8], c18)
                avx_f32x8_store(~c[2, 0], c2)
                avx_f32x8_store(~c[2, 8], c28)
                avx_f32x8_store(~c[3, 0], c3)
                avx_f32x8_store(~c[3, 8], c38)
                avx_f32x8_store(~c[4, 0], c4)
                avx_f32x8_store(~c[4, 8], c48)
                avx_f32x8_store(~c[5, 0], c5)
                avx_f32x8_store(~c[5, 8], c58)

            micro_kernel = micro_kernel_6x16
            # if m_size % 6 != 0 and m_size % 4 == 0:
            #     micro_kernel = micro_kernel_4x16
            #     tile_m = 4
            # After all these, adjust block_m to be a multiple of tile_m
            # block_m = (block_m + tile_m - 1) // tile_m * tile_m

            @hidet.script
            def macro_kernel(
                    a: packed_a_type, b: packed_b_type, c_in_macro: float32[m_size, n_size], ib: int32, jb: int32,
                    pb: int32, is_first: bool
            ):
                mpanels = (ib + tile_m - 1) // tile_m
                npanels = (jb + tile_n - 1) // tile_n
                _mr = ib % tile_m
                _nr = jb % tile_n

                # # Loop 2
                # fuse the two loops mpanel, npanel into one loop
                para = 'p' + str(nthreads)
                for mnpanel in grid(mpanels * npanels, attrs=para):
                    mpanel = mnpanel // npanels
                    npanel = mnpanel % npanels
                    mr = tile_m if mpanel != mpanels - 1 or _mr == 0 else _mr
                    nr = tile_n if npanel != npanels - 1 or _nr == 0 else _nr
                    ii = mpanel * tile_m
                    jj = npanel * tile_n
                    # micro-kernel
                    if mr == tile_m and nr == tile_n:
                        micro_kernel(~a[ii, 0], ~b[0, jj], ~c_in_macro[ii, jj], pb, m_size, n_size, is_first)
                    else:
                        temp_c = tensor(
                            scope=DeclareScope.Default, dtype='float32', layout=row_layout(tile_m, tile_n)
                        )
                        micro_kernel(~a[ii, 0], ~b[0, jj], temp_c, pb, tile_m, tile_n, True)
                        if is_first:
                            for remain_row, remain_col in grid(mr, nr):
                                c_in_macro[ii + remain_row, jj + remain_col] = temp_c[remain_row, remain_col]
                        else:
                            for remain_row, remain_col in grid(mr, nr):
                                c_in_macro[ii + remain_row, jj + remain_col] += temp_c[remain_row, remain_col]

            @hidet.script
            def matmul_kernel_x86(a: float32[m_size, k_size], b: float32[k_size, n_size], c: float32[m_size, n_size]):
                mbs = (m_size + block_m - 1) // block_m
                nbs = (n_size + block_n - 1) // block_n
                kbs = (k_size + block_k - 1) // block_k

                packed_a_alloc = avx_malloc(block_m * block_k * 32, 64)
                packed_b_alloc = avx_malloc(block_k * block_n * 32, 64)

                packed_a = as_tensor_pointer(
                    packed_a_alloc, float32, layout=row_layout(aip_outer_rows, 1) * col_layout(tile_m, block_k)
                )
                packed_b = as_tensor_pointer(
                    packed_b_alloc, float32, layout=row_layout(1, bip_outer_cols) * row_layout(block_k, tile_n)
                )

                for mb in range(mbs):
                    i = mb * block_m
                    ib = min(block_m, m_size - i)
                    for kb in range(kbs):
                        p = kb * block_k
                        pb = min(block_k, k_size - p)

                        mp = ib // tile_m
                        mr = ib % tile_m

                        for micropanel_idx in grid(mp, attrs=attr_packing):
                            panel_row_start = micropanel_idx * tile_m
                            m8 = pb // 8
                            m8r = pb % 8
                            for packing_col_idx in grid(m8):
                                pack_col_start = packing_col_idx * 8
                                v0 = avx_f32x8_load(~a[i + panel_row_start, p + pack_col_start])
                                v1 = avx_f32x8_load(~a[i + panel_row_start + 1, p + pack_col_start])
                                v2 = avx_f32x8_load(~a[i + panel_row_start + 2, p + pack_col_start])
                                v3 = avx_f32x8_load(~a[i + panel_row_start + 3, p + pack_col_start])
                                v4 = avx_f32x8_load(~a[i + panel_row_start + 4, p + pack_col_start])
                                v5 = avx_f32x8_load(~a[i + panel_row_start + 5, p + pack_col_start])

                                unpack0 = avx_f32x8_unpacklo(v0, v1)
                                unpack1 = avx_f32x8_unpackhi(v0, v1)
                                unpack2 = avx_f32x8_unpacklo(v2, v3)
                                unpack3 = avx_f32x8_unpackhi(v2, v3)
                                unpack4 = avx_f32x8_unpacklo(v4, v5)
                                unpack5 = avx_f32x8_unpackhi(v4, v5)

                                shf0 = avx_f32x8_shuffle(unpack0, unpack2, 0x44)
                                shf1 = avx_f32x8_shuffle(unpack4, unpack0, 0xE4)
                                shf2 = avx_f32x8_shuffle(unpack2, unpack4, 0xEE)
                                shf3 = avx_f32x8_shuffle(unpack5, unpack1, 0xE4)
                                shf4 = avx_f32x8_shuffle(unpack3, unpack5, 0xEE)
                                shf5 = avx_f32x8_shuffle(unpack1, unpack3, 0x44)

                                low_shf1 = avx_f32x8_cast_f32x4(shf1)
                                res0 = avx_f32x8_insert_f32x4(shf0, low_shf1, 0x1)
                                res1 = avx_f32x8_permute2f32x4(shf0, shf1, 0x31)

                                low_shf5 = avx_f32x8_cast_f32x4(shf5)
                                res2 = avx_f32x8_insert_f32x4(shf2, low_shf5, 0x1)
                                res3 = avx_f32x8_permute2f32x4(shf2, shf5, 0x31)

                                low_shf4 = avx_f32x8_cast_f32x4(shf4)
                                res4 = avx_f32x8_insert_f32x4(shf3, low_shf4, 0x1)
                                res5 = avx_f32x8_permute2f32x4(shf3, shf4, 0x31)

                                avx_f32x8_store_aligned(~packed_a[panel_row_start, pack_col_start], res0)
                                avx_f32x8_store_aligned(~packed_a[panel_row_start + 2, pack_col_start + 1], res2)
                                avx_f32x8_store_aligned(~packed_a[panel_row_start + 4, pack_col_start + 2], res4)
                                avx_f32x8_store_aligned(~packed_a[panel_row_start, pack_col_start + 4], res1)
                                avx_f32x8_store_aligned(~packed_a[panel_row_start + 2, pack_col_start + 5], res3)
                                avx_f32x8_store_aligned(~packed_a[panel_row_start + 4, pack_col_start + 6], res5)
                            if m8r > 0:
                                remaining_start_col = m8 * 8
                                for remain_off in range(m8r):
                                    curr_remain_col = remaining_start_col + remain_off
                                    for micropanel_row in range(tile_m):
                                        packed_a[panel_row_start + micropanel_row, curr_remain_col] = a[
                                            i + micropanel_row + panel_row_start, p + curr_remain_col
                                        ]
                        if mr > 0:
                            remain_start_row = mp * tile_m
                            for remain_col in grid(pb):
                                for remain_row in range(mr):
                                    packed_a[remain_start_row + remain_row, remain_col] = a[
                                        i + remain_start_row + remain_row, p + remain_col
                                    ]
                                remain_row = mr
                                while remain_row < tile_m:
                                    packed_a[remain_start_row + remain_row, remain_col] = 0.0
                                    remain_row += 1

                        for nb in range(nbs):
                            j = nb * block_n
                            jb = min(block_n, n_size - j)
                            np = jb // tile_n
                            nr = jb % tile_n

                            for idx in grid(np * pb, attrs=attr_packing):
                                micropanel_idx = idx // pb
                                micropanel_row = idx % pb
                                panel_col_start = micropanel_idx * tile_n
                                b0 = avx_f32x8_load(~b[p + micropanel_row, j + panel_col_start])
                                b8 = avx_f32x8_load(~b[p + micropanel_row, j + panel_col_start + 8])

                                avx_f32x8_store_aligned(~packed_b[micropanel_row, panel_col_start], b0)
                                avx_f32x8_store_aligned(~packed_b[micropanel_row, panel_col_start + 8], b8)

                            if nr > 0:
                                remain_col_start = np * tile_n
                                for remain_row in grid(pb):
                                    for remain_col in range(nr):
                                        packed_b[remain_row, remain_col + remain_col_start] = b[
                                            p + remain_row, j + remain_col + remain_col_start
                                        ]
                                    remain_col = nr
                                    while remain_col < tile_n:
                                        packed_b[remain_row, remain_col_start + remain_col] = 0.0
                                        remain_col += 1
                            macro_kernel(packed_a, packed_b, ~c[i, j], ib, jb, pb, kb == 0)
                avx_free(packed_a_alloc)
                avx_free(packed_b_alloc)

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