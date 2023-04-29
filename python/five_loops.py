import numpy.testing
from hidet.ir import Constant
from hidet.ir.stmt import DeclareScope

import hidet

def matmul_kernel5():
    from hidet.transforms.generate_packed_func import add_packed_func
    from hidet.lang import attr
    from hidet.lang import float32, int32
    from hidet.lang import as_tensor_pointer, tensor
    from hidet.lang.mapping import repeat, spatial, auto_map
    from hidet.lang.layout import row_layout, local_layout, col_layout

    from hidet.lang.avx import avx_f32x4_broadcast, avx_f32x4_fmadd, avx_f32x4_load, avx_f32x4_store


    with hidet.lang.script_module() as script_module:

        @hidet.lang.script
        def matmul_kernel(
                a_ptr: ~float32,
                b_ptr: ~float32,
                c_ptr: ~float32,
                m_size: int32,
                n_size: int32,
                k_size: int32
        ):
            a = as_tensor_pointer(a_ptr, float32, [m_size, k_size])
            b = as_tensor_pointer(b_ptr, float32, [k_size, n_size])
            c = as_tensor_pointer(c_ptr, float32, [m_size, n_size])

            MC: int32 = 264
            NC: int32 = 2016
            KC: int32 = 48

            MR: int32 = 4
            NR: int32 = 4

            j = 0
            while j < n_size:
                jb = min(NC, n_size - j)
                # Loop 4
                b_col = j
                c_col = j

                p = 0
                while p < k_size:
                    pb = min(KC, k_size - p)
                    # Loop 3
                    a_col = p
                    b_row = p
                    i = 0
                    while i < m_size:
                        ib = min(MC, m_size - i)
                        a_row = i
                        c_row = i
                        # Loop 2
                        jj = 0
                        while jj < jb:
                            jb2 = min(NR, jb - jj)

                            # Loop 1
                            ii = 0
                            while ii < ib:
                                ib2 = min(MR, ib - ii)

                                # micro-kernel
                                c0_0123 = avx_f32x4_load(~c[c_row, c_col])

                                c1_0123 = avx_f32x4_load(~c[c_row+1, c_col])

                                c2_0123 = avx_f32x4_load(~c[c_row+2, c_col])

                                c3_0123 = avx_f32x4_load(~c[c_row+3, c_col])

                                aa_col = a_col
                                bb_row = b_row
                                for pp in range(pb):
                                    bb_0123 = avx_f32x4_load(~b[bb_row, b_col])
                                    aa = avx_f32x4_broadcast(~a[a_row, aa_col])
                                    c0_0123 = avx_f32x4_fmadd(aa, bb_0123, c0_0123)
                                    aa = avx_f32x4_broadcast(~a[a_row+1, aa_col])
                                    c1_0123 = avx_f32x4_fmadd(aa, bb_0123, c1_0123)
                                    aa = avx_f32x4_broadcast(~a[a_row+2, aa_col])
                                    c2_0123 = avx_f32x4_fmadd(aa, bb_0123, c2_0123)
                                    aa = avx_f32x4_broadcast(~a[a_row+3, aa_col])
                                    c3_0123 = avx_f32x4_fmadd(aa, bb_0123, c3_0123)
                                    aa_col += 1
                                    bb_row += 1
                                avx_f32x4_store(~c[a_row, jj], c0_0123)
                                avx_f32x4_store(~c[a_row+1, jj], c1_0123)
                                avx_f32x4_store(~c[a_row+2, jj], c2_0123)
                                avx_f32x4_store(~c[a_row+3, jj], c3_0123)
                                a_row += 1
                                c_row += 1
                                ii += MR
                            b_col += 1
                            c_col += 1
                            jj += NR
                        i += MC

                    p += KC

                j += NC

#################################################3
    assert isinstance(matmul_kernel, hidet.ir.Function)
    matmul_kernel.kind = 'host_kernel'

    ir_module = script_module.ir_module()
    add_packed_func(ir_module, matmul_kernel, pack_func_name='matmul6')
    compiled_function = hidet.driver.build_ir_module(ir_module)
    return compiled_function

def ff():
    func = matmul_kernel5()

    for m, n, k in [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (1024, 512, 768), (333, 444, 555),
                    (480, 480, 480), (720, 720, 720), (720, 960, 1440)]:
        a = hidet.randn([m, k], dtype='float32').cpu()
        b = hidet.randn([k, n], dtype='float32').cpu()
        c = hidet.zeros([m, n]).cpu()
        func(a, b, c, m, n, k)
        numpy.testing.assert_allclose(
            actual=c.cpu().numpy(),
            desired=a.cpu().numpy() @ b.cpu().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

        hidet_latency = hidet.utils.benchmark_func(
            lambda: func(a, b, c, m, n, k), repeat=2
        )

        np_latency = hidet.utils.benchmark_func(
            lambda: a.cpu().numpy() @ b.cpu().numpy()
        )

        print(f'{m} x {k} x {n}: hidet takes {hidet_latency:.2f} ms')
        print(f'{m} x {k} x {n}: numpy takes {np_latency: .2f} ms')



ff()

# 256 x 256 x 256: hidet takes 16.81 ms
# 256 x 256 x 256: numpy takes  0.20 ms
# 512 x 512 x 512: hidet takes 111.15 ms
# 512 x 512 x 512: numpy takes  0.57 ms
# 1024 x 1024 x 1024: hidet takes 889.30 ms
# 1024 x 1024 x 1024: numpy takes  2.55 ms
# 1024 x 768 x 512: hidet takes 332.83 ms
# 1024 x 768 x 512: numpy takes  1.10 ms
# 333 x 555 x 444: hidet takes 68.46 ms
# 333 x 555 x 444: numpy takes  0.81 ms
# 480 x 480 x 480: hidet takes 91.30 ms
# 480 x 480 x 480: numpy takes  0.94 ms
# 720 x 720 x 720: hidet takes 309.26 ms
# 720 x 720 x 720: numpy takes  2.32 ms
# 720 x 1440 x 960: hidet takes 822.23 ms
# 720 x 1440 x 960: numpy takes  4.66 ms

###### -O3:
# 256 x 256 x 256: hidet takes 1.62 ms
# 256 x 256 x 256: numpy takes  0.23 ms
# 512 x 512 x 512: hidet takes 13.41 ms
# 512 x 512 x 512: numpy takes  0.66 ms
# 1024 x 1024 x 1024: hidet takes 182.31 ms
# 1024 x 1024 x 1024: numpy takes  2.29 ms
# 1024 x 768 x 512: hidet takes 36.49 ms
# 1024 x 768 x 512: numpy takes  1.14 ms
# 333 x 555 x 444: hidet takes 7.55 ms
# 333 x 555 x 444: numpy takes  0.47 ms
# 480 x 480 x 480: hidet takes 10.13 ms
# 480 x 480 x 480: numpy takes  0.89 ms
# 720 x 720 x 720: hidet takes 28.35 ms
# 720 x 720 x 720: numpy takes  1.46 ms
# 720 x 1440 x 960: hidet takes 74.57 ms
# 720 x 1440 x 960: numpy takes  4.57 ms


#### -O2:
# 256 x 256 x 256: hidet takes 1.69 ms
# 256 x 256 x 256: numpy takes  0.17 ms
# 512 x 512 x 512: hidet takes 13.53 ms
# 512 x 512 x 512: numpy takes  0.66 ms
# 1024 x 1024 x 1024: hidet takes 181.54 ms
# 1024 x 1024 x 1024: numpy takes  2.40 ms
# 1024 x 768 x 512: hidet takes 36.49 ms
# 1024 x 768 x 512: numpy takes  1.17 ms
# 333 x 555 x 444: hidet takes 7.96 ms
# 333 x 555 x 444: numpy takes  0.47 ms
# 480 x 480 x 480: hidet takes 10.24 ms
# 480 x 480 x 480: numpy takes  0.61 ms
# 720 x 720 x 720: hidet takes 28.71 ms
# 720 x 720 x 720: numpy takes  2.18 ms
# 720 x 1440 x 960: hidet takes 74.16 ms
# 720 x 1440 x 960: numpy takes  3.25 ms