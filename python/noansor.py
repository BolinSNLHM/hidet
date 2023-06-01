import numpy as np

import hidet
from hidet.graph.ops import matmul_x86
from hidet.option import debug_cache_tuning

import tvm
from tvm import te, auto_scheduler

@auto_scheduler.register_workload
def matmul_ansor(M, K, N, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)
    C = te.placeholder((M, N), name="C", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    rst = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="matmul_ansor",
        attrs={"layout_free_placeholders": [B]},        # Enable automatic layout transform for B TODO: What is this?
    )

    return [A, B, rst]

target = tvm.target.Target("llvm -mcpu=core-avx2")


debug_cache_tuning(True)
hidet.option.search_space(0)
hidet.option.parallel_build(True)
for m, n, k in [(2048, 2048, 2048), (2047, 2047, 2047), (2046, 2046, 2046), (2045, 2045, 2045), (2044, 2044, 2044),
                (2043, 2043, 2043), (2042, 2042, 2042)]:
    a = hidet.randn([m, k], device='cpu')
    b = hidet.randn([k, n], device='cpu')
    x1 = hidet.symbol_like(a)
    x2 = hidet.symbol_like(b)
    y = matmul_x86(x1, x2)
    graph: hidet.FlowGraph = hidet.trace_from(y, inputs=[x1, x2])
    opt_graph = hidet.graph.optimize(graph)
    compiled_func = opt_graph.nodes[0].task_func

    c = hidet.zeros([m, n], device='cpu')

    compiled_func(a, b, c)

    c2 = hidet.zeros([m, n], device='cpu')
    compiled_func(a, b, c2)

    c3 = hidet.zeros([m, n], device='cpu')
    compiled_func(a, b, c3)

    try:
        np.testing.assert_allclose(
            actual=c.numpy(),
            desired=a.numpy() @ b.numpy(),
            rtol=1e-3,
            atol=1e-3
        )

        np.testing.assert_allclose(
            actual=c2.numpy(),
            desired=a.numpy() @ b.numpy(),
            rtol=1e-3,
            atol=1e-3
        )

        np.testing.assert_allclose(
            actual=c3.numpy(),
            desired=a.numpy() @ b.numpy(),
            rtol=1e-3,
            atol=1e-3
        )

        print("m={} n={} k={}: PASS".format(m, n, k))

    except AssertionError as e:
        for i in range(m):
            for j in range(n):
                if abs((c.numpy())[i, j] - (a.numpy() @ b.numpy())[i, j]) > 1e-3:
                    print(i, j, (c.numpy())[i, j], (a.numpy() @ b.numpy())[i, j])

    hidet_latency = hidet.utils.benchmark_func(
        lambda: compiled_func(a, b, c), repeat=100
    )
    np_latency = hidet.utils.benchmark_func(
        lambda: a.numpy() @ b.numpy(), repeat=100
    )


    with open(f"./perf_opt-packing.txt", 'a+') as f:
        print(f'm={m}, k={k}, n={n}: hidet takes {hidet_latency:.2f} ms\n')
        print(f'm={m}, k={k}, n={n}: numpy takes {np_latency:.2f} ms\n')
        print(f'm={m}, k={k}, n={n}: hidet achieves {np_latency / hidet_latency * 100:.2f}% of numpy efficiency\n')



