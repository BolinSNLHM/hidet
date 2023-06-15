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
hidet.option.search_space(2)
hidet.option.cache_dir("./cache-branchISfused-realistic-inputs")
hidet.option.parallel_build(True)
# for m, k, n in [(1024, 768, 3072), (1024, 768, 768)]:
# for m, n, k in [(128, 768, 768), (1024, 768, 3072), (1024, 768, 768)]:
# for m, n, k in [(128, 768, 768)]:
for m, n, k in [(768, 768, 768), (1024, 768, 768), (128, 3072, 768)]:
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

    try:
        np.testing.assert_allclose(
            actual=c.numpy(),
            desired=a.numpy() @ b.numpy(),
            rtol=1e-3,
            atol=1e-3
        )
        print("passed for m={}, n={}, k={}".format(m, n, k))
    except AssertionError as e:
        raise e

    hidet_latency = hidet.utils.benchmark_func(
        lambda: compiled_func(a, b, c), repeat=100
    )

    np_latency = hidet.utils.benchmark_func(
        lambda: a.numpy() @ b.numpy(), repeat=100
    )

    ansor_task = tvm.auto_scheduler.SearchTask(func=matmul_ansor, args=(m, k, n, "float32"), target=target)
    log_file = f"./temps/AB-matmul_{m}x{k}x{n}.json"
    # tune_option = auto_scheduler.TuningOptions(
    #     num_measure_trials=1000,
    #     measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    #     verbose=2,
    # )
    #
    # ansor_task.tune(tune_option)
    sch, args = ansor_task.apply_best(log_file)

    with open(f"./temps/script-matmul_AB_TIR_{m}x{k}x{n}", 'w') as f:
        f.write(str(tvm.lower(sch, args, simple_mode=True).script()))
    ansor_func = tvm.build(sch, args, target)
    dev = tvm.cpu()
    a_tvm = tvm.nd.array(a.numpy(), device=dev)
    b_tvm = tvm.nd.array(b.numpy(), device=dev)
    c_tvm = tvm.nd.empty((m, n), device=dev)

    ansor_func(a_tvm, b_tvm, c_tvm)

    np.testing.assert_allclose(
        actual=c_tvm.numpy(),
        desired=a_tvm.numpy() @ b_tvm.numpy(),
        rtol=1e-3,
        atol=1e-3
    )

    ansor_latency = hidet.utils.benchmark_func(
        lambda: ansor_func(a_tvm, b_tvm, c_tvm), repeat=100
    )

    with open(f"./temps/perf-branchISfused-realistic-inputs.txt", 'a+') as f:
        f.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')
        f.write(f'm={m}, k={k}, n={n}: hidet takes {hidet_latency:.2f} ms\n')
        f.write(f'm={m}, k={k}, n={n}: numpy takes {np_latency:.2f} ms\n')
        f.write(f'm={m}, k={k}, n={n}: ansor takes {ansor_latency:.2f} ms\n')
        f.write(f'm={m}, k={k}, n={n}: hidet achieves {np_latency / hidet_latency * 100:.2f}% of numpy efficiency\n')
        f.write(f'm={m}, k={k}, n={n}: hidet achieves {ansor_latency / hidet_latency * 100:.2f}% of ansor efficiency\n')
        f.write('\n')



