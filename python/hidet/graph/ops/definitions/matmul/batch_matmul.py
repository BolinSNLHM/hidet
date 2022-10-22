from hidet.ir.func import IRModule
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, TensorNode, compute, reduce, input_like
from hidet.graph.ops.definitions.arithmatic import broadcast_shape
from hidet.graph.ops.definitions.transform import broadcast, unsqueeze, transpose
from hidet.ffi import cuda


class BatchMatmulTask(Task):
    def __init__(self, a: TensorNode, b: TensorNode, mma: str = 'simt'):
        batch_size, m_size, k_size = a.const_shape()
        batch_size, k_size, n_size = b.const_shape()
        self.batch_size: int = batch_size
        self.m_size: int = m_size
        self.k_size: int = k_size
        self.n_size: int = n_size
        self.mma: str = mma
        c = compute(
            name='c',
            shape=[batch_size, m_size, n_size],
            fcompute=lambda r, i, j: reduce(
                shape=[k_size],
                fcompute=lambda k: a[r, i, k] * b[r, k, j],
                reduce_type='sum'
            ),
        )
        super().__init__(
            name='matmul',
            inputs=[a, b],
            outputs=[c],
            attributes={
                'batch_size': batch_size,
                'm_size': m_size,
                'n_size': n_size,
                'k_size': k_size,
                'mma': mma
            }
        )

    def implement_cuda(self) -> IRModule:
        from hidet.graph.ops.schedules.cuda.matmul import batched_matmul_cuda_schedule_simt, batched_matmul_cuda_schedule_wmma, batched_matmul_cuda_schedule_mma
        if self.mma == 'simt':
            return batched_matmul_cuda_schedule_simt(self)
        elif self.mma.startswith('wmma'):
            return batched_matmul_cuda_schedule_wmma(self)
        elif self.mma.startswith('mma'):
            return batched_matmul_cuda_schedule_mma(self)
        else:
            raise ValueError('Can not recognize mma type {}, candidates: {}'.format(self.mma, ['simt', 'wmma', 'mma']))


class BatchMatmulOp(Operator):
    def __init__(self, a: Tensor, b: Tensor, algo, mma: str = 'simt'):
        if not (len(a.shape) == len(b.shape) == 3 and a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1]):
            raise ValueError('Matrix multiplication expect tensor A and B with shape [B, M, K] and [B, K, N]' +
                             ', got {} and {}'.format(a.shape, b.shape))
        task = BatchMatmulTask(input_like(a, 'a'), input_like(b, 'b'), mma)
        super().__init__(
            inputs=[a, b],
            task=task,
            attributes={
                'algo': algo,
                'mma': mma
            }
        )


def batch_matmul(a: Tensor, b: Tensor, mma: str = 'simt') -> Tensor:
    """Batched matrix multiplication.

    Parameters
    ----------
    a: Tensor
        The lhs operand with shape [batch_size, m_size, k_size].

    b: Tensor
        The rhs operand with shape [batch_size, k_size, n_size].

    mma: str
        The matrix-multiplication-accumulate (mma) in warp level:

        - 'simt':
           Use cuda core to do the warp-level mma (simt stands for single-instruction-multiple-threads).
        - 'wmma':
           Use wmma instruction.
        - 'mma':
           Use mma instruction.

        See also:
        https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions

    Returns
    -------
    c: Tensor
        The result tensor of matrix multiplication.
    """
    return BatchMatmulOp(a, b, mma).get_output(0)

    # if len(a.shape) < 2 or len(b.shape) < 2:
    #     raise ValueError('Current only support matrix multiplication with two matrices whose rank >= 2.')
    #
    # if len(a.shape) == 2 and len(b.shape) == 2:
    #     if ta:
    #         aa = a.transpose([-1, -2]).barrier().transpose([-1, -2])
    #     else:
    #         aa = a
    #     if tb:
    #         bb = b.transpose([-1, -2]).barrier().transpose([-1, -2])
    #     else:
    #         bb = b
    #     aa = aa.unsqueeze(0)
    #     bb = bb.unsqueeze(0)
    #     cc = batch_matmul_internal(aa, bb, algo, mma, ta, tb, tc)
    #     cc = cc.squeeze(0)
    #     if tc:
    #         cc = cc.transpose([-1, -2]).barrier().transpose([-1, -2])
    #     return cc
    # else:
    #     if ta:
    #         aa = a.transpose([-1, -2]).barrier().transpose([-1, -2])
    #     else:
    #         aa = a
    #     if tb:
    #         bb = b.transpose([-1, -2]).barrier().transpose([-1, -2])
    #     else:
    #         bb = b
    #     stack_shape = broadcast_shape(aa.shape[:-2], bb.shape[:-2])
    #     aa = broadcast(aa, shape=stack_shape + a.shape[-2:]).flatten(end_dim=-2)
    #     bb = broadcast(bb, shape=stack_shape + b.shape[-2:]).flatten(end_dim=-2)
    #     cc = batch_matmul_internal(aa, bb, algo, mma, ta, tb, tc)
    #     if tc:
    #         cc = cc.transpose([-1, -2]).barrier().transpose([-1, -2])
    #     return cc


# def batch_matmul_internal(a: Tensor, b: Tensor, algo: str = 'default', mma: str = 'default', ta=True, tb=False, tc=False) -> Tensor:
#     mma_candidates = [
#         'default', 'simt', 'wmma', 'mma',
#         'wmma_f16_f16', 'wmma_f16_f32', 'wmma_bf16_f32', 'wmma_tf32_f32',
#         'mma_f16_f16', 'mma_f16_f32', 'mma_bf16_f32', 'mma_tf32_f32',
#         'mma_custom'
#     ]
#     algo_candidates = ['default', 'direct', 'parallel_k']
#     if mma not in mma_candidates:
#         raise ValueError('Can not recognize mma {}, candidates: {}'.format(mma, mma_candidates))
#     if algo not in algo_candidates:
#         raise ValueError('Can not recognize algorithm {}, candidates: {}'.format(algo, algo_candidates))
#     return BatchMatmulOp(a, b, algo, mma, ta, tb, tc).get_output(0)