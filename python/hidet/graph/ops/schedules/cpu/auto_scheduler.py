from typing import List, Dict

from hidet.ir.builders import FunctionBuilder
from hidet.ir.compute import TensorNode, GridCompute
from hidet.ir.expr import Call, Expr, Var, convert
from hidet.ir.functors import collect, rewrite
from hidet.ir.stmt import Stmt, BufferStoreStmt, EvaluateStmt
from ..auto_scheduler import AutoScheduler, ComputeExprLower


class CpuAutoScheduler(AutoScheduler):
    def __init__(self):
        super().__init__()

    def schedule_grid_compute(self, gc: GridCompute, node: TensorNode, node_map: Dict[TensorNode, Expr]) -> Stmt:
        from hidet.ir.mapping import row_repeat, TaskMapping
        used_tensors: List[TensorNode] = collect(gc.value, TensorNode, stop_when_found=True)
        param_tensors: List[TensorNode] = used_tensors + [node]
        params: List[Var] = [Var(tensor.name, tensor.data_type) for tensor in param_tensors]

        with FunctionBuilder(name=f'compute_{node.name}', kind='host_kernel') as fb:
            # set function parameters
            fb.extend_params(params)

            mapping: TaskMapping = row_repeat(*gc.shape)
            iter_names = [f'i{i}' for i in range(len(gc.shape))]
            with fb.for_mapping(iter_names, mapping, convert(0)) as task_index:
                out_param: Var = params[-1]
                param_map: Dict[TensorNode, Expr] = {tensor_node: param_var for tensor_node, param_var in zip(param_tensors, params)}
                compute_lower = ComputeExprLower(gc.value, param_map=param_map)
                stmts, value = compute_lower.lower()
                rmap = {axis: axis_value for axis, axis_value in zip(gc.axes, task_index)}
                stmts, value = [rewrite(stmt, rmap) for stmt in stmts], rewrite(value, rmap)
                fb += stmts
                fb += BufferStoreStmt(out_param, task_index, value)
        func = fb.get()
        func_var = self.add_function(func)
        return EvaluateStmt(Call(func_var, args=[node_map[param_tensor] for param_tensor in param_tensors]))
