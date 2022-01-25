from typing import Dict, Callable, Set, Union, Optional, Tuple
from hidet.ir.type import FuncType, ScalarType
from hidet.ir.expr import Var, Call
from hidet.ir.stmt import AsmStmt
from hidet.ir.func import Function
from hidet.ir.task import Thread
from hidet.ir.dialects.lowlevel import VoidType, PointerType, ReferenceType
from hidet.ir.builders import FunctionBuilder

_primitive_functions: Dict[str, Tuple[Var, FuncType, Optional[Function]]] = {}


def is_primitive_function(name):
    return name in _primitive_functions


def get_primitive_function(name: str) -> Tuple[Var, FuncType, Optional[Function]]:
    assert name in _primitive_functions
    return _primitive_functions[name]


def register_primitive_function(name, func_or_ftype: Union[Function, FuncType]):
    if isinstance(func_or_ftype, Function):
        func = func_or_ftype
        func_type = FuncType.from_func(func)
    elif isinstance(func_or_ftype, FuncType):
        func = None
        func_type = func_or_ftype
    else:
        raise False
    v = Var(name, func_type)
    assert name not in _primitive_functions
    _primitive_functions[name] = (v, func_type, func)


def syncthreads() -> Call:
    if '__syncthreads' not in _primitive_functions:
        register_primitive_function('__syncthreads', FuncType([], VoidType()))
    func_var = get_primitive_function('__syncthreads')[0]
    return Call(func_var, [])


def lds128(reg0, reg1, reg2, reg3, smem_addr) -> Call:
    if 'lds128' not in _primitive_functions:
        with FunctionBuilder('lds128', attrs={'worker': Thread()}) as fb:
            # params
            regs_vars = [Var(f'reg{i}', ReferenceType(ScalarType('float32'))) for i in range(4)]
            smem_addr_var = Var('smem_addr', PointerType(ScalarType('float32')))
            fb.extend_params(regs_vars + [smem_addr_var])
            # body
            body = AsmStmt(
                r"{"
                r"  .reg.u64 u64addr;"
                r"  cvta.to.shared.u64 u64addr, %4;"
                r"  ld.shared.v4.f32 {%0, %1, %2, %3}, [u64addr];"
                r"}",
                outputs=[('=f', reg) for reg in regs_vars],
                inputs=[('l', smem_addr_var)],
                is_volatile=True
            )
            fb.set_body(body)
        register_primitive_function('lds128', fb.get())
    func_var = get_primitive_function('lds128')[0]
    return Call(func_var, [reg0, reg1, reg2, reg3, smem_addr])


def sts128(reg0, reg1, reg2, reg3, smem_addr) -> Call:
    if 'sts128' not in _primitive_functions:
        with FunctionBuilder('sts128', attrs={'worker': Thread()}) as fb:
            # params
            regs_vars = [Var(f'reg{i}', ReferenceType(ScalarType('float32'))) for i in range(4)]
            smem_addr_var = Var('smem_addr', PointerType(ScalarType('float32')))
            fb.extend_params(regs_vars + [smem_addr_var])
            # body
            body = AsmStmt(
                r"{" 
                r"  .reg.u64 u64addr;"
                r"  cvta.to.shared.u64 u64addr, %0;"
                r"  st.shared.v4.f32 [u64addr], {%1, %2, %3, %4};"
                r"}",
                outputs=[],
                inputs=[('l', smem_addr_var)] + [('f', reg) for reg in regs_vars],
                is_volatile=True
            )
            fb.set_body(body)
        register_primitive_function('sts128', fb.get())
    func_var = get_primitive_function('sts128')[0]
    return Call(func_var, [reg0, reg1, reg2, reg3, smem_addr])