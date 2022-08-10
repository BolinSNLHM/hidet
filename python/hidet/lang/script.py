from __future__ import annotations
from typing import Callable, Tuple, Optional, List, Any, Dict
from types import FunctionType
import ast as py_ast
import inspect
from hidet.ir import IRModule, Function, Var, FuncType
from .transpiler import PythonToHidetTranslator


def eliminate_indent(source: str) -> Tuple[str, int]:
    lines = source.split('\n')
    indent = len(source)
    for line in lines:
        if len(line.strip()) == 0:
            continue
        indent = min(indent, len(line) - len(line.lstrip()))
    source = '\n'.join([line[indent:] for line in lines])
    return source, indent


def eliminate_decorators(source: str) -> Tuple[str, int]:
    lines = source.split('\n')
    num_decorators = 0
    for line in lines:
        if len(line) > 0 and line[0] == '@':
            num_decorators += 1
        else:
            break
    source = '\n'.join(lines[num_decorators:])
    return source, num_decorators


def script(func: FunctionType) -> Function:
    # Extract the source code of given function
    lines, start_line = inspect.getsourcelines(func)
    file = inspect.getsourcefile(func)
    source = ''.join(lines)
    source, col_offset = eliminate_indent(source)
    source, inc_lineno = eliminate_decorators(source)
    start_line += inc_lineno
    parsed: py_ast.AST = py_ast.parse(source=source)

    # Get the environment (binding of free variables)
    # See the data model of python for the details of func.__closure__ and func.__code__:
    #     https://docs.python.org/3/reference/datamodel.html
    func_freevar_names: List[str] = list(func.__code__.co_freevars)
    func_freevar_cells: List[Any] = [v.cell_contents for v in func.__closure__] if func.__closure__ else []
    assert len(func_freevar_names) == len(func_freevar_cells)
    env: Dict[str, Any] = {name: value for name, value in zip(func_freevar_names, func_freevar_cells)}
    func_annotations: Dict[str, Any] = func.__annotations__

    # Translate the Python function into Hidet function
    translator = PythonToHidetTranslator(
        file=file,
        start_lineno=start_line,
        start_column=col_offset,
        env=env,
        func_annotations=func_annotations
    )
    hidet_function = translator(parsed)

    # add function to current script module
    ctx = ScriptModuleContext.current_context()
    ctx.append_function(hidet_function)
    return hidet_function


class ScriptModuleContext:
    contexts: List[ScriptModuleContext] = []

    def __init__(self):
        self.name2var: Dict[str, Var] = {}
        self.functions: List[Function] = []

    def __enter__(self):
        self.contexts.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.contexts.pop()

    @staticmethod
    def current_context() -> ScriptModuleContext:
        if len(ScriptModuleContext.contexts) == 0:
            msg = (
                'Can only define script function in script module:\n\n'
                'with hidet.script_module() as module:\n'
                '    @hidet.script\n'
                '    def kernel_function():\n'
                '        ...\n'
            )
            raise ValueError(msg)
            # add the fallback context
        return ScriptModuleContext.contexts[-1]

    def append_function(self, function: Function):
        self.functions.append(function)
        self.name2var[function.name] = Var(hint=function.name, type=FuncType.from_func(function))

    def lookup(self, name: str) -> Optional[Var]:
        if name not in self.name2var:
            return None
        return self.name2var[name]

    def ir_module(self) -> IRModule:
        return IRModule(
            funcs={func.name: func for func in self.functions},
            task=None,
            global_vars=self.name2var
        )


def script_module() -> ScriptModuleContext:
    return ScriptModuleContext()