from hidet.ir.type import TypeNode
from hidet.ir.expr import Expr, TensorElement, Var


class VoidType(TypeNode):
    pass


class PointerType(TypeNode):
    def __init__(self, base_type):
        super().__init__()
        self.base_type = base_type


class ReferenceType(TypeNode):
    def __init__(self, base_type):
        super().__init__()
        self.base_type = base_type


class Cast(Expr):
    def __init__(self, expr, target_type):
        self.expr = expr
        self.target_type = target_type


class Dereference(Expr):
    def __init__(self, expr):
        self.expr = expr


class Address(Expr):
    def __init__(self, expr):
        self.expr = expr


class Reference(Expr):
    def __init__(self, expr):
        assert isinstance(expr, (TensorElement, Var)), "only l-value can be referenced."
        self.expr = expr


def pointer_type(base_type):
    return PointerType(base_type)